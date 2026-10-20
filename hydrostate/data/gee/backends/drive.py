"""Drive export transport, verified downloads and explicit artifact cleanup."""

import hashlib
import os
from pathlib import Path


def drive_client(args):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    credentials = Credentials.from_authorized_user_file(args.drive_token)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials.valid:
        raise RuntimeError("Drive OAuth credentials are not valid")
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def list_files(drive, query):
    found, token = [], None
    while True:
        result = (
            drive.files()
            .list(
                q=query,
                fields="nextPageToken,files(id,name,size,md5Checksum)",
                pageToken=token,
                pageSize=1000,
            )
            .execute(num_retries=5)
        )
        found.extend(result.get("files", []))
        token = result.get("nextPageToken")
        if not token:
            return found


def download(drive, remote, path):
    from googleapiclient.http import MediaIoBaseDownload

    def valid(p):
        if not p.exists() or p.stat().st_size != int(remote["size"]):
            return False
        with p.open("rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest() == remote.get("md5Checksum")

    if valid(path):
        return
    temp = Path(str(path) + ".part")
    with temp.open("wb") as handle:
        dl = MediaIoBaseDownload(
            handle, drive.files().get_media(fileId=remote["id"]), chunksize=8 * 1024**2
        )
        done = False
        while not done:
            _, done = dl.next_chunk(num_retries=5)
        handle.flush()
        os.fsync(handle.fileno())
    if not valid(temp):
        raise RuntimeError(f"Download checksum/size mismatch: {path}")
    os.replace(temp, path)


class IncompleteExport(RuntimeError):
    """A completed task's files are not all visible or locally available yet."""


def resume_download(session, remote, path, *, chunk_bytes=32 * 1024**2):
    """Download immutable Drive blobs with persistent HTTP Range recovery."""
    import json
    import time

    from hydrostate.data.gee.contracts import atomic_json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = int(remote["size"])
    checksum = remote.get("md5Checksum")
    if not checksum:
        raise ValueError("Drive export has no MD5 checksum")

    def verified(candidate):
        if not candidate.exists() or candidate.stat().st_size != expected:
            return False
        with candidate.open("rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest() == checksum

    if verified(path):
        return
    part = Path(str(path) + ".part")
    receipt = Path(str(part) + ".json")
    identity = {k: remote[k] for k in ("id", "size", "md5Checksum")}
    if not receipt.exists() or json.loads(receipt.read_text()) != identity:
        part.unlink(missing_ok=True)
        atomic_json(receipt, identity)
    if part.exists() and part.stat().st_size >= expected:
        if verified(part):
            os.replace(part, path)
            receipt.unlink(missing_ok=True)
            return
        part.unlink()
    offset = part.stat().st_size if part.exists() else 0
    with part.open("ab") as output:
        while offset < expected:
            end = min(expected, offset + chunk_bytes) - 1
            for attempt in range(5):
                response = None
                try:
                    response = session.get(
                        f"https://www.googleapis.com/drive/v3/files/{remote['id']}",
                        params={"alt": "media"},
                        headers={"Range": f"bytes={offset}-{end}"},
                        stream=True,
                        timeout=(30, 180),
                    )
                    if response.status_code in (429, 500, 502, 503, 504):
                        response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError(f"Drive did not honor Range: HTTP {response.status_code}")
                    required = f"bytes {offset}-{end}/{expected}"
                    if response.headers.get("Content-Range") != required:
                        raise ValueError("Drive Content-Range changed or is incomplete")
                    received = 0
                    for block in response.iter_content(chunk_size=1024**2):
                        output.write(block)
                        received += len(block)
                    output.flush()
                    if received != end - offset + 1:
                        raise OSError("Truncated Drive byte range")
                    os.fsync(output.fileno())
                    offset = end + 1
                    break
                except Exception:
                    output.flush()
                    output.truncate(offset)
                    if attempt == 4:
                        raise
                    time.sleep(min(2**attempt, 16))
                finally:
                    if response is not None:
                        response.close()
    if not verified(part):
        part.unlink()
        raise ValueError("Drive checksum/size mismatch; unverified file retained remotely")
    os.replace(part, path)
    receipt.unlink(missing_ok=True)


class DriveTransport:
    def __init__(self, ee, root, settings, *, token=None, quota_project=None):
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        self.ee, self.root, self.settings = ee, Path(root), settings
        self.credentials = (
            Credentials.from_authorized_user_file(token)
            if token
            else ee.data.get_persistent_credentials()
        )
        if quota_project:
            self.credentials = self.credentials.with_quota_project(quota_project)
        self.service = build("drive", "v3", credentials=self.credentials, cache_discovery=False)
        from hydrostate.data.gee.contracts import fingerprint

        self.folder_name = "HydroState_region_" + fingerprint(str(self.root.resolve()))[:16]
        self.folder_id = None

    def available_bytes(self):
        quota = (
            self.service.about().get(fields="storageQuota").execute(num_retries=5)["storageQuota"]
        )
        return (
            int(quota["limit"]) - int(quota.get("usage", 0)) if quota.get("limit") else float("inf")
        )

    def ensure_folder(self):
        folders = list_files(
            self.service,
            f"name='{self.folder_name}' "
            "and mimeType='application/vnd.google-apps.folder' and trashed=false",
        )
        if len(folders) > 1:
            raise RuntimeError("Ambiguous regional Drive folder name")
        self.folder_id = (
            folders[0]["id"]
            if folders
            else self.service.files()
            .create(
                body={"name": self.folder_name, "mimeType": "application/vnd.google-apps.folder"},
                fields="id",
            )
            .execute(num_retries=5)["id"]
        )

    @staticmethod
    def prefix(job_id):
        return "hsr_" + job_id

    def submit(self, job, task_id):
        from hydrostate.data.gee.products import scene_expression

        ee = self.ee
        if job.get("layout") == "native_v1":
            from affine import Affine

            from hydrostate.data.gee.native import native_expression

            tr = Affine(*job["transform"])
            corners = [tr * p for p in [
                (0, 0), (job["width"], 0),
                (job["width"], job["height"]), (0, job["height"]), (0, 0)
            ]]
            area = ee.Geometry.Polygon([corners], job["crs"], False)
            task = ee.batch.Export.image.toDrive(
                image=native_expression(ee, job),
                description=self.prefix(job["id"]),
                folder=self.folder_name,
                fileNamePrefix=self.prefix(job["id"]),
                region=area, crs=job["crs"], crsTransform=job["transform"],
                maxPixels=job["width"] * job["height"] + 1,
                fileFormat="GeoTIFF",
                fileDimensions=self.settings["file_dimensions"], shardSize=256,
                skipEmptyTiles=False,
            )
            task._request_id = task_id
            task.start()
            return task.id
        left, top = job["transform"][2], job["transform"][5]
        area = ee.Geometry.Rectangle(
            [left, top - job["height"] * 10, left + job["width"] * 10, top],
            job["grid"]["crs"],
            False,
        )
        task = ee.batch.Export.image.toDrive(
            image=scene_expression(ee, job["scene"]),
            description=self.prefix(job["id"]),
            folder=self.folder_name,
            fileNamePrefix=self.prefix(job["id"]),
            region=area,
            crs=job["grid"]["crs"],
            crsTransform=job["transform"],
            maxPixels=job["width"] * job["height"] + 1,
            fileFormat="GeoTIFF",
            fileDimensions=self.settings["file_dimensions"],
            shardSize=256,
            # Explicit zero-valid pixels make missing files distinguishable from empty regions.
            skipEmptyTiles=False,
        )
        task._request_id = task_id
        task.start()
        return task.id

    def status(self, task_id):
        return self.ee.data.getTaskStatus(task_id)[0]

    def reconcile(self, job_id):
        matches = [
            t
            for t in self.ee.batch.Task.list()
            if t.config.get("description") == self.prefix(job_id)
        ]
        if len(matches) > 1:
            raise RuntimeError("Multiple remote tasks match one export receipt")
        return matches[0].id if matches else None

    def inventory(self, job_id):
        prefix = self.prefix(job_id)
        files = list_files(
            self.service,
            f"'{self.folder_id}' in parents and name contains '{prefix}' and trashed=false",
        )
        files = [
            f
            for f in files
            if (f["name"] == prefix + ".tif" or f["name"].startswith(prefix + "-"))
            and f["name"].endswith(".tif")
        ]
        if len({f["name"] for f in files}) != len(files):
            raise RuntimeError("Duplicate names in a regional Drive export")
        if any(not f.get("size") or not f.get("md5Checksum") for f in files):
            raise IncompleteExport("Export files are not fully published")
        return sorted(files, key=lambda f: f["name"])

    def local_path(self, job_id, remote):
        export_id = remote.get("export_id", job_id)
        return self.root / "staging" / "drive" / export_id / (remote["id"] + ".tif")

    def fetch(self, job_id, files):
        from concurrent.futures import ThreadPoolExecutor

        from google.auth.transport.requests import AuthorizedSession

        def fetch_one(remote):
            path = self.local_path(job_id, remote)
            with AuthorizedSession(self.credentials) as session:
                resume_download(session, remote, path)
            return path

        with ThreadPoolExecutor(max_workers=self.settings["download_workers"]) as pool:
            return list(pool.map(fetch_one, files))

    def cleanup(self, receipt):
        from googleapiclient.errors import HttpError

        if receipt["status"] != "committed":
            raise RuntimeError("Refusing to delete exports before cache commit")
        allowed = {receipt["id"], *(p["id"] for p in receipt["payload"].get("parts", []))}
        for remote in receipt["files"]:
            export_id = remote.get("export_id", receipt["id"])
            if export_id not in allowed:
                raise RuntimeError("Refusing deletion outside recorded native export parts")
            prefix = self.prefix(export_id)
            try:
                actual = (
                    self.service.files()
                    .get(fileId=remote["id"], fields="id,name,parents")
                    .execute(num_retries=5)
                )
                if (
                    self.folder_id not in actual.get("parents", [])
                    or actual["name"] != remote["name"]
                ):
                    raise RuntimeError("Refusing deletion of a file outside the recorded export")
                if not (
                    actual["name"] == prefix + ".tif" or actual["name"].startswith(prefix + "-")
                ):
                    raise RuntimeError("Refusing deletion of an unrelated Drive file")
                self.service.files().delete(fileId=remote["id"]).execute(num_retries=5)
            except HttpError as exc:
                if exc.resp.status != 404:
                    raise
        for remote in receipt["files"]:
            self.local_path(receipt["id"], remote).unlink(missing_ok=True)


def ingest_export(cache, job, paths):
    """Validate the entire export coverage before committing any new scene chunks."""
    if job.get("layout") == "native_v1":
        from hydrostate.data.gee.native import ingest_native

        return ingest_native(cache, job, paths)
    from contextlib import ExitStack

    import numpy as np
    import rasterio
    from rasterio.windows import Window

    from hydrostate.data.gee.contracts import input_shape

    if job["contract"] != cache.contract or job["chunk"] != cache.chunk:
        raise ValueError("Export does not match this cache contract")
    with ExitStack() as stack:
        rasters = [stack.enter_context(rasterio.open(p)) for p in paths]
        areas = []
        for src in rasters:
            if src.crs != rasterio.crs.CRS.from_string(job["grid"]["crs"]):
                raise ValueError("Export CRS mismatch")
            if src.count != len(job["bands"]):
                raise ValueError("Export band count mismatch")
            if any(src.descriptions) and list(src.descriptions) != job["bands"]:
                raise ValueError("Export band identity/order mismatch")
            tr = src.transform
            if not np.allclose([tr.a, tr.b, tr.d, tr.e], [10, 0, 0, -10], atol=1e-7, rtol=0):
                raise ValueError("Export pixel geometry mismatch")
            col = (tr.c - job["transform"][2]) / 10
            row = (job["transform"][5] - tr.f) / 10
            if abs(col - round(col)) > 1e-5 or abs(row - round(row)) > 1e-5:
                raise ValueError("Export pixel origin is not aligned")
            col, row = round(col), round(row)
            if (
                min(col, row) < 0
                or col + src.width > job["width"]
                or row + src.height > job["height"]
            ):
                raise ValueError("Export extent exceeds its request")
            bounds = (col, row, col + src.width, row + src.height)
            for a, b, c, d in areas:
                if min(bounds[2], c) > max(col, a) and min(bounds[3], d) > max(row, b):
                    raise ValueError("Overlapping/duplicate export files")
            areas.append(bounds)
        if sum(src.width * src.height for src in rasters) != job["width"] * job["height"]:
            raise IncompleteExport("Missing export parts; no cache chunks were committed")
        done = cache.done(job["scene"])
        h, w = input_shape(cache.grid)
        for cy, cx in job["chunks"]:
            if done[cy, cx]:
                continue
            y, x = cy * cache.chunk, cx * cache.chunk
            height, width = min(cache.chunk, h - y), min(cache.chunk, w - x)
            local_x, local_y = x - job["x"], y - job["y"]
            cover = [
                (src, bounds)
                for src, bounds in zip(rasters, areas)
                if bounds[0] <= local_x
                and bounds[1] <= local_y
                and bounds[2] >= local_x + width
                and bounds[3] >= local_y + height
            ]
            if len(cover) != 1:
                raise ValueError("Export part boundaries must align to cache chunks")
            src, bounds = cover[0]
            raw = src.read(window=Window(local_x - bounds[0], local_y - bounds[1], width, height))
            if not np.isfinite(raw).all() or not np.isin(raw[-1], [0, 1]).all():
                raise ValueError("Invalid pixel values or validity band")
            cache.write_chunk(job["scene"], cy, cx, raw[:-1], raw[-1] > 0)
        final = cache.done(job["scene"])
        if any(not final[cy, cx] for cy, cx in job["chunks"]):
            raise RuntimeError("Cache commit verification failed")
