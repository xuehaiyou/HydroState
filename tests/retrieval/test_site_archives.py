import json
import zipfile

import pytest

from hydrostate.data.site.archives import extract, inventory


def test_permanent_nested_extraction_and_resume(tmp_path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("station/data.csv", "time,value\n20200101,0.2\n")
    outer = tmp_path / "source.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.write(inner, "network/inner.zip")
    assert inventory([outer])[0]["nested_archives"] == 1
    record = extract(outer)
    csv = tmp_path / "source/network/inner/station/data.csv"
    assert csv.read_text().endswith("0.2\n")
    before = csv.stat().st_mtime_ns
    assert extract(outer)["sha256"] == record["sha256"]
    assert csv.stat().st_mtime_ns == before
    assert json.loads((tmp_path / "source/_EXTRACT_SUCCESS.json").read_text())["complete"]


def test_reject_traversal_and_existing_directory(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        inventory([path])
    with pytest.raises(ValueError, match="Unsafe"):
        extract(path)
    assert not (tmp_path / "escape").exists()
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as archive:
        archive.writestr("data", "value")
    (tmp_path / "good").mkdir()
    with pytest.raises(FileExistsError, match="Unowned"):
        extract(good)
