#!/usr/bin/env bash
# Re-run with the same snapshot to resume; a new snapshot reuses the remote Zarr pool.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

snapshot_dir="${1:?Usage: bash scripts/sync_hpc_samples.sh SNAPSHOT_DIRECTORY}"
snapshot_dir="$(realpath "$snapshot_dir")"
snapshot_name="$(basename "$snapshot_dir")"
source_root="${SOURCE_ROOT:-/mnt/d/Hydrostate/samples}"
remote_host="${HPC_HOST:-xiaoz@hpc2021.hku.hk}"
remote_root="${HPC_DATA_ROOT:-/fossfs/xiaozhen/HydroState/samples_v1}"
remote_python="${HPC_PYTHON:-/home/xiaoz/miniconda/envs/hydrostate/bin/python}"
remote_code="${HPC_CODE_ROOT:-/home/xiaoz/HydroState}"

# These values appear in remote shell commands; permit ordinary absolute paths only.
[[ "$snapshot_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]
for value in "$remote_root" "$remote_python" "$remote_code"; do
    [[ "$value" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "Unsupported remote path: $value" >&2; exit 2; }
done
test -s "$snapshot_dir/samples.parquet"
test -s "$snapshot_dir/zarr-files.txt"
test -s "$snapshot_dir/snapshot.json"

ssh_options=(-o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=yes)
if [[ -f /mnt/c/Users/zhenxiao/.ssh/known_hosts ]]; then
    ssh_options+=(-o UserKnownHostsFile=/mnt/c/Users/zhenxiao/.ssh/known_hosts)
fi
if [[ -f /home/xiaoz/.ssh/id_rsa ]]; then
    ssh_options+=(-o IdentitiesOnly=yes -i /home/xiaoz/.ssh/id_rsa)
fi
printf -v rsync_ssh '%q ' ssh "${ssh_options[@]}"
remote_snapshot="$remote_root/manifests/$snapshot_name"
local_hash="$(sha256sum "$snapshot_dir/samples.parquet")"
local_hash="${local_hash%% *}"
remote_hash="$(ssh "${ssh_options[@]}" "$remote_host" \
    "if test -f '$remote_snapshot/samples.parquet'; then sha256sum '$remote_snapshot/samples.parquet'; fi")"
remote_hash="${remote_hash%% *}"
if [[ -n "$remote_hash" && "$remote_hash" != "$local_hash" ]]; then
    echo "Remote snapshot already exists with a different manifest; choose a new snapshot name." >&2
    exit 1
fi

ssh "${ssh_options[@]}" "$remote_host" "mkdir -p '$remote_snapshot'"
rsync -rlt --partial --partial-dir=.rsync-partial --stats --info=progress2 \
    -e "$rsync_ssh" --files-from="$snapshot_dir/zarr-files.txt" \
    "$source_root/" "$remote_host:$remote_root/"
if [[ -n "$remote_hash" ]]; then
    echo "Snapshot already published and unchanged: $remote_snapshot/samples.parquet"
    exit 0
fi

rsync -lt -e "$rsync_ssh" "$snapshot_dir/samples.parquet" \
    "$remote_host:$remote_snapshot/samples.parquet.upload"
rsync -lt -e "$rsync_ssh" "$snapshot_dir/snapshot.json" \
    "$remote_host:$remote_snapshot/snapshot.json"
ssh "${ssh_options[@]}" "$remote_host" \
    "cd '$remote_code' && '$remote_python' tools/validate_zarr.py \
    '$remote_snapshot/samples.parquet.upload' --data-root '$remote_root' && \
    mv '$remote_snapshot/samples.parquet.upload' '$remote_snapshot/samples.parquet'"
echo "Published: $remote_snapshot/samples.parquet"
