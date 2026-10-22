"""Exercise the actual shell launchers with simulated CUDA and training commands."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAKE_COMMAND = r'''
import contextlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if Path(sys.argv[0]).name == "python" and sys.argv[1] == "-":
    count = int(os.environ["TEST_VISIBLE_GPUS"])
    cuda = SimpleNamespace(
        device_count=lambda: count,
        is_available=lambda: count > 0,
        device=lambda index: contextlib.nullcontext(),
        get_device_name=lambda index: f"Test GPU {index}",
        get_device_properties=lambda index: SimpleNamespace(total_memory=48 * 1024**3),
        is_bf16_supported=lambda **kwargs: os.environ["TEST_BF16"] == "1",
    )
    sys.modules["torch"] = SimpleNamespace(
        cuda=cuda, __version__="test", version=SimpleNamespace(cuda="test")
    )
    sys.argv = sys.argv[1:]
    exec(compile(sys.stdin.read(), "<GPU preflight>", "exec"))
else:
    Path(os.environ["TEST_LAUNCH_RECORD"]).write_text(json.dumps(dict(
        command=Path(sys.argv[0]).name, args=sys.argv[1:]
    )))
'''


@pytest.fixture
def launch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "torchrun"):
        path = bin_dir / name
        path.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
        path.chmod(0o755)
    conda_base = tmp_path / "conda"
    profile = conda_base / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        f'conda() {{ :; }}\nexport PATH={shlex.quote(str(bin_dir))}:"$PATH"\n'
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    for name in ("config.json", "weights.pth"):
        (model_path / name).write_text("preflight fixture")
    manifest, statistics = tmp_path / "samples.parquet", tmp_path / "target_stats.json"
    manifest.write_text("preflight fixture")
    statistics.write_text("preflight fixture")
    record = tmp_path / "launch.json"
    # Do not inherit a login shell's experiment settings or explicit GPU count.
    env = {key: value for key, value in os.environ.items() if not key.startswith("HYDROSTATE_")}
    env.update(
        HYDROSTATE_CONDA_BASE=str(conda_base),
        HYDROSTATE_CODE_ROOT=str(PROJECT_ROOT),
        HYDROSTATE_DATA_ROOT=str(tmp_path),
        HYDROSTATE_MODEL_PATH=str(model_path),
        HYDROSTATE_MANIFEST=str(manifest),
        HYDROSTATE_TARGET_STATS=str(statistics),
        HYDROSTATE_WORK_DIR=str(tmp_path / "work"),
        TEST_LAUNCH_RECORD=str(record),
    )

    def run(count, script="12_train_l40s.sh", requested=None, bf16=True, precision="bfloat16"):
        environment = dict(env, TEST_VISIBLE_GPUS=str(count), TEST_BF16=str(int(bf16)))
        environment["HYDROSTATE_PRECISION"] = precision
        if requested is not None:
            environment["HYDROSTATE_GPUS_PER_NODE"] = requested
        result = subprocess.run(
            [
                "bash", str(PROJECT_ROOT / script if script.startswith("scripts/")
                            else PROJECT_ROOT / "scripts/legacy/slurm" / script),
                "resume=True", "work_dir=/tmp/experiment with spaces",
            ],
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
        )
        invocation = json.loads(record.read_text()) if record.exists() else None
        return result, invocation

    return run


@pytest.mark.parametrize("script", ["11_train_hpc.sh", "12_train_l40s.sh"])
@pytest.mark.parametrize("count", [1, 2, 4])
def test_launcher_uses_visible_gpu_count_and_preserves_arguments(launch, script, count):
    result, invocation = launch(count, script=script)
    assert result.returncode == 0, result.stderr
    args = invocation["args"]
    if count == 1:
        assert invocation["command"] == "python"
        assert args[args.index("--launcher") + 1] == "none"
        assert "single-GPU" in result.stdout
    else:
        assert invocation["command"] == "torchrun"
        assert f"--nproc_per_node={count}" in args
        assert args[args.index("--launcher") + 1] == "pytorch"
        assert "DDP" in result.stdout
    assert args[-3:] == [
        "optim_wrapper.dtype=bfloat16", "resume=True", "work_dir=/tmp/experiment with spaces"
    ]
    assert args[args.index("tools/train.py") + 1] == (
        "configs/experiments/legacy/hydrostate_l40s.py" if script.startswith("12_") else "configs/experiments/legacy/hydrostate_hpc.py"
    )


@pytest.mark.parametrize("requested", ["auto", "1"])
def test_explicit_auto_or_matching_count(launch, requested):
    result, invocation = launch(1, requested=requested)
    assert result.returncode == 0, result.stderr
    assert invocation["command"] == "python"


@pytest.mark.parametrize(
    "settings,message",
    [
        (dict(count=0), "No CUDA GPU is visible"),
        (dict(count=1, requested="2"), "Visible 1 GPUs, explicitly requested 2"),
        (dict(count=1, requested="0"), "positive integer"),
        (dict(count=1, requested="invalid"), "positive integer"),
        (dict(count=1, bf16=False), "GPU lacks native BF16"),
    ],
)
def test_preflight_failures_do_not_launch_training(launch, settings, message):
    result, invocation = launch(**settings)
    assert result.returncode != 0
    assert message in result.stderr
    assert invocation is None


def test_explicit_float16_allows_gpu_without_native_bf16(launch):
    result, invocation = launch(1, bf16=False, precision="float16")
    assert result.returncode == 0, result.stderr
    assert "optim_wrapper.dtype=float16" in invocation["args"]


def test_current_launcher_selects_current_model_and_one_gpu(launch):
    result, invocation = launch(1, script="scripts/train.sh")
    assert result.returncode == 0, result.stderr
    assert invocation["command"] == "python"
    assert invocation["args"][1] == "configs/hydrostate_mtlora.py"


def test_current_launcher_refuses_multiple_visible_gpus(launch):
    result, invocation = launch(2, script="scripts/train.sh")
    assert result.returncode != 0 and invocation is None
    assert "Visible 2 GPUs, explicitly requested 1" in result.stderr
