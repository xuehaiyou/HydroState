"""Mixed manual/product batches preserve masks, shapes and source-independent weights."""

import json

import numpy as np
import pandas as pd
import torch

from hydrostate.data.retrieval.contract import RetrievalSchema
from hydrostate.data.site.contract import SiteSchema
from hydrostate.datasets.collate import hydro_collate
from hydrostate.datasets.joint_dataset import JointRetrievalDataset
from hydrostate.datasets.manual_joint_dataset import ManualJointDataset
from hydrostate.datasets.retrieval_dataset import as_tensors


def test_mixed_manual_and_product_collation(monkeypatch):
    def inputs():
        return dict(s1=torch.zeros((5, 2, 120, 120)), target_time={"water": torch.tensor([0, 1])})

    def base_item(self, index):
        targets = as_tensors(
            {
                k: np.zeros(s.tail_shape, s.dtype)
                for schema in (RetrievalSchema(), SiteSchema())
                for k, s in schema.targets.items()
            }
        )
        targets["metainfo"] = dict(sample_id="product", supervision_source="product")
        return dict(inputs=inputs(), data_samples=targets)

    class Manual:
        table = pd.DataFrame([dict(spatial_group="flood:test", target_time="2020-01-01")])

        def __getitem__(self, index):
            labels = {"metainfo": dict(sample_id="manual", source="sen1floods11")}
            for c in ("water", "snow_ice", "land"):
                labels[c] = torch.ones((1, 40, 40)) if c == "water" else torch.zeros((1, 40, 40))
                labels[c + "_valid"] = torch.full((1, 40, 40), c == "water", dtype=torch.bool)
            return dict(inputs=inputs(), data_samples=labels)

    monkeypatch.setattr(JointRetrievalDataset, "__getitem__", base_item)
    data = ManualJointDataset.__new__(ManualJointDataset)
    data.base_length, data.manual, data.training, data.split = 1, Manual(), False, "train"
    raw = hydro_collate([data[0], data[1]])
    targets = raw["data_samples"]
    assert targets["manual_water_valid"][1].all()
    assert not targets["manual_land_valid"].any()
    assert not targets["water_valid"].any()  # no double product loss on manual labels
    assert not targets["site_sm_native_valid"].any()
    assert not targets["site_et_native_valid"].any()
    assert targets["metainfo"]["supervision_source"] == ["product", "manual"]


def test_slurm_submit_receipt_prevents_duplicates(tmp_path, monkeypatch):
    from hydrostate.experiments.manual_cover_launch import submit

    calls = []
    monkeypatch.setattr("subprocess.check_output", lambda cmd, **kw: calls.append(cmd) or "123\n")
    assert submit(tmp_path, ["sbatch", "run.sh"], {"phase": "test"}) == "123"
    assert submit(tmp_path, ["sbatch", "run.sh"], {"phase": "test"}) == "123"
    assert len(calls) == 1
    assert json.loads((tmp_path / "submission.json").read_text())["job_id"] == "123"


def test_residual_pair_reuses_joint_reference_without_retraining_baselines(tmp_path):
    from mmengine.config import Config

    from hydrostate.experiments.manual_cover import RESIDUAL_VARIANTS, prepare_residual_pair
    from hydrostate.storage.packs import file_hash

    root, baseline, output = (tmp_path / name for name in ("manual", "baseline", "pair"))
    root.mkdir()
    (baseline / "CGC").mkdir(parents=True)
    marker = root / "_PUBLISHED.json"
    marker.write_text('{"complete": true}')
    (baseline / "prepared.json").write_text(json.dumps({"publication_sha256": file_hash(marker)}))
    loader = dict(
        dataset=dict(
            type="ManualJointDataset",
            manual_root=str(root),
            manual_selection=str(baseline / "manual_selection.parquet"),
        )
    )
    cfg = Config(
        dict(
            model=dict(
                type="CoSAHydroStateModel", router=dict(type="CGCRouter"), manual_cover_weight=2
            ),
            train_dataloader=loader,
            val_dataloader=loader,
            test_dataloader=loader,
            load_from=None,
            resume=False,
            default_hooks=dict(checkpoint=dict(save_best="manual_joint_score")),
        )
    )
    path = baseline / "CGC/config.py"
    cfg.dump(str(path))
    digest = file_hash(path)
    prepare_residual_pair(baseline, root, output)
    prepare_residual_pair(baseline, root, output)
    assert file_hash(path) == digest
    assert set(d.name for d in output.iterdir() if d.is_dir()) == set(RESIDUAL_VARIANTS)
    for name in RESIDUAL_VARIANTS:
        candidate = Config.fromfile(output / name / "config.py")
        assert candidate.model.type == "BackgroundResidualModel"
        assert candidate.model.router == dict(center=name == "bg_residual_centered")
        assert candidate.model.manual_cover_weight == 2
        for key in ("train_dataloader", "val_dataloader", "test_dataloader", "default_hooks"):
            assert candidate[key] == cfg[key]


def test_final_test_does_not_resume_training_state(tmp_path, monkeypatch):
    from mmengine.config import Config

    from hydrostate.experiments.manual_cover_launch import finish_test

    work = tmp_path / "work"
    work.mkdir()
    (work / "best_manual_joint_score_epoch_6.pth").touch()
    config = tmp_path / "config.py"
    Config(dict(work_dir=str(work), resume=True)).dump(str(config))
    commands = []
    monkeypatch.setattr("subprocess.run", lambda command, **kw: commands.append(command))
    finish_test(str(config))
    assert "resume=False" in commands[0]
