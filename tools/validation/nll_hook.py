"""Assertions and a machine-readable report for the short real-data NLL run."""

import json
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from hydrostate.losses import UncertaintyWeightedNLL
from hydrostate.normalization import TASKS


@HOOKS.register_module()
class NLLVerificationHook(Hook):
    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        self.tracked = {}
        for category, prefix in (
            ("encoder", "encoder.backbone.model.blocks.11."),
            ("decoder", "decoder."),
            *((f"{task}_head", f"heads.{task}.") for task in TASKS),
        ):
            name, parameter = next(
                (name, parameter)
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.requires_grad and parameter.ndim > 1
            )
            self.tracked[category] = (name, parameter, parameter.detach().clone())
        for task, parameter in model.loss_module.log_variances.items():
            self.tracked[f"{task}_log_variance"] = (
                f"loss_module.log_variances.{task}",
                parameter,
                parameter.detach().clone(),
            )
        self.statistics = {name: value.clone() for name, value in model.loss_module.named_buffers()}
        self.iterations = []
        self.validation = None

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        row = {key: float(value) for key, value in outputs.items()}
        if not all(torch.isfinite(torch.tensor(value)) for value in row.values()):
            raise RuntimeError(f"Nonfinite training output: {row}")
        self.iterations.append(row)

    def after_val_epoch(self, runner, metrics=None):
        if not metrics or not all(
            torch.isfinite(torch.tensor(metrics[f"{task}_rmse"])) for task in TASKS
        ):
            raise RuntimeError(f"Missing or nonfinite validation RMSE: {metrics}")
        self.validation = metrics

    def after_train(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        changes, max_difference = {}, 0.0
        for category, (name, parameter, before) in self.tracked.items():
            if not torch.isfinite(parameter).all() or torch.equal(parameter.detach(), before):
                raise RuntimeError(f"Missing or nonfinite parameter update: {name}")
            changes[category] = dict(
                parameter=name,
                max_abs_change=float((parameter.detach() - before).abs().max()),
                optimizer_steps=int(runner.optim_wrapper.optimizer.state[parameter]["step"]),
            )
            if runner.world_size > 1:
                reference = parameter.detach().clone()
                torch.distributed.broadcast(reference, src=0)
                difference = (parameter.detach() - reference).abs().max()
                torch.distributed.all_reduce(difference, op=torch.distributed.ReduceOp.MAX)
                max_difference = max(max_difference, float(difference))
        if max_difference != 0:
            raise RuntimeError(f"DDP parameter difference: {max_difference}")
        for name, value in model.loss_module.named_buffers():
            torch.testing.assert_close(value, self.statistics[name], atol=0, rtol=0)
        if self.validation is None:
            raise RuntimeError("Validation loop did not complete")
        checkpoint = Path(runner.work_dir) / f"iter_{runner.iter}.pth"
        if runner.rank == 0:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            restored = UncertaintyWeightedNLL(model.loss_module.target_statistics)
            restored.load_state_dict(
                {
                    key.removeprefix("loss_module."): value
                    for key, value in state.items()
                    if key.startswith("loss_module.")
                }
            )
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(
                    value, model.loss_module.state_dict()[key].cpu(), atol=0, rtol=0
                )
            del state
        report = dict(
            rank=runner.rank,
            world_size=runner.world_size,
            gpu=torch.cuda.get_device_name(),
            batch_size_per_gpu=runner.train_dataloader.batch_size,
            gradient_accumulation=runner.optim_wrapper._accumulative_counts,
            iterations=len(self.iterations),
            max_rank_parameter_difference=max_difference,
            parameter_updates=changes,
            target_statistics=model.loss_module.target_statistics,
            final_variances={
                task: float(model.loss_module.variance(task).detach()) for task in TASKS
            },
            active_batches={
                task: sum(row[f"valid_count_{task}"] > 0 for row in self.iterations)
                for task in TASKS
            },
            validation=self.validation,
            checkpoint=str(checkpoint),
            normalization_frozen=True,
            checkpoint_loss_restored=(runner.rank == 0),
            logs=self.iterations,
        )
        path = Path(runner.work_dir) / f"nll_verification_rank{runner.rank}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        runner.logger.info("NLL verification passed: %s", path)
