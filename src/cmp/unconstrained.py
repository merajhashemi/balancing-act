from typing import Union

import cooper
import torch
from torch import nn

import src.cmp.utils as cmp_utils
import src.models as models
from src.utils import EvalMeters, TrainMeters


class BaselineProblem(cooper.ConstrainedMinimizationProblem):
    def __init__(
        self,
        num_protected_groups: list[int],
        label_smoothing: float = 0.0,
        weight_decay: float = 0.0,
        device: torch.device = torch.device("cpu"),
        intersectional: bool = True,
    ):
        self.num_protected_groups = num_protected_groups
        self.intersectional = intersectional
        self.num_constraints = 0
        self.weight_decay = weight_decay

        self.loss_fn, self.accuracy_fn = cmp_utils.build_metric_callables(label_smoothing)

        self.meters = TrainMeters("BaselineProblem")
        super().__init__()

    def precompute_baseline_metrics(self, baseline_model: torch.nn.Module, dataloaders, device):
        """
        Populates the accuracy of the reference baseline model on the training and
        validation sets.
        """

        for split, dataloader in dataloaders.items():
            stats = cmp_utils.compute_model_acc_stats(self, baseline_model, dataloader, device)
            setattr(self, f"baseline_{split}_avg_acc", stats["avg_acc"])
            setattr(self, f"baseline_{split}_group_acc", stats["group_acc"])

    def compute_cmp_state(
        self,
        model: nn.Module,
        train_batch: cmp_utils.Batch,
        meters: Union[TrainMeters, EvalMeters],
        is_training: bool,
        epoch: int,
    ) -> cooper.CMPState:
        assert train_batch.protected_attr is not None, "protected_attr must be provided"

        input, target, protected_attr = train_batch.input, train_batch.target, train_batch.protected_attr
        batch_size = train_batch.input.size(0)

        logits = model(input)
        batch_metrics = cmp_utils.compute_batch_metrics(
            self, logits, target, protected_attr, self.num_protected_groups, self.intersectional
        )
        _group_counts = batch_metrics.group_counts
        misc = {"metrics": batch_metrics}

        _avg_acc, _group_acc = batch_metrics.avg_acc.detach(), batch_metrics.group_acc.detach()
        _avg_loss, _group_loss = batch_metrics.avg_loss, batch_metrics.group_loss

        meters.avg_loss.update(_avg_loss.detach(), batch_size)
        meters.group_loss.update(_group_loss.detach(), _group_counts)
        meters.avg_acc.update(_avg_acc, batch_size)
        meters.group_acc.update(_group_acc, _group_counts)

        loss = batch_metrics.avg_loss
        if self.weight_decay != 0:
            misc["sq_l2_norm"] = models.model_sq_l2_norm(model)
            loss += self.weight_decay * misc["sq_l2_norm"] / 2

        return cooper.CMPState(loss=loss, misc=misc)

    def compute_defects(
        self,
        avg_acc: torch.Tensor,
        group_acc: torch.Tensor,
        baseline_avg_acc: torch.Tensor,
        baseline_group_acc: torch.Tensor,
        group_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the accuracy gaps and the difference between the group accuracy gaps
        and the average accuracy gap"""

        # Do not consider log accuracy gap for unobserved groups
        group_acc_gaps = baseline_group_acc[group_indices] - group_acc[group_indices]
        avg_acc_gap = baseline_avg_acc - avg_acc

        return group_acc_gaps - avg_acc_gap, group_acc_gaps, avg_acc_gap
