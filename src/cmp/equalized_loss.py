import math
from typing import Union

import cooper
import torch
from torch import nn

import src.cmp.utils as cmp_utils
import src.models as models
from src.cmp.cyclic_buffer import GroupCyclicBuffer
from src.utils import EvalMeters, TrainMeters


class EqualizedLossProblem(cooper.ConstrainedMinimizationProblem):
    """
    Minimization problem for a classification task, with per-group constraints aiming to
    equalize the loss of each protected group with the overall loss.

    Formally, the constraints are:
        loss(group=G) - avg_loss == 0  (for each protected group G)

    Given that a specific mini-batch may not contain examples from all protected groups,
    we employ "Indexed" constraints and multipliers. This means that we only estimate
    constraints for the observed groups. See Cooper's documentation for more details.

    Args:
        num_protected_groups: number of protected groups.
        last_pruning_epoch: last epoch in which pruning is performed.
        apply_mitigation_while_pruning: whether to apply the mitigation method while pruning.
        label_smoothing: label smoothing parameter for the loss function.
        weight_decay: weight decay parameter for the L2 regularization term.
        device: device for storing the Lagrange multipliers.
        intersectional: whether to use intersectional constraints.
    """

    def __init__(
        self,
        num_protected_groups: list[int],
        last_pruning_epoch: int,
        apply_mitigation_while_pruning: bool = False,
        detach_model_constraint_contribution: bool = False,
        abs_equality_constraint: bool = False,
        label_smoothing: float = 0.0,
        weight_decay: float = 0.0,
        device: torch.device = torch.device("cpu"),
        intersectional: bool = True,
        buffer_memory: int = None,
    ):
        self.num_protected_groups = num_protected_groups
        self.intersectional = intersectional

        # This CMP implements one constraint per protected group
        if intersectional:
            self.num_constraints = math.prod(num_protected_groups)
        else:
            self.num_constraints = sum(num_protected_groups)

        # `is_indexed` is set to True to allow for "indexed" multipliers.
        multiplier_kwargs = {"shape": self.num_constraints, "device": device, "is_indexed": True}
        constraint_kwargs = {
            "constraint_type": cooper.ConstraintType.EQUALITY,
            "formulation_type": cooper.FormulationType.LAGRANGIAN,
        }
        self.constraint_group = cooper.ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs)

        self.last_pruning_epoch = last_pruning_epoch
        self.apply_mitigation_while_pruning = apply_mitigation_while_pruning
        self.detach_model_constraint_contribution = detach_model_constraint_contribution
        self.abs_equality_constraint = abs_equality_constraint

        self.weight_decay = weight_decay

        self.loss_fn, self.accuracy_fn = cmp_utils.build_metric_callables(label_smoothing)

        self.meters = TrainMeters("EqualizedLossProblem")

        self.constraint_buffer = None
        if buffer_memory is not None and buffer_memory > 0:
            self.constraint_buffer = GroupCyclicBuffer(self.num_protected_groups, buffer_memory, intersectional, device)
        super().__init__()

    def precompute_baseline_metrics(self, baseline_model: torch.nn.Module, dataloaders: dict, device):
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
        assert self.baseline_train_avg_acc is not None, "baseline_train_avg_acc must be precomputed"

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

        if not is_training:
            # We only compute defects for the validation loop at the end of an epoch,
            # not at the batch level.
            return cooper.CMPState(loss=loss, misc=misc)

        if not self.apply_mitigation_while_pruning and epoch < self.last_pruning_epoch:
            return cooper.CMPState(loss=loss, misc=misc)

        # The violation is calculated using the train batch, just as the loss.
        group_indices = _group_counts.nonzero(as_tuple=False).flatten()

        violation, _, _ = self.compute_defects(
            avg_acc=_avg_acc,
            group_acc=_group_acc,
            baseline_avg_acc=self.baseline_train_avg_acc,
            baseline_group_acc=self.baseline_train_group_acc,
            group_indices=group_indices,
            avg_loss=_avg_loss,
            group_loss=_group_loss,
        )

        if self.constraint_buffer is None:
            strict_violation = violation
            strict_indices = group_indices
        else:
            self.constraint_buffer.update(
                protected_attr=protected_attr, per_sample_metric=batch_metrics.sample_loss.detach()
            )
            strict_counts, group_loss_for_strict, avg_loss_for_strict = self.constraint_buffer.query()
            strict_indices = strict_counts.nonzero(as_tuple=False).flatten()

            strict_violation, _, _ = self.compute_defects(
                avg_acc=_avg_acc,
                group_acc=_group_acc,
                baseline_avg_acc=self.baseline_train_avg_acc,
                baseline_group_acc=self.baseline_train_group_acc,
                group_indices=strict_indices,
                avg_loss=avg_loss_for_strict,
                group_loss=group_loss_for_strict,
            )

        constraint_state = cooper.ConstraintState(
            violation=violation,
            strict_violation=strict_violation,
            constraint_features=group_indices,
            strict_constraint_features=strict_indices,
        )
        return cooper.CMPState(loss=loss, observed_constraints=[(self.constraint_group, constraint_state)], misc=misc)

    def compute_defects(
        self,
        avg_acc: torch.Tensor,
        group_acc: torch.Tensor,
        baseline_avg_acc: torch.Tensor,
        baseline_group_acc: torch.Tensor,
        group_indices: torch.Tensor,
        avg_loss: torch.Tensor,
        group_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Accuracy gaps with respect to the baseline model are computed for logging
        # purposes only.
        group_acc_gaps = baseline_group_acc[group_indices] - group_acc[group_indices]
        avg_acc_gap = baseline_avg_acc - avg_acc

        if self.detach_model_constraint_contribution:
            constraint_violations = group_loss[group_indices] - avg_loss.detach()
        else:
            constraint_violations = group_loss[group_indices] - avg_loss

        if self.abs_equality_constraint:
            constraint_violations.abs_()

        return constraint_violations, group_acc_gaps, avg_acc_gap
