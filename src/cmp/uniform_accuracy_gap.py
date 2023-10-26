import math
from typing import Union

import cooper
import torch
from torch import nn

import src.cmp.utils as cmp_utils
import src.models as models
from src.cmp.cyclic_buffer import GroupCyclicBuffer
from src.utils import EvalMeters, TrainMeters


class UniformAccuracyGapProblem(cooper.ConstrainedMinimizationProblem):
    """

    Minimization problem for a classification task, with per-group constraints aiming to
    bound the gap in accuracy for each group to be at most the overall accuracy gap
    (across the entire dataset) plus a tolerance. Gaps are measured with respect to the
    performance of a pretrained baseline model.

    Define the gap in accuracy for a given protected group be:
        accuracy_gap(group=G) = acc(baseline|group=G) - acc(compressed|group=G)

    And let the overall accuracy gap be:
        avg_accuracy_gap = acc(baseline) - acc(compressed)

    Constraints are of the form:
        accuracy_gap(group=G) <= avg_accuracy_gap + tol  (for each protected group G)

    We consider the cross entropy loss with respect to the correct label as a surrogate
    metric for computing gradients of the constraints.

    Whereas `acc(compressed)` and `acc(compressed|group=G)` are estimated on each
    observed mini-batch, we compute `baseline_avg_acc` and `acc(baseline|group=G)` over
    the *entire* dataset. The computation of accuracy metrics for the baseline model is
    done once at the beginning of training by calling `precompute_baseline_metrics`.
    This avoids recomputing the baseline metrics at every batch and also yields a more
    accurate estimate of the baseline metrics.

    Given that a specific mini-batch may not contain examples from all protected groups,
    we employ "Indexed" constraints and multipliers. This means that we only estimate
    constraints for the observed groups. See Cooper's documentation for more details.

    Args:
        num_protected_groups: number of protected groups.
        tolerance: non-negative tolerance for the accuracy gap constraints.
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
        tolerance: float,
        last_pruning_epoch: int,
        apply_mitigation_while_pruning: bool = False,
        detach_model_constraint_contribution: bool = False,
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
            "constraint_type": cooper.ConstraintType.INEQUALITY,
            "formulation_type": cooper.FormulationType.LAGRANGIAN,
        }
        self.constraint_group = cooper.ConstraintGroup(**constraint_kwargs, multiplier_kwargs=multiplier_kwargs)

        self.last_pruning_epoch = last_pruning_epoch
        self.apply_mitigation_while_pruning = apply_mitigation_while_pruning
        self.detach_model_constraint_contribution = detach_model_constraint_contribution

        assert tolerance >= 0, "tolerance must be non-negative"
        self.tolerance = tolerance

        self.weight_decay = weight_decay

        self.loss_fn, self.accuracy_fn = cmp_utils.build_metric_callables(label_smoothing)

        self.meters = TrainMeters("UniformAccuracyGapProblem")

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

            if split == "train":
                # Storing the loss for the computation of surrogates
                setattr(self, f"baseline_{split}_avg_loss", stats["avg_loss"])
                setattr(self, f"baseline_{split}_group_loss", stats["group_loss"])

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

        # The proxy violation is calculated using the train batch, just as the loss.
        group_indices = _group_counts.nonzero(as_tuple=False).flatten()

        # For the cross entropy loss, the sign is flipped as we want to minimize
        # the compressed model's loss as opposed to maximizing its accuracy.
        surrogate = self.compute_surrogate(
            avg_metric=-_avg_loss,
            group_metric=-_group_loss,
            baseline_average_metric=-self.baseline_train_avg_loss,
            baseline_group_metric=-self.baseline_train_group_loss,
            group_indices=group_indices,
        )

        if self.constraint_buffer is None:
            # Compute defects based on the batch accuracies
            strict_indices = group_indices
            group_acc_for_violations = _group_acc
            avg_acc_for_violations = _avg_acc
        else:
            # Update and query the cyclic buffer for computing the defects.
            # Note that if the batch contains more samples than the buffer can store,
            # the excess samples are not used for computing the accuracies.
            self.constraint_buffer.update(
                protected_attr=protected_attr, per_sample_metric=batch_metrics.sample_acc.bool()
            )
            strict_counts, group_acc_for_violations, avg_acc_for_violations = self.constraint_buffer.query()
            strict_indices = strict_counts.nonzero(as_tuple=False).flatten()

        strict_violation, _, _ = self.compute_defects(
            avg_acc=avg_acc_for_violations,
            group_acc=group_acc_for_violations,
            baseline_avg_acc=self.baseline_train_avg_acc,
            baseline_group_acc=self.baseline_train_group_acc,
            group_indices=strict_indices,
        )

        constraint_state = cooper.ConstraintState(
            violation=surrogate,
            strict_violation=strict_violation,
            constraint_features=group_indices,
            strict_constraint_features=strict_indices,
        )

        return cooper.CMPState(loss=loss, observed_constraints=[(self.constraint_group, constraint_state)], misc=misc)

    def compute_surrogate(
        self,
        avg_metric: torch.Tensor,
        group_metric: torch.Tensor,
        baseline_average_metric: torch.Tensor,
        baseline_group_metric: torch.Tensor,
        group_indices: torch.Tensor,
    ):
        """
        Compute the surrogate "constraint": group_surrogate_gap <= avg_surrogate_gap + tol
        """

        group_surrogate_gap = baseline_group_metric - group_metric
        avg_surrogate_gap = baseline_average_metric - avg_metric

        # We do not add the tolerance to the surrogate term as it does not
        # contribute to the gradients.
        if self.detach_model_constraint_contribution:
            surrogate = group_surrogate_gap[group_indices] - avg_surrogate_gap.detach()
        else:
            surrogate = group_surrogate_gap[group_indices] - avg_surrogate_gap

        return surrogate

    def compute_defects(
        self,
        avg_acc: torch.Tensor,
        group_acc: torch.Tensor,
        baseline_avg_acc: torch.Tensor,
        baseline_group_acc: torch.Tensor,
        group_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the defects for the accuracy gap constraint"""

        group_acc_gaps = baseline_group_acc[group_indices] - group_acc[group_indices]
        avg_acc_gap = baseline_avg_acc - avg_acc

        # Constraints: group_acc_gap <= avg_acc_gap + tol
        constraint_violations_without_tol = group_acc_gaps - avg_acc_gap
        constraint_violations = constraint_violations_without_tol - self.tolerance

        return constraint_violations, group_acc_gaps, avg_acc_gap
