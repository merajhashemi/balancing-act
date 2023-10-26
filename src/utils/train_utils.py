import logging
import math
from typing import Sequence, Tuple, Union

import cooper
import torch
import torch.nn as nn

import src.models as models
import src.models.utils as model_utils
from src.cmp.utils import Batch
from src.utils.experiment_meters import EvalMeters

logger = logging.getLogger(__name__)


def ensure_iterable(x):
    if isinstance(x, (list, tuple)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.flatten().tolist()
    else:
        return [x]


def flatten_tensor_for_logging(tensor: torch.Tensor, indices=None, prefix: str = ""):
    iterable_tensor = ensure_iterable(tensor)

    # Ensure logs are 00, 01, 02, ... instead of 0, 1, 2, ... if two digit indices are
    # needed.
    num_entries = len(iterable_tensor) if indices is None else len(indices)
    num_digits = len(str(num_entries - 1))

    if indices is None:
        # Enumerate the tensor if no indices are provided
        return {str(prefix) + "_" + str(k).zfill(num_digits): v for k, v in enumerate(iterable_tensor)}
    else:
        # The entries in the tensor correspond to the provided indices
        assert len(indices) == len(tensor.flatten())
        return {str(prefix) + "_" + str(k).zfill(num_digits): v for k, v in zip(indices, iterable_tensor)}


def extract_model_stats_logs(model: nn.Module, prefix: str = "", skip_keys: tuple[str] = ("layer_stats",)) -> dict:
    # Compute model stats and extract dict from dataclass
    model_stats_dict = models.get_model_stats(model).__dict__

    # Skip "layer_stats" and other requested keys when logging model_stats
    return {prefix + k: v for k, v in model_stats_dict.items() if k not in skip_keys}


def append_ordered_statistics(dictionary, values, quantiles, prefix="", suffix=""):

    new_dict = {}
    for q in quantiles:
        if q == 1.0:
            name = f"max_"
        else:
            name = f"quantile_{q}_"

        new_dict[name] = torch.quantile(values, q).item()

    new_dict = {prefix + k + suffix: v for k, v in new_dict.items()}
    dictionary.update(new_dict)

    return dictionary


def extract_constraint_logs(
    observed_constraints: Sequence[Tuple[cooper.ConstraintGroup, cooper.ConstraintState]], prefix: str = ""
) -> dict:
    """
    Extract the constraint violations and multipliers from a CMPState object and
    write them to a dictionary for logging.

    Current implementation assumes that there is only one constraint group observed in
    the CMPState.
    """

    constraint_log = {}

    if len(observed_constraints) == 0:
        return constraint_log

    if len(observed_constraints) > 1:
        raise NotImplementedError(
            "Extracting the constraint logs for problems with multiple constraint groups is not implemented."
        )

    # NOTE: assuming there is only one constraint group
    constraint_group, constraint_state = observed_constraints[0]
    group_indices = constraint_state.constraint_features
    strict_group_indices = constraint_state.strict_constraint_features

    # Logging *all* multipliers, not just the ones for the observed constraints
    multipliers = constraint_group.multiplier.weight.clone().detach()
    multipliers = flatten_tensor_for_logging(multipliers, prefix=prefix + "multiplier")
    constraint_log.update(multipliers)

    # Constraints violations are only logged for the observed constraints
    if constraint_state.strict_violation is not None:
        if strict_group_indices is None:
            strict_group_indices = group_indices
        violations = flatten_tensor_for_logging(
            constraint_state.strict_violation, indices=strict_group_indices.tolist(), prefix=prefix + "violation"
        )
    else:
        violations = flatten_tensor_for_logging(
            constraint_state.violation, indices=group_indices.tolist(), prefix=prefix + "violation"
        )

    constraint_log.update(violations)

    return constraint_log


def process_batch(batch: tuple[torch.Tensor], device: Union[str, torch.device]):
    input, target = batch[0].to(device), batch[1].to(device)
    protected_attr = batch[2].to(device) if len(batch) == 3 else None

    return Batch(input=input, target=target, protected_attr=protected_attr)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    cmp: cooper.ConstrainedMinimizationProblem,
    constrained_optimizer: cooper.optim.ConstrainedOptimizer,
    device: Union[str, torch.device],
    _step: int,
    _epoch: int,
):
    model.train()
    # If the model has no masks, this is a no-op.
    model_utils.set_execution_mode_(model, masked=True)

    batch_logs = []

    for batch_id, train_batch in enumerate(train_loader, start=1):
        train_batch = process_batch(train_batch, device=device)

        cmp_fn_kwargs = dict(model=model, train_batch=train_batch)
        cmp_fn = lambda: cmp.compute_cmp_state(**cmp_fn_kwargs, is_training=True, epoch=_epoch, meters=cmp.meters)
        cmp_state, lagrangian_store = constrained_optimizer.roll(compute_cmp_state_fn=cmp_fn, return_multipliers=True)

        if batch_id % math.ceil(len(train_loader) / 50) == 0:
            logger.info(f"Train - Epoch {_epoch} Batch {batch_id} Acc {cmp_state.misc['metrics'].avg_acc.detach()}")

        train_loss, train_acc = cmp_state.loss, cmp_state.misc["metrics"].avg_acc
        current_batch_log = {
            "_step": _step + batch_id,
            "train/batch/loss": train_loss.detach(),
            "train/batch/acc": train_acc.detach(),
        }

        if cmp.num_constraints > 0:
            current_batch_log.update({"train/batch/lagrangian": lagrangian_store.lagrangian.detach()})

            prefix = "train/batch/"
            if "Loss" in cmp.__class__.__name__:
                # NOTE: logging the loss-based violations as `loss_violation`
                prefix += "loss_"

            constraint_log = extract_constraint_logs(observed_constraints=cmp_state.observed_constraints, prefix=prefix)
            current_batch_log.update(constraint_log)

        batch_logs.append(current_batch_log)

    # No need to detach since the avg is computed on-the-fly.
    epoch_log = {
        "_epoch": _epoch,
        "train/epoch/loss": cmp.meters.avg_loss.avg,
        "train/epoch/acc": cmp.meters.avg_acc.avg,
    }
    epoch_log.update(flatten_tensor_for_logging(cmp.meters.group_loss.avg, prefix="train/epoch/group_loss"))
    epoch_log.update(flatten_tensor_for_logging(cmp.meters.group_acc.avg, prefix="train/epoch/group_acc"))

    return batch_logs, epoch_log, (_step + batch_id)


@torch.inference_mode()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cmp: cooper.ConstrainedMinimizationProblem,
    device: Union[str, torch.device],
    _epoch: int,
    split_prefix: str,
):
    eval_meters = do_eval_forward_pass(model, dataloader, cmp, device, _epoch)

    epoch_log = {
        "_epoch": _epoch,
        f"{split_prefix}/epoch/loss": eval_meters.avg_loss.avg,
        f"{split_prefix}/epoch/acc": eval_meters.avg_acc.avg,
    }
    epoch_log.update(flatten_tensor_for_logging(eval_meters.group_loss.avg, prefix=f"{split_prefix}/epoch/group_loss"))
    epoch_log.update(flatten_tensor_for_logging(eval_meters.group_acc.avg, prefix=f"{split_prefix}/epoch/group_acc"))

    split = split_prefix.split("_")[0]
    if hasattr(cmp, f"baseline_{split}_avg_acc"):
        # We log the accuracy gaps and constraint violations aggregated over the entire
        # evaluation set. Note that we only log accuracy-based violations and not
        # surrogates.
        #
        # These metrics are also logged for CMPs without constraints, even though
        # accuracy gaps are not used for the optimization.

        group_indices = eval_meters.group_acc.synced_count.nonzero(as_tuple=False).flatten()

        # split is only useful for "train_eval" case
        baseline_avg_acc = getattr(cmp, f"baseline_{split}_avg_acc")
        baseline_group_acc = getattr(cmp, f"baseline_{split}_group_acc")

        defect_kwargs = dict(
            avg_acc=eval_meters.avg_acc.avg,
            group_acc=eval_meters.group_acc.avg,
            baseline_avg_acc=baseline_avg_acc,
            baseline_group_acc=baseline_group_acc,
        )

        if "Loss" in cmp.__class__.__name__:
            defect_kwargs["avg_loss"] = eval_meters.avg_loss.avg
            defect_kwargs["group_loss"] = eval_meters.group_loss.avg

        constraint_violations, group_acc_gaps, avg_acc_gap = cmp.compute_defects(
            **defect_kwargs, group_indices=group_indices
        )

        acc_gap_violations_without_tol = group_acc_gaps - avg_acc_gap
        if "Loss" in cmp.__class__.__name__:
            # NOTE: logging the loss-based violations as `loss_violation`, and logging
            # group_acc - avg_acc as `violation` to facilitate comparisons to other CMPs
            epoch_log.update(
                flatten_tensor_for_logging(constraint_violations, prefix=f"{split_prefix}/epoch/loss_violation")
            )
            epoch_log.update(
                flatten_tensor_for_logging(acc_gap_violations_without_tol, prefix=f"{split_prefix}/epoch/violation")
            )
        else:
            epoch_log.update(
                flatten_tensor_for_logging(constraint_violations, prefix=f"{split_prefix}/epoch/violation")
            )

        epoch_log.update(flatten_tensor_for_logging(group_acc_gaps, prefix=f"{split_prefix}/epoch/group_acc_gap"))
        epoch_log.update({f"{split_prefix}/epoch/accuracy_gap": avg_acc_gap})
        epoch_log.update({f"{split_prefix}/epoch/accuracy_gap_std": torch.std(group_acc_gaps, unbiased=True)})

        if len(acc_gap_violations_without_tol) > 10:
            # 0.5 is the median, 0.95 is the 95th percentile, 1.0 is the max
            quantiles = [0.5, 0.75, 0.90, 0.92, 0.95, 0.97, 0.99, 1.0]
        else:
            quantiles = [0.5, 1.0]

        epoch_log = append_ordered_statistics(
            epoch_log,
            acc_gap_violations_without_tol,
            quantiles,
            prefix=f"{split_prefix}/epoch/",
            suffix="acc_gap_violations_without_tol",
        )

    return epoch_log


@torch.inference_mode()
def do_eval_forward_pass(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cmp: cooper.ConstrainedMinimizationProblem,
    device: Union[str, torch.device],
    _epoch: int,
):
    model.eval()
    # If the model has no masks, this is a no-op.
    model_utils.set_execution_mode_(model, masked=True)

    meters = EvalMeters()

    for batch_id, batch in enumerate(dataloader, start=1):
        train_batch = process_batch(batch, device=device)

        cmp_fn_kwargs = dict(model=model, train_batch=train_batch)
        cmp_state = cmp.compute_cmp_state(**cmp_fn_kwargs, is_training=False, epoch=_epoch, meters=meters)

        if batch_id % math.ceil(len(dataloader) / 10) == 0:
            logger.info(f"Val - Epoch {_epoch} Batch {batch_id} Acc {cmp_state.misc['metrics'].avg_acc.detach()}")

    return meters
