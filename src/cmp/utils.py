from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils import AverageMeter


@dataclass
class Batch:
    input: torch.Tensor
    target: torch.Tensor
    protected_attr: torch.Tensor = None


def unpack_sparse_and_pad(sparse_tensor: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpacks a sparse tensor into a dense tensor, and pads the tensor with
    zeros to the specified number of elements. This is useful for computing the
    gradient of a sparse tensor through a dense tensor operation."""

    dense = sparse_tensor.to_dense()
    return F.pad(dense, (0, numel - len(dense)), "constant", 0.0)


def add_dense_to_sparse(dense_tensor: Tensor, sparse_tensor: Tensor) -> Tensor:
    """Adds a constant or dense tensor to a sparse tensor. In the case of a dense
    tensor, we assume that the indices of the entries in the sparse tensor match
    those of the dense tensor."""

    if not sparse_tensor.is_coalesced() and dense_tensor.numel() > 1:
        raise RuntimeError("Unsafe addition operation with un-coalesced sparse tensor.")

    indices = sparse_tensor._indices()
    values = sparse_tensor._values()
    return torch.sparse_coo_tensor(indices, values + dense_tensor, sparse_tensor.shape)


def unpack_indices_from_protected_attr(
    protected_attr: torch.Tensor, num_protected_groups: list[int], intersectional: bool
) -> torch.Tensor:

    assert len(protected_attr.shape) == 2
    # we make the transpose in order to have every protected attribute of the same group in the same row
    # for instance, the above example will be [[0, 0, 5], [0, 1, 1]]
    indices = protected_attr.T.clone()
    if protected_attr.shape[1] != 1:
        if not intersectional:
            # If there are multiple protected attributes, we need to flatten the indices and repeat the input
            # When flattening, we need to add the cumulative sum of the number of protected groups to the indices
            # to ensure that the indices are unique across all protected attributes
            indices[1:].add_(torch.tensor(num_protected_groups, dtype=torch.long, device=indices.device).cumsum(0)[:-1])
        else:
            # similar to the above section, but we need to cover all combination of protected attributes
            indices = torch.tensor(
                np.ravel_multi_index(indices.cpu().numpy(), num_protected_groups),
                dtype=torch.long,
                device=indices.device,
            )

        # This is needed to get the matrix in the right shape for sparse_coo_tensor
        # The indices will be of shape [batch_size, 1] and input will be of shape [batch_size]
        # having it the other way will throw a RuntimeError: numel: integer multiplication overflow
        indices = indices.reshape(1, -1)

    return indices


def compute_average_by_group(
    input: torch.Tensor, protected_attr: torch.Tensor, num_protected_groups: list[int], intersectional: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the average of the input tensor by group. The input tensor is
    Args:
        input: Tensor of shape (batch_size,) containing the losses to be averaged.
        protected_attr: Tensor of shape (batch_size, len(num_protected_groups)) containing the protected attribute for each entry in the batch.
        num_protected_groups: List of integers containing the number of protected groups for each protected attribute.
        intersectional: Boolean indicating whether to compute the average by intersectional groups.
    Returns:
        Tuple of two tensors. The first tensor contains the average of the input tensor by group. The second tensor
        contains the number of entries in each group.
    Examples:
        protected_attr = [[0, 0], [0, 1] [5, 1]] -> indicates a batch of 3 entries with 2 protected attributes.
        num_protected_groups = [5, 2] -> indicates that the first protected attribute has 5 groups and the second
        protected attribute has 2 groups.
    """
    if protected_attr is None:
        raise ValueError("Cannot aggregate metrics without protected_attr")

    # Get indices based on protected attribute for each entry in the batch of
    # measurements. Need to manipulate to have the right shape since
    # torch.sparse_coo_tensor has a very rigid API.
    indices = unpack_indices_from_protected_attr(protected_attr, num_protected_groups, intersectional)
    if protected_attr.shape[1] != 1 and not intersectional:
        input = input.repeat(len(num_protected_groups))

    group_sums = torch.sparse_coo_tensor(indices, input).coalesce()
    group_counts = torch.sparse_coo_tensor(indices, torch.ones_like(input, dtype=torch.long)).coalesce()

    group_means = group_sums.values() / group_counts.values()
    group_means = torch.sparse_coo_tensor(group_sums.indices(), group_means).coalesce()

    if not intersectional:
        dense_group_means = unpack_sparse_and_pad(group_means, sum(num_protected_groups))
        dense_group_counts = unpack_sparse_and_pad(group_counts, sum(num_protected_groups))
    else:
        dense_group_means = unpack_sparse_and_pad(group_means, np.prod(num_protected_groups))
        dense_group_counts = unpack_sparse_and_pad(group_counts, np.prod(num_protected_groups))

    return dense_group_means, dense_group_counts


@torch.inference_mode()
def compute_model_acc_stats(cmp, model, dataloader, device):
    model.eval()

    avg_acc_meter = AverageMeter()
    group_acc_meter = AverageMeter()

    avg_loss_meter = AverageMeter()
    group_loss_meter = AverageMeter()

    for batch in dataloader:
        inputs, target = batch[0].to(device), batch[1].to(device)
        protected_attr = batch[2].to(device) if len(batch) == 3 else None

        batch_size = inputs.size(0)

        logits = model(inputs)
        batch_metrics = compute_batch_metrics(
            cmp, logits, target, protected_attr, cmp.num_protected_groups, cmp.intersectional
        )

        avg_acc_meter.update(batch_metrics.avg_acc.detach(), batch_size)
        group_acc_meter.update(batch_metrics.group_acc.detach(), batch_metrics.group_counts)

        avg_loss_meter.update(batch_metrics.avg_loss.detach(), batch_size)
        group_loss_meter.update(batch_metrics.group_loss.detach(), batch_metrics.group_counts)

    return {
        "avg_acc": avg_acc_meter.avg,
        "group_acc": group_acc_meter.avg,
        "avg_loss": avg_loss_meter.avg,
        "group_loss": group_loss_meter.avg,
    }


@dataclass
class BatchMetrics:
    """ """

    sample_loss: torch.Tensor
    sample_acc: torch.Tensor

    avg_loss: torch.Tensor = field(init=False)
    avg_acc: torch.Tensor = field(init=False)

    protected_attr: torch.Tensor = None
    num_protected_groups: list[int] = None
    intersectional: bool = False

    group_loss: torch.Tensor = field(init=False)
    group_acc: torch.Tensor = field(init=False)
    group_counts: torch.Tensor = field(init=False)

    def __post_init__(self):
        # Compute average metrics regardless of whether protected_attr is provided
        self.avg_loss = self.sample_loss.mean(dim=0)
        self.avg_acc = self.sample_acc.mean(dim=0)

        if self.protected_attr is None:
            pass

        averaging_kwargs = {
            "protected_attr": self.protected_attr,
            "num_protected_groups": self.num_protected_groups,
            "intersectional": self.intersectional,
        }

        assert sum(self.num_protected_groups) > 0, "Must specify num_protected_groups if protected_attr is provided"

        self.group_loss, self.group_counts = compute_average_by_group(self.sample_loss, **averaging_kwargs)
        self.group_acc, _ = compute_average_by_group(self.sample_acc, **averaging_kwargs)


def compute_batch_metrics(cmp, logits, target, protected_attr, num_protected_groups, intersectional):
    """
    Builds a generic BatchMetrics object by computing the loss and accuracy
    of the provided logits and targets. If protected_attr is provided, then
    the metrics are also aggregated by group.
    """
    sample_loss = cmp.loss_fn(logits, target)
    sample_acc = cmp.accuracy_fn(logits, target)

    return BatchMetrics(
        sample_loss=sample_loss,
        sample_acc=sample_acc,
        protected_attr=protected_attr,
        num_protected_groups=num_protected_groups,
        intersectional=intersectional,
    )


def build_metric_callables(label_smoothing: float):
    loss_fn = partial(F.cross_entropy, reduction="none", label_smoothing=label_smoothing)
    accuracy_fn = compute_per_sample_accuracy

    return loss_fn, accuracy_fn


def compute_per_sample_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_labels = torch.argmax(logits, dim=1)
    return (predicted_labels == target).float()
