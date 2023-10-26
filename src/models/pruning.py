import logging
import math

import torch
import torch.nn as nn

from src.models.utils import get_masked_layers
from src.sparse import MaskedLayer, SparsityScheduler

logger = logging.getLogger(__name__)


@torch.no_grad()
def unstructured_magnitude_prune_layer_(layer: MaskedLayer, sparsity_amount: float):
    """Prune the smallest elements in the weight tensor

    Args:
        sparsity_amount (float): Represents the fraction of parameters to prune
            (between 0.0 and 1.0).
    """

    if not 0 <= sparsity_amount <= 1.0:
        raise ValueError("sparsity_amount should be between 0.0 and 1.0 ")

    weight, _ = layer.get_parameters(masked=True)
    k = math.ceil(sparsity_amount * weight.numel())  # number of parameters to prune
    _, smallest_idx = torch.topk(torch.abs(weight).view(-1), k=k, largest=False)

    mask = torch.ones_like(layer.weight, memory_format=torch.contiguous_format, dtype=torch.bool)
    mask.view(-1)[smallest_idx] = False  # must be contiguous for inplace operation

    assert mask.shape == layer.weight.shape

    layer.update_weight_mask_(weight_mask=mask)
    layer.update_bias_mask_(bias_mask=None)


@torch.no_grad()
def structured_magnitude_prune_layer_(layer: MaskedLayer, sparsity_amount: float):
    raise NotImplementedError


def layerwise_model_pruning_(model, sparsity_amount, sparsity_type):
    if sparsity_type == "unstructured":
        prune_layer_fn = unstructured_magnitude_prune_layer_
    elif sparsity_type == "structured":
        prune_layer_fn = structured_magnitude_prune_layer_
    else:
        raise ValueError(f"Unknown sparsity type {sparsity_type}")

    logger.info(f"Applying {sparsity_type} pruning with sparsity {sparsity_amount}")

    for layer_name, layer in get_masked_layers(model):
        prune_layer_fn(layer, sparsity_amount)


def magnitude_prune_model_(model: nn.Module, sparsity_scheduler: SparsityScheduler, epoch: int):
    """
    Apply layer-wise magnitude pruning to the model, according to the sparsity
    amount prescribed by the sparsity scheduler. Note that magnitude pruning is
    only applied on certain epochs, based on the `pruning_frequency` of the
    sparsity scheduler.
    """

    if sparsity_scheduler is not None and sparsity_scheduler.should_sparsify(epoch):
        layerwise_model_pruning_(
            model=model,
            sparsity_amount=sparsity_scheduler.sparsity_amount(epoch),
            sparsity_type=sparsity_scheduler.sparsity_type,
        )
