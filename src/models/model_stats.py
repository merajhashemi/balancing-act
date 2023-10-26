from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn as nn

import src.models.utils as model_utils
import src.sparse as sparse


@dataclass
class ModelStats:
    """
    Model-level statistics, including sparsity and L2 regularization.

    Args:
        layer_stats: Dict containing a `LayerStats` object for each of the
            layers in the model.
        num_params: Total number of trainable parameters in the model. This
            counts both the weights and the biases.
        num_active_params: Number of active parameters in the model for the
            current masks. This count matches `num_params` for models without
            any MaskedLayers.
        num_sparse_params: Number of sparsifiable parameters in the model layer.
            This only counts parameters that can be removed through
            sparsification: for example, by applying magnitude pruning.
        num_active_sparse_params: Number of active sparsifiable parameters in
            the model.
        sq_l2_norm: Squared L2 norm of the model's parameters. This counts the
            squared L2 norm for both weights and biases.
    """

    layer_stats: dict[str, sparse.LayerStats] = field(repr=False)

    num_params: int
    num_active_params: Union[int, torch.Tensor]

    num_sparse_params: int
    num_active_sparse_params: Union[int, torch.Tensor]

    sq_l2_norm: Optional[Union[float, torch.Tensor]] = None


def create_dense_layer_stats(layer: Union[nn.Conv2d, nn.Linear, nn.BatchNorm2d]) -> sparse.LayerStats:
    """Auxiliary function to create a LayerStats object for standard dense
    Pytorch layers.
    """
    assert type(layer) in (nn.Conv2d, nn.Linear, nn.BatchNorm2d)

    num_params = sum([param.numel() for param in layer.parameters()])
    num_active_params = num_params

    if isinstance(layer, nn.BatchNorm2d):
        sq_l2_norm = 0.0
    else:
        sq_l2_norm = torch.linalg.norm(layer.weight) ** 2
        sq_l2_norm += torch.linalg.norm(layer.bias) ** 2 if layer.bias is not None else 0.0

    num_sparse_params, num_active_sparse_params = 0, 0

    return sparse.LayerStats(
        layer_type=layer.__class__.__name__,
        num_params=num_params,
        num_active_params=num_active_params,
        num_sparse_params=num_sparse_params,
        num_active_sparse_params=num_active_sparse_params,
        sq_l2_norm=sq_l2_norm,
    )


def model_sq_l2_norm(model: torch.nn.Module) -> torch.Tensor:
    """Standalone utility to compute a model's squared L2 norm, without having
    to compute the complete ModelStats object."""

    sq_l2_norm = torch.tensor([0.0], device=next(model.parameters()).device)

    for layer_name, layer in model_utils.get_layers(model):

        if isinstance(layer, nn.BatchNorm2d):
            # BatchNorm2d layers do not count towards the l2 computation
            # This includes also sparse.MaskedBatchNorm2d layers
            pass
        elif isinstance(layer, sparse.MaskedLayer):
            with torch.no_grad():
                _weight, _bias = layer.get_parameters(masked=True)

            sq_l2_norm = torch.linalg.norm(_weight) ** 2
            sq_l2_norm += 0 if _bias is None else torch.linalg.norm(_bias) ** 2

        elif isinstance(layer, (nn.Conv2d, nn.Linear)):
            sq_l2_norm = torch.linalg.norm(layer.weight) ** 2
            sq_l2_norm += 0 if layer.bias is None else torch.linalg.norm(layer.bias) ** 2

    return sq_l2_norm


def get_model_stats(model: torch.nn.Module) -> ModelStats:

    layer_stats = {}

    for layer_name, layer in model_utils.get_layers(model):

        if isinstance(layer, (sparse.MaskedLayer, sparse.MaskedBatchNorm2d)):
            layer_stats[layer_name] = layer.layer_stats()
        elif isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            layer_stats[layer_name] = create_dense_layer_stats(layer)

    attr_names = [
        "num_params",
        "num_active_params",
        "num_sparse_params",
        "num_active_sparse_params",
        "sq_l2_norm",
    ]

    model_stats_kwargs = {}

    for attr_name in attr_names:
        per_layer_values = []
        for layer_name in layer_stats:
            attr_value = getattr(layer_stats[layer_name], attr_name)
            if attr_value is not None:
                per_layer_values.append(attr_value)
        model_stats_kwargs[attr_name] = sum(per_layer_values)

    return ModelStats(layer_stats=layer_stats, **model_stats_kwargs)
