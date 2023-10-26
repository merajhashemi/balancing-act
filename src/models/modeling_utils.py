from typing import Any, Type, Union

import torch
import torch.nn as nn

import src.sparse as sparse

BNLayer = Union[nn.BatchNorm2d, sparse.MaskedBatchNorm2d]
Conv2dLayer = Union[
    nn.Conv2d,
    sparse.MaskedConv2d,
]


def choose_correct_bn_type(previous_conv: Conv2dLayer, norm_layer: Type[BNLayer]) -> Type[BNLayer]:
    """Choose the correct batch normalization layer based on the previous
    convolutional layer.
    """
    if type(previous_conv) == nn.Conv2d:
        return nn.BatchNorm2d
    elif isinstance(previous_conv, sparse.MaskedConv2d):
        if not (issubclass(norm_layer, sparse.MaskedBatchNorm2d)):
            raise ValueError("A MaskedConv2d layer must be followed by a subclass of MaskedBatchNorm2d.")
        return norm_layer
    else:
        raise ValueError(f"Unknown convolutional layer type {type(previous_conv)}.")


def create_general_conv2d(
    conv_layer: Type[Conv2dLayer],
    use_bias: bool,
    masked_layer_kwargs: dict,
    **kwargs: Any,
) -> Conv2dLayer:

    # We overwrite masked_layer_kwargs for nn.Conv2d layers
    if conv_layer == nn.Conv2d:
        masked_layer_kwargs = {}

    return conv_layer(bias=use_bias, **masked_layer_kwargs, **kwargs)


def apply_bn_to_conv_out(
    bn: BNLayer,
    conv_out: Union[torch.Tensor, sparse.masked_layers.MaskedForwardOutput],
):
    """
    Apply a BN layer to the result of a convolutional layer.
    """

    if isinstance(conv_out, torch.Tensor):
        return bn(conv_out)
    elif isinstance(bn, sparse.MaskedBatchNorm2d):
        # Masked layers return a tuple (out, mask), so we unpack it before
        # applying the MaskedConv2d layer.
        return bn(*conv_out)
