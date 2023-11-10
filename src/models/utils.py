from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

import src.sparse as _sparse


def patch_resnet_state_dict_(state_dict: dict) -> None:
    """Edit a model state_dict to be compatible with pretrained ResNet models
    from Pytorch."""

    for k in list(state_dict.keys()):
        _k = k.replace("downsample.0", "downsample_conv")
        _k = _k.replace("downsample.1", "downsample_bn")
        state_dict[_k] = state_dict.pop(k)


def patch_mobilenet_state_dict_(state_dict: dict) -> None:
    for k in list(state_dict.keys()):
        new_k = k.replace("features.1.conv.2", "features.1.bn")
        new_k = new_k.replace("conv.3", "bn")
        new_k = new_k.replace("0.0", "0.conv1")
        new_k = new_k.replace("1.0", "1.conv1")
        new_k = new_k.replace("18.0", "18.conv1")
        new_k = new_k.replace("0.1", "0.bn1")
        new_k = new_k.replace("1.1", "1.bn1")
        new_k = new_k.replace("18.1", "18.bn1")

        state_dict[new_k] = state_dict.pop(k)


def load_utkface_model(model_placeholder: nn.Module, checkpoint_dir: str, target_attribute: str) -> nn.Module:
    """Load the pretrained MobileNetV2 UTKFace model."""

    pretrained_state_dict = torch.load(Path(checkpoint_dir) / f"mobilenet_v2_utkface_{target_attribute}.pt")
    patch_mobilenet_state_dict_(pretrained_state_dict)
    populate_dummy_extra_states_(model_placeholder, pretrained_state_dict)

    model_placeholder.load_state_dict(pretrained_state_dict)

    return model_placeholder


def load_fairface_model(model_placeholder: nn.Module, checkpoint_dir: str, target_attribute: str) -> nn.Module:
    """Load the pretrained fairface model."""
    if target_attribute == "race":
        # The order of the race attribute in the pretrained model is different
        # from the order of the classes in our dataset class
        idx = [3, 5, 1, 0, 6, 2, 4]
    elif target_attribute == "gender":
        idx = [7, 8]
    else:
        raise ValueError(f"Unknown target attribute {target_attribute}")

    pretrained_state_dict = torch.load(Path(checkpoint_dir) / "res34_fair_align_multi_7_20190809.pt")
    pretrained_state_dict["fc.weight"] = pretrained_state_dict["fc.weight"][idx]
    pretrained_state_dict["fc.bias"] = pretrained_state_dict["fc.bias"][idx]

    patch_resnet_state_dict_(pretrained_state_dict)
    populate_dummy_extra_states_(model_placeholder, pretrained_state_dict)

    model_placeholder.load_state_dict(pretrained_state_dict)

    return model_placeholder


def load_cifar100_model(model_placeholder: nn.Module, checkpoint_dir: str) -> nn.Module:
    """
    Load the pretrained CIFAR100 model. We use the models
    available at https://github.com/chenyaofo/pytorch-cifar-models
    """

    pretrained_state_dict = torch.load(Path(checkpoint_dir) / "cifar100_resnet56-f2eff4c8.pt")

    patch_resnet_state_dict_(pretrained_state_dict)
    populate_dummy_extra_states_(model_placeholder, pretrained_state_dict)

    model_placeholder.load_state_dict(pretrained_state_dict)

    return model_placeholder


def populate_dummy_extra_states_(model: nn.Module, state_dict: dict) -> None:
    """Populate the `_extra_state` buffers of the MaskedLayers in the model
    with dummy values. This is useful for loading a pretrained model that does
    not have these buffers.
    """

    # If a layer of the model has an `_extra_state` and we are working with MaskedLayers,
    # this means that the layer has a mask buffer. We need to add a placeholder for
    # this buffer when loading the pretrained state dict.
    for k, v in model.state_dict().items():
        if k.endswith("._extra_state") and k not in state_dict:
            state_dict[k] = v


def get_layers(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Create an iterator over the layers in the model. Only take the leaf
    modules (skip container modules like `Sequential`).
    """
    for layer_name, layer in model.named_modules():
        if next(layer.children(), None) is None:
            yield layer_name, layer


def get_masked_layers(model: nn.Module) -> Iterable[tuple[str, _sparse.MaskedLayer]]:
    """Create an iterator over the MaskedLayers in the model."""
    for layer_name, layer in get_layers(model):
        if isinstance(layer, _sparse.MaskedLayer):
            yield layer_name, layer


def get_regular_layers(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    """Create an iterator over trainable but non-MaskedLayers in the model."""
    regular_types = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)

    for layer_name, layer in get_layers(model):
        if type(layer) in regular_types:
            yield layer_name, layer


def set_execution_mode_(model: nn.Module, masked: bool) -> None:
    """Set the execution mode of the modules corresponding to MaskedLayers inside
    the model to either "masked" or "non-masked"=dense. For models without
    MaskedLayers, this function is a no-op.
    """

    for layer_name, layer in get_masked_layers(model):
        layer.do_masked_forward = masked
