from operator import attrgetter

import pytest
import torch
import torch.nn as nn
import torchvision

import src.models.utils as model_utils
import src.sparse as _sparse
from src.models import MobileNet_V2


@pytest.fixture(
    params=[
        {"num_classes": 1000, "input_shape": (3, 224, 224)},
    ]
)
def data_kwargs(request):
    return request.param


@pytest.fixture(
    params=[
        (nn.Conv2d, nn.BatchNorm2d),
        (_sparse.MaskedConv2d, _sparse.MaskedBatchNorm2d),
    ]
)
def layer_types(request):
    """Fixture for the type of convolutional and BN layers used inside the model."""
    return {"conv_layer": request.param[0], "norm_layer": request.param[1]}


@pytest.fixture
def bloat_input(data_kwargs):
    """Generate random input with matching shape and batch size of 10."""
    x = torch.randn(10, *data_kwargs["input_shape"])
    if torch.cuda.is_available():
        x = x.cuda()

    return x


@pytest.fixture
def torchvision_mobilenet(data_kwargs):
    pre_weights = attrgetter("MobileNet_V2_Weights.IMAGENET1K_V1")(torchvision.models)
    model = torchvision.models.mobilenet_v2(weights=pre_weights, num_classes=data_kwargs["num_classes"])
    if torch.cuda.is_available():
        model = model.cuda()

    return model


@pytest.fixture
def mobilenet_model(data_kwargs):
    model = MobileNet_V2(num_classes=data_kwargs["num_classes"])
    if torch.cuda.is_available():
        model = model.cuda()

    return model


@pytest.mark.parametrize("do_masked_forward", [True, False])
def test_mobilenet_forward_pass(do_masked_forward, mobilenet_model, torchvision_mobilenet, data_kwargs, bloat_input):
    """ " Test that the forward pass of a MobileNetV2 model with potentially sparse
    layers has the same shape as the forward pass of a torchvision ResNet model.
    """
    tv_model_state_dict = torchvision_mobilenet.state_dict()
    model_utils.patch_mobilenet_state_dict_(tv_model_state_dict)
    mobilenet_model.load_state_dict(tv_model_state_dict)
    model_utils.set_execution_mode_(mobilenet_model, masked=do_masked_forward)

    mobilenet_model.eval()
    torchvision_mobilenet.eval()

    out1 = mobilenet_model(bloat_input)
    out2 = torchvision_mobilenet(bloat_input)

    # Check that forwards have the right shapes
    assert out1.shape == out2.shape
    assert out2.shape == (bloat_input.shape[0], data_kwargs["num_classes"])
    # check if the forwards match
    assert torch.allclose(out1, out2)
