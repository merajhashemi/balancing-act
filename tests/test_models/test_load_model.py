import ml_collections as mlc
import pytest
import torch
from torch import nn

import src.cmp as _cmp
from src.sparse import MaskedBatchNorm2d, MaskedConv2d
from src.utils.experiment_utils import construct_models


@pytest.fixture(params=[32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 10])
def num_classes(request):
    return request.param


@pytest.fixture(params=[(3, 128, 128), (3, 256, 256)])
def input_shape(request):
    return request.param


@pytest.fixture(params=[(nn.Conv2d, nn.BatchNorm2d), (MaskedConv2d, MaskedBatchNorm2d)])
def conv_and_norm_layer(request):
    return request.param


@pytest.fixture(params=["ResNet18", "ResNet34"])
def model_name(request):
    return request.param


def get_config(model_name, input_shape, num_classes, conv_layer, norm_layer):
    config = mlc.ConfigDict()
    config.model = mlc.ConfigDict()
    config.model.model_name = model_name
    config.model.input_shape = input_shape
    config.model.num_classes = num_classes
    config.model.conv_layer = conv_layer
    config.model.norm_layer = norm_layer

    config.data = mlc.ConfigDict()
    config.data.dataset_name = "foo"

    config.train = mlc.ConfigDict()
    config.train.cmp_class = None
    config.train.pretrained_model_runid = None

    config.train.cmp_class = _cmp.BaselineProblem

    return config


def test_construct_models(model_name, batch_size, num_classes, conv_and_norm_layer, input_shape):
    config = get_config(model_name, input_shape, num_classes, *conv_and_norm_layer)
    input = torch.randn((batch_size, *input_shape))
    model, _ = construct_models(config, num_classes, input_shape)

    assert model(input).shape == (batch_size, num_classes)
