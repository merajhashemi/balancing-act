import pickle
import tempfile

import pytest
import torch
import torch.nn as nn
import torchvision

import src.models as _models
import src.models.pruning as _pruning
import src.models.utils as model_utils
import src.sparse as _sparse


@pytest.fixture(
    params=[
        {"num_classes": 2, "input_shape": (3, 128, 128)},
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


@pytest.fixture(params=["ResNet18", "ResNet50"])
def resnet_type(request):
    """Fixture for the type of ResNet model to test."""
    return request.param


@pytest.fixture
def torchvision_resnet(resnet_type, data_kwargs):
    model_type = getattr(torchvision.models, resnet_type.lower())
    model = model_type(weights=None, num_classes=data_kwargs["num_classes"])

    if torch.cuda.is_available():
        model = model.cuda()

    return model


@pytest.fixture
def resnet_model(resnet_type, data_kwargs, layer_types):
    model_class = getattr(_models, resnet_type)
    model = model_class(data_kwargs["num_classes"], **layer_types)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def test_resnet_creation(resnet_model, torchvision_resnet):
    """Test the instantiation of a ResNet model with potentially sparse layers
    and verify that the number of total parameters matches the sum of
    sparsifiable and non-sparsifiable parameters."""

    # The first conv layer is a regular nn.Conv2d, and the FC layer is a nn.Linear
    assert type(resnet_model.conv1) == torch.nn.Conv2d
    assert type(resnet_model.fc) == torch.nn.Linear

    model_stats = _models.get_model_stats(resnet_model)
    torchvision_num_params = sum([_.numel() for _ in torchvision_resnet.parameters()])

    # Ensure that the number of parameters in the model is the same as the torchvision model.
    assert model_stats.num_params == torchvision_num_params

    # Ensure that any gap between num_params and num_sparse_params matches
    # exactly non-sparsifiable layers
    num_non_sparsifiable = 0

    for layer_name, layer in model_utils.get_regular_layers(resnet_model):
        num_non_sparsifiable += model_stats.layer_stats[layer_name].num_params

    alternate_count = model_stats.num_sparse_params + num_non_sparsifiable
    assert model_stats.num_params == alternate_count

    if resnet_model._conv_layer == _sparse.MaskedConv2d:
        assert model_stats.num_sparse_params > 0


@pytest.mark.parametrize("do_masked_forward", [True, False])
def test_resnet_forward_shape(do_masked_forward, resnet_model, torchvision_resnet, data_kwargs, bloat_input):
    """ " Test that the forward pass of a ResNet model with potentially sparse
    layers has the same shape as the forward pass of a torchvision ResNet model.
    """
    model_utils.set_execution_mode_(resnet_model, masked=do_masked_forward)
    out1 = resnet_model(bloat_input)
    out2 = torchvision_resnet(bloat_input)

    # Check that forwards have the right shapes
    assert out1.shape == out2.shape
    assert out2.shape == (bloat_input.shape[0], data_kwargs["num_classes"])


def alter_model_density_(model: nn.Module, value: float):

    if len(list(model_utils.get_masked_layers(model))) > 0:
        # Model has masked layers, but not gated layers
        # Multiply by 100 to saturate the sigmoid
        sparsity_amount = 1 - torch.sigmoid(100 * torch.tensor(value)).item()
        _pruning.layerwise_model_pruning_(model, sparsity_amount, "unstructured")
    else:
        raise ValueError("Model has no sparsifiable layers.")


def test_resnet_model_stats(resnet_model, bloat_input, layer_types):
    """Verify monotonicity of model_stats regarding number of active parameters
    and that the model outputs for "very dense" and "very sparse" models are
    different.
    """

    if layer_types["conv_layer"] in [nn.Conv2d]:
        pytest.skip("This test only applies to models with sparse layers.")

    model_utils.set_execution_mode_(resnet_model, masked=False)
    unmasked_output = resnet_model(bloat_input)
    model_stats = _models.get_model_stats(resnet_model)

    # Make the model fully *dense* by setting its weight_log_alpha to a very
    # negative value
    alter_model_density_(resnet_model, 10.0)

    masked_dense_output = resnet_model(bloat_input)
    dense_model_stats = _models.get_model_stats(resnet_model)

    # Make the model very *sparse* by setting its weight_log_alpha to a very
    # negative value
    alter_model_density_(resnet_model, -10.0)
    model_utils.set_execution_mode_(resnet_model, masked=True)

    masked_sparse_output = resnet_model(bloat_input)
    sparse_model_stats = _models.get_model_stats(resnet_model)

    assert not torch.allclose(unmasked_output, masked_sparse_output)
    assert not torch.allclose(masked_sparse_output, masked_dense_output)

    for layer_name in model_stats.layer_stats.keys():

        stats = model_stats.layer_stats[layer_name]
        dense_stats = dense_model_stats.layer_stats[layer_name]
        sparse_stats = sparse_model_stats.layer_stats[layer_name]

        for attr_name in ["num_active_params"]:  # , "num_active_sparse_params"]:

            # More parameters should be active for the denser model
            assert 0.0 <= getattr(sparse_stats, attr_name) <= getattr(stats, attr_name)
            assert getattr(stats, attr_name) <= getattr(dense_stats, attr_name)

        for attr_name in ["num_params", "num_sparse_params"]:
            # All models are the same, just with different log_alpha parameters
            # so the total number of (sparsifiable) parameters should match.
            assert getattr(sparse_stats, attr_name) == getattr(stats, attr_name)
            assert getattr(stats, attr_name) == getattr(dense_stats, attr_name)


def test_model_pickling(resnet_model):
    """Ensure that the model can be pickled"""

    with tempfile.TemporaryFile(mode="w+b") as temp_file:
        pickle.dump(resnet_model, temp_file)
