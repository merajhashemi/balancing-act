import pytest
import torch

import src.models as models
import src.models.utils as model_utils
import src.sparse as sparse


@pytest.fixture(
    params=[
        {"num_classes": 10, "input_shape": (28, 28)},
        {"num_classes": 1000, "input_shape": (100,)},
    ]
)
def data_kwargs(request):
    return request.param


@pytest.fixture(params=[[], [5, 10], [100, 100]])
def hidden_dims(request):
    return request.param


@pytest.fixture(params=[True, False])
def bias(request):
    return request.param


@pytest.fixture(
    params=[
        torch.nn.Linear,
        sparse.MaskedLinear,
    ]
)
def layer_type(request):
    """Fixture for the type of linear layer used inside the model."""
    return request.param


@pytest.fixture
def bloat_input(data_kwargs):
    """Generate random input with matching shape and batch size of 10."""
    x = torch.randn(10, *data_kwargs["input_shape"])
    if torch.cuda.is_available():
        x = x.cuda()

    return x


@pytest.fixture
def mlp_model(data_kwargs, hidden_dims, bias, layer_type):

    model = models.MLP(
        input_shape=data_kwargs["input_shape"],
        hidden_dims=hidden_dims,
        num_classes=data_kwargs["num_classes"],
        bias=bias,
        act_fn=torch.nn.ReLU,
        layer_type=layer_type,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def test_mlp_creation(mlp_model, hidden_dims):
    # This includes regular and masked linear layers
    linear_layers = [_ for _ in mlp_model.main if isinstance(_, torch.nn.Linear)]
    assert len(linear_layers) == (len(hidden_dims) + 1)


@pytest.mark.parametrize("do_masked_forward", [True, False])
def test_mlp_forward(do_masked_forward, mlp_model, data_kwargs, bloat_input):
    """Test the forward pass of an MLP model."""

    model_utils.set_execution_mode_(mlp_model, masked=do_masked_forward)
    out = mlp_model(bloat_input)

    assert out.shape == (bloat_input.shape[0], data_kwargs["num_classes"])
