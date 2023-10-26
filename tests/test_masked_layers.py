import copy

import pytest
import torch

import src.models.pruning as pruning
import src.sparse as sparse


@pytest.fixture(params=[(3, 2), (4, 4)])
def layer_in_out_dim(request):
    return request.param


@pytest.fixture(params=[sparse.MaskedLinear, sparse.MaskedConv2d])
def layer_type(request):
    return request.param


@pytest.mark.parametrize("sparsity_amount", [0.0, 0.5, 1.0])
def test_magnitude_prune_float(sparsity_amount, layer_type, layer_in_out_dim):
    """Tests magnitude pruning function by pruning a conv layer and comparing it
    to the known result."""

    in_dim, out_dim = layer_in_out_dim
    if layer_type == sparse.MaskedConv2d:
        layer = layer_type(in_channels=in_dim, out_channels=out_dim, kernel_size=3)
    else:
        layer = layer_type(in_features=in_dim, out_features=out_dim)

    # Get index of largest and smallest weights in magnitude
    smallest_weight_idx = torch.argmin(torch.abs(layer.weight))
    largest_weight_idx = torch.argmax(torch.abs(layer.weight))
    largest_weight = layer.weight.flatten()[largest_weight_idx]

    pruning.unstructured_magnitude_prune_layer_(layer, sparsity_amount)

    weight, _ = layer.get_parameters(True)

    # Check that the smallest weight is now zero and the largest weight is unchanged
    if sparsity_amount > 0:
        assert weight.flatten()[smallest_weight_idx] == 0
    if sparsity_amount < 1:
        assert weight.flatten()[largest_weight_idx] == largest_weight

    # Check that the number of active parameters is correct
    num_active_params = torch.count_nonzero(weight)
    num_params = torch.numel(layer.weight)

    assert torch.isclose(num_active_params / num_params, torch.tensor(1 - sparsity_amount))


@pytest.mark.parametrize("sparsity_amount", [0.0, 0.5, 1.0])
def test_layer_forward(sparsity_amount, layer_type, layer_in_out_dim):
    in_dim, out_dim = layer_in_out_dim
    if layer_type == sparse.MaskedConv2d:
        layer = layer_type(in_channels=in_dim, out_channels=out_dim, kernel_size=3)
        input = torch.randn(1, in_dim, 32, 32)
    else:
        layer = layer_type(in_features=in_dim, out_features=out_dim)
        input = torch.randn(1, in_dim)

    pruned_layer = copy.deepcopy(layer)
    pruning.unstructured_magnitude_prune_layer_(pruned_layer, sparsity_amount)

    pruned_layer.do_masked_forward = True
    dense_output, _ = layer(input)
    pruned_output, _ = pruned_layer(input)

    if sparsity_amount == 0:
        assert torch.allclose(dense_output, pruned_output)
    else:
        assert not torch.allclose(dense_output, pruned_output)
