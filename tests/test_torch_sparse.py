import pytest
import torch

from src.cmp.utils import add_dense_to_sparse


def test_sparse_tensor_ops():
    """Simple test to verify desired behavior in backward pass through sparse tensor
    operations"""
    indices1, values1 = torch.tensor([[0, 2]]), torch.tensor([5.3, -1.2])
    sparse_tensor1 = torch.sparse_coo_tensor(indices1, values1)

    # Create a tensor influenced by a Parameter
    x = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
    indices2 = torch.tensor([[0, 1, 2]])
    values2 = torch.stack([torch.tensor([5.0]), x, x**2]).squeeze()
    sparse_tensor2 = torch.sparse_coo_tensor(indices2, values2).coalesce()

    # Make original tensors dense and compute sum. Verify that gradient works correctly
    res0 = torch.sum(sparse_tensor1.to_dense() * sparse_tensor2.to_dense())
    res0.backward(retain_graph=True)
    assert x.grad == 2 * (0.5) * (-1.2)

    x.grad = None
    sparse_tensor3 = sparse_tensor1 + sparse_tensor2
    assert sparse_tensor3._indices().tolist() == [[0, 1, 2]]

    res1 = torch.sparse.sum(sparse_tensor1 * sparse_tensor2)
    res1.backward(retain_graph=True)
    assert x.grad == 2 * (0.5) * (-1.2)

    y = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=True)

    sparse_tensor4 = add_dense_to_sparse(-y, sparse_tensor2)
    res2 = torch.sparse.sum(sparse_tensor4)
    res2.backward()

    ground_truth_sum = torch.tensor([-2.0, -6.5, -6.75])
    assert torch.allclose(sparse_tensor4.coalesce().values(), ground_truth_sum)
    assert y.grad == -3.0


def test_sparse_unpacking():

    numel = 10
    indices1, values1 = torch.tensor([[0, 2]]), torch.tensor([5.3, -1.2])
    sparse_tensor1 = torch.sparse_coo_tensor(indices1, values1)

    dense = sparse_tensor1.to_dense()
    dense = torch.nn.functional.pad(dense, (0, numel - len(dense)), "constant", 0.0)

    assert torch.allclose(dense, torch.tensor([5.3, 0.0, -1.2] + [0.0] * (numel - 3)))
