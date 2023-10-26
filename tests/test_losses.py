import functools
import timeit

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.cmp.utils import compute_average_by_group, unpack_sparse_and_pad


@pytest.fixture(params=[32, 64, 256])
def batch_size(request):
    return request.param


@pytest.fixture(params=[5, 100, 1000])
def num_classes(request):
    return request.param


@pytest.fixture(params=[[5], [10], [500], [1000]])
def num_protected_groups(request):
    return request.param


@pytest.fixture
def loss(batch_size, num_classes):
    logits = torch.randn(batch_size, num_classes)
    target = torch.randint(num_classes, size=(batch_size,))
    return F.cross_entropy(logits, target, reduction="none")


@pytest.fixture
def data_to_aggregate(loss, batch_size, num_protected_groups):
    protected_attr = torch.randint(sum(num_protected_groups), size=(batch_size,))
    return loss, protected_attr


def test_cross_entropy(batch_size):
    logits = torch.randn(batch_size, 5)
    target = torch.randint(5, size=(batch_size,))
    loss = F.cross_entropy(logits, target, reduction="none")
    loss2 = F.cross_entropy(logits, target)
    assert torch.allclose(loss.mean(), loss2)


def naive_aggregate_metric_by_group(loss, protected_attr):
    indices = torch.unique(protected_attr)
    counts = []
    sums = []
    for ix in indices:
        masked = loss[protected_attr == ix]
        sums.append(masked.sum())
        counts.append(masked.shape[0])

    sums = torch.tensor(sums).float()

    indices = indices.unsqueeze(0)
    group_sums = torch.sparse_coo_tensor(indices, sums).coalesce()
    group_counts = torch.sparse_coo_tensor(indices, counts).coalesce()
    group_means = group_sums.values() / group_counts.values()
    group_means = torch.sparse_coo_tensor(indices, group_means).coalesce()

    return group_means, group_counts


def test_aggregate_metric_by_group(data_to_aggregate, num_protected_groups):
    fast_means, fast_counts = compute_average_by_group(*data_to_aggregate, num_protected_groups, intersectional=False)
    naive_means, naive_counts = naive_aggregate_metric_by_group(*data_to_aggregate)

    for fast, naive in [(fast_means, naive_means), (fast_counts, naive_counts)]:
        assert torch.allclose(fast, unpack_sparse_and_pad(naive, sum(num_protected_groups)))


def test_time_aggregate_metric_by_group(data_to_aggregate, num_protected_groups):
    timit_number = 25

    lmbda = functools.partial(compute_average_by_group, *data_to_aggregate, num_protected_groups, intersectional=False)
    t_records = timeit.repeat(lmbda, repeat=5, number=timit_number)
    s_per_call = np.min(t_records) / timit_number
    print(f"Fast: {s_per_call * 1e6:.2f} usec/call -- {1 / s_per_call:.2f} calls/sec")

    lmbda = functools.partial(naive_aggregate_metric_by_group, *data_to_aggregate)
    t_records = timeit.repeat(lmbda, repeat=5, number=timit_number)
    s_per_call = np.min(t_records) / timit_number
    print(f"Naive: {s_per_call * 1e6:.2f} usec/call -- {1 / s_per_call:.2f} calls/sec")
