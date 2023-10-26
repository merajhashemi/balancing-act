import pytest
import torch


@pytest.fixture(params=[32, 256])
def batch_size(request):
    return request.param


@pytest.fixture(params=[5, 1000])
def num_classes(request):
    return request.param


@pytest.fixture(params=[[5], [1000]])
def num_protected_groups(request):
    return request.param


@pytest.fixture(params=[0.3])
def ema_gamma(request):
    return request.param


@pytest.fixture
def inputs(batch_size, num_classes):
    inputs = torch.randn((batch_size, num_classes))

    if torch.cuda.is_available():
        inputs = inputs.cuda()

    return inputs


@pytest.fixture
def protected_attr(batch_size, num_protected_groups):
    if sum(num_protected_groups) == 0:
        protected_attr = None
    else:
        protected_attr = torch.randint(sum(num_protected_groups), size=(batch_size,))

        if torch.cuda.is_available():
            protected_attr = protected_attr.cuda()

    return protected_attr


@pytest.fixture
def target(batch_size, num_classes):
    # One integer label for each sample
    target = torch.randint(num_classes, size=(batch_size,))

    if torch.cuda.is_available():
        target = target.cuda()

    return target
