from math import isclose

import pytest

from src.sparse import SparsityScheduler


def issmaller(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """Returns True if `a` is smaller than `b` with some tolerance, False otherwise"""
    return a < b + tolerance


@pytest.fixture(params=[(0, 10), (19, 20), (21, 30), (48, 51)])
def last_pruning_and_num_epochs(request):
    return request.param


@pytest.fixture
def last_pruning_epoch(last_pruning_and_num_epochs):
    return last_pruning_and_num_epochs[0]


@pytest.fixture
def num_epochs(last_pruning_and_num_epochs):
    return last_pruning_and_num_epochs[1]


@pytest.fixture(params=[0, 11])
def init_pruning_epoch(request, last_pruning_epoch):
    if request.param > last_pruning_epoch:
        pytest.skip("init_pruning_epoch must be smaller than last_pruning_epoch")

    return request.param


@pytest.fixture(params=[1.0, 0.9])
def sparsity_final(request):
    return request.param


@pytest.fixture(params=[1, 10])
def pruning_frequency(request):
    return request.param


@pytest.fixture(params=[0.1])
def sparsity_initial(request, last_pruning_epoch, sparsity_final):
    if last_pruning_epoch == 0:
        return sparsity_final

    return request.param


@pytest.fixture()
def sparsity_scheduler(last_pruning_epoch, sparsity_final, pruning_frequency, sparsity_initial, init_pruning_epoch):
    scheduler = SparsityScheduler(
        last_pruning_epoch=last_pruning_epoch,
        sparsity_final=sparsity_final,
        sparsity_initial=sparsity_initial,
        init_pruning_epoch=init_pruning_epoch,
        pruning_frequency=pruning_frequency,
    )
    return scheduler


def test_scheduler(
    num_epochs,
    last_pruning_epoch,
    sparsity_final,
    pruning_frequency,
    sparsity_initial,
    init_pruning_epoch,
    sparsity_scheduler,
):

    current_sparsity = sparsity_initial

    for epoch in range(num_epochs):
        # Simulate a training loop

        is_first_or_last_pruning_epoch = (epoch == init_pruning_epoch) or (epoch == last_pruning_epoch)

        if epoch < init_pruning_epoch:
            # Should not sparsify at this stage
            assert not sparsity_scheduler.should_sparsify(epoch)
        elif epoch > last_pruning_epoch:
            # Should not sparsify at this stage
            assert not sparsity_scheduler.should_sparsify(epoch)

        elif is_first_or_last_pruning_epoch or (epoch - init_pruning_epoch) % pruning_frequency == 0:
            # Should sparsify at this stage
            assert sparsity_scheduler.should_sparsify(epoch)

            # new_sparsity value must be between the initial and final values
            new_sparsity = sparsity_scheduler.sparsity_amount(epoch)
            assert issmaller(sparsity_initial, current_sparsity)
            assert issmaller(current_sparsity, sparsity_final)

            # new_sparsity value must be greater than the previous value
            assert issmaller(current_sparsity, new_sparsity)

            current_sparsity = new_sparsity
        else:
            # The conditions for sparsifying are not met
            assert not sparsity_scheduler.should_sparsify(epoch)

    assert isclose(current_sparsity, sparsity_final)
