import pytest

from src.cmp import BaselineProblem
from src.cmp import utils as cmp_utils

from .cmp_fixtures import *


def test_meters_init(inputs, target, protected_attr, num_protected_groups):
    """Ensure that cmp.meters are initialized properly: to contain *new* meters."""
    cmp = BaselineProblem(num_protected_groups=num_protected_groups)

    # At initialization, the meters should be empty
    assert cmp.meters.group_loss.new_count == 0

    # No need to create a model, just pass inputs into an Identity function
    batch = cmp_utils.Batch(input=inputs, target=target, protected_attr=protected_attr)
    state = cmp.compute_cmp_state(
        model=torch.nn.Identity(),
        train_batch=batch,
        meters=cmp.meters,
        is_training=True,
        epoch=0,
    )

    new_cmp = BaselineProblem(num_protected_groups=num_protected_groups)

    # Ensure that new meter attributes are fresh
    assert new_cmp.meters.group_loss.new_count == 0


def test_BaselineProblem(inputs, target, protected_attr, num_protected_groups):
    if (protected_attr is not None) and len(protected_attr.shape) != 1:
        raise ValueError("This test expects protected_attr to be a 1D tensor")

    cmp = BaselineProblem(num_protected_groups=num_protected_groups)

    # No need to create a model, just pass inputs into an Identity function
    batch = cmp_utils.Batch(input=inputs, target=target, protected_attr=protected_attr)
    state = cmp.compute_cmp_state(
        model=torch.nn.Identity(),
        train_batch=batch,
        meters=cmp.meters,
        is_training=True,
        epoch=0,
    )

    # Check that the loss is a scalar
    assert state.loss.shape == torch.Size([])

    if protected_attr is not None:
        unique_attrs = torch.unique(protected_attr)
        for key, metric in state.misc.items():
            if key.endswith("_by_group"):
                group_sum, group_count = metric
                # Verify that the number of groups matches the number of unique
                # elements in the protected_attr tensor
                assert len(unique_attrs) == group_sum.indices().shape[-1]
