import pytest
import torch

import src.cmp.cyclic_buffer as cb

DEVICE = torch.device("cpu")


@pytest.fixture(params=[5, 10])
def memory(request):
    return request.param


@pytest.fixture(params=[2, 10])
def num_attributes(request):
    return request.param


# def test_helper_for_buffer(memory, num_attributes):
#     cyclic_buffer = cb.GroupCyclicBuffer(num_attributes, memory, DEVICE)

#     # Fill the buffer for the first group, but not for the second group.
#     protected_attr = torch.tensor([0] * memory + [1] * (memory - 1))
#     per_sample_accuracy = torch.tensor([1.0] * (2 * memory - 1))

#     cyclic_buffer.update(protected_attr=protected_attr, per_sample_accuracy=per_sample_accuracy)

#     # Get a batch with more samples than the buffer can store for the first group, and
#     # that fills the buffer for the second group.
#     protected_attr = torch.tensor([0] * (memory + 1) + [1] * memory)
#     per_sample_accuracy = torch.tensor([0.0] * (memory + 1) + [1.0] * memory)
#     batch_group_acc = torch.tensor([0.0, 1.0] + [0.0] * (num_attributes - 2))
#     batch_group_counts = torch.tensor([memory + 1, memory] + [0] * (num_attributes - 2))

#     # The batch information should be used for the first group, and the buffer
#     # information for the second group.
#     counts, group_acc, avg_acc = cb.update_and_query_buffer(
#         buffer=cyclic_buffer,
#         protected_attr=protected_attr,
#         per_sample_accuracy=per_sample_accuracy,
#         batch_group_acc=batch_group_acc,
#         batch_group_counts=batch_group_counts,
#     )

#     assert torch.allclose(counts, torch.tensor([memory + 1, memory] + [0] * (num_attributes - 2)))
#     assert torch.allclose(group_acc, torch.tensor([0.0, 1.0]))

#     # The average accuracy should use the batch information for the first group, and
#     # the buffer information for the second group.
#     expected_avg_acc = memory / (2 * memory + 1)
#     assert torch.allclose(avg_acc, torch.tensor(expected_avg_acc))

#     # Regardless of the returns, the buffer should be updated with the batch information.
#     buffer_counts, buffer_group_acc, buffer_avg_acc = cyclic_buffer.query()
#     buffer_indices = torch.where(buffer_counts > 0)[0]
#     assert torch.allclose(buffer_group_acc[buffer_indices], torch.tensor([0.0, 1.0]))
#     assert torch.allclose(buffer_avg_acc, torch.tensor(0.5))


def test_cyclic_buffer(memory, num_attributes):
    cyclic_buffer = cb.GroupCyclicBuffer(
        num_attributes=num_attributes, memory=memory, intersectional=False, device=DEVICE
    )

    init_counts, init_accuracies, init_avg_accuracy = cyclic_buffer.query()
    assert sum(init_counts) == 0
    assert sum(init_accuracies) == 0
    assert torch.isclose(init_avg_accuracy, torch.tensor(0.0))

    # Five 0s, three 1s
    protected_attr = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]).unsqueeze(-1)
    # 100% accuracy for every sample
    per_sample_accuracy = torch.tensor([1.0] * 8)

    cyclic_buffer.update(protected_attr=protected_attr, per_sample_metric=per_sample_accuracy.bool())
    counts, accuracies, avg_accuracy = cyclic_buffer.query()
    indices = torch.where(counts > 0)[0]

    if memory == 5:
        # The memory for group 0 is full, but not for the remainder groups. Therefore, only group 0
        # is returned by the query.
        assert torch.allclose(counts[indices], torch.tensor([5]))
        assert torch.allclose(accuracies[indices], torch.tensor([1.0]))
        assert torch.allclose(avg_accuracy, torch.tensor(1.0))
    else:
        assert sum(counts) == 0
        assert sum(accuracies) == 0

    # Apply as many samples as the memory, all incorrectly classified. This should
    # overwrite all of the previous samples and return the average accuracy of 0.
    protected_attr = torch.tensor([0] * memory + [1] * memory).unsqueeze(-1)
    per_sample_accuracy = torch.tensor([0.0] * (2 * memory))

    cyclic_buffer.update(protected_attr=protected_attr, per_sample_metric=per_sample_accuracy.bool())
    counts, accuracies, avg_accuracy = cyclic_buffer.query()
    indices = torch.where(counts > 0)[0]
    assert torch.allclose(counts[indices], torch.tensor([memory, memory]))
    assert torch.allclose(accuracies[indices], torch.tensor([0.0]))
    assert torch.allclose(avg_accuracy, torch.tensor(0.0))

    # Get 50% accuracy on the first group. Do not modify the second group.
    protected_attr = torch.tensor([0] * memory).unsqueeze(-1)
    num_wrong, num_correct = torch.ceil(torch.tensor(memory / 2)).int(), torch.floor(torch.tensor(memory / 2)).int()
    per_sample_accuracy = torch.tensor([0.0] * num_wrong + [1.0] * num_correct)

    expected_acc = (memory // 2) / memory

    cyclic_buffer.update(protected_attr=protected_attr, per_sample_metric=per_sample_accuracy.bool())
    counts, accuracies, _ = cyclic_buffer.query()
    indices = torch.where(counts > 0)[0]
    assert counts[indices].tolist() == [memory, memory]
    assert torch.allclose(accuracies[indices], torch.tensor([expected_acc, 0.0]))
