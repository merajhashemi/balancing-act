from collections import deque

import numpy as np
import torch

from src.cmp.utils import unpack_indices_from_protected_attr


class GroupCyclicBuffer:
    def __init__(self, num_protected_groups: list[int], memory: int, intersectional: bool, device) -> None:

        self.num_protected_groups = num_protected_groups
        self.num_constraints = np.prod(num_protected_groups) if intersectional else np.sum(num_protected_groups)
        self.memory = memory
        self.intersectional = intersectional
        self.device = device

        # deques have O(1) append and pop operations
        self.buffers = [deque(maxlen=memory) for _ in range(self.num_constraints)]

    def reset(self) -> None:
        """Reset the buffer."""
        for buffer in self.buffers:
            buffer.clear()

    @property
    def counts(self) -> torch.Tensor:
        """Return the number of samples per group."""
        return torch.tensor([len(buffer) for buffer in self.buffers])

    def update(self, protected_attr: torch.Tensor, per_sample_metric: torch.Tensor) -> None:
        """Update the buffer with the new predictions and labels.

        Args:
            protected_attr: The protected attribute of the group.
            per_sample_metric: The per sample metric.
        """
        indices = unpack_indices_from_protected_attr(protected_attr, self.num_protected_groups, self.intersectional)
        if protected_attr.shape[1] != 1 and not self.intersectional:
            per_sample_metric = per_sample_metric.repeat(len(self.num_protected_groups))

        for i in range(self.num_constraints):
            filtered_accuracies = per_sample_metric[indices.view(-1) == i]
            self.buffers[i].extend(filtered_accuracies.tolist())

    def query(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the average metric per group for full buffers only"""
        group_counts, group_values = [], []
        for i in range(self.num_constraints):
            if len(self.buffers[i]) == self.memory:
                group_counts.append(self.memory)
                group_values.append(torch.tensor(self.buffers[i]).mean(dtype=torch.float))
            else:
                group_counts.append(0)
                group_values.append(torch.tensor(0.0))

        group_counts = torch.tensor(group_counts, device=self.device)
        group_values = torch.tensor(group_values, device=self.device)
        avg_metric = torch.sum(group_values * group_counts) / torch.sum(group_counts)
        if torch.isnan(avg_metric):
            avg_metric = torch.tensor(0.0, device=self.device)
        avg_metric = avg_metric.to(self.device)

        return group_counts, group_values, avg_metric
