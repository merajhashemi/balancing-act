from collections import defaultdict
from typing import Any, Callable, Optional

import torch
import torchvision.datasets


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(
        self,
        root: str,
        target_attributes: Optional[tuple[str, ...]] = None,
        protected_attributes: Optional[tuple[str, ...]] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        has_protected_attributes: bool = True,
    ):
        """
        `has_protected_attributes` is used as a flag to determine how to interpret the `None` passes to `protected_attributes`
        if the value of `has_protected_attributes` is True, then we use all 100 classes as protected, otherwsie we use None of them
        """
        super().__init__(root, train, transform=transform, target_transform=target_transform)

        # By default, we use all attributes as protected and targeted
        # class_to_idx (dict): Dict with items (class_name, class_index) from the torchvision dataset
        # targets (list): The class_index value for each image in the dataset from the torchvision dataset
        # classes (list): List of the class name tuples from the torchvision dataset
        if target_attributes is not None:
            raise NotImplementedError("Handling different subsets of target attribute not implemented.")

        # We make no assumptions about the protected attribute. This can be empty
        # as well, to have a baseline model which doesn't have any protected attributes
        # when the protected group is `None` and has_protected_attributes is `True` we
        # take all the classes as protected.
        if protected_attributes is None:
            if has_protected_attributes is True:
                self.num_protected_groups = [100]
            else:
                self.num_protected_groups = [0]
        else:
            raise NotImplementedError("Does not support having a subset of classes as protected")

        self.protected_attr = protected_attributes
        self.has_protected_attributes = has_protected_attributes
        self.num_classes = len(self.classes)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:

        # Super call handles formatting and transforms of image and targets
        image, target = super().__getitem__(index)

        if self.protected_attr is None:
            if self.has_protected_attributes is True:
                protected_attribute = torch.tensor(target, dtype=torch.long).unsqueeze(-1)
            else:
                protected_attribute = torch.tensor([])
        else:
            raise NotImplementedError("Does not support having a subset of classes as protected")

        return image, target, protected_attribute

    def get_protected_attr_indices(self) -> list[list[int]]:
        if not self.has_protected_attributes:
            return [list(range(len(self)))]

        protected_attributes = self.targets

        indices = defaultdict(list)
        for index, protected_attr in enumerate(protected_attributes):
            indices[protected_attr].append(index)

        return list(indices.values())
