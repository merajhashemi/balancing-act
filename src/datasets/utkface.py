import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)


class UTKFace(VisionDataset):
    """Large Scale Face (UTKFace) Dataset `https://susanqq.github.io/UTKFace`"""

    def __init__(
        self,
        root: str,
        target_attribute: str,
        protected_attributes: Optional[list[str]] = [],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_ratio: float = 0.2,
        age_buckets: tuple[int, ...] = (10, 15, 20, 25, 30, 40, 50, 60),
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.attr_names = ["race", "age", "gender"]

        attr_to_column = {"race": 0, "age": 1, "gender": 2}
        assert target_attribute in self.attr_names
        self._target_filter_idx = attr_to_column[target_attribute]
        self._protected_filter_idx = [attr_to_column[_] for _ in protected_attributes]

        self._data_dir = Path(self.root) / "utkface" / "UTKFace"
        self._image_files = sorted(self._data_dir.glob("*.jpg"))
        self._remove_corrupt_files()

        self.age_buckets = age_buckets
        self.label_mapping = {
            "race": {"White": 0, "Black": 1, "Asian": 2, "Indian": 3, "Others": 4},
            "age": {f"Bucket_{i}": i for i in range(len(self.age_buckets) + 1)},
            "gender": {"Male": 0, "Female": 1},
        }

        # Split into train and validation
        indices = np.random.RandomState(seed=1357).permutation(len(self._image_files))
        test_size = int(test_ratio * len(indices))
        if train:
            self._indices = indices[test_size:]
        else:
            self._indices = indices[:test_size]

        # Call to self._load_attrs() must happen before filtering _image_files
        self.attr = self._load_attrs()[self._indices]
        self._image_files = [self._image_files[i] for i in self._indices]

        self.num_protected_groups = [len(self.label_mapping[_]) for _ in protected_attributes]

        self.num_classes = len(self.label_mapping[target_attribute])

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        image_path = self._image_files[index]
        image = Image.open(image_path)

        target = self.attr[index, self._target_filter_idx]

        if len(self._protected_filter_idx) == 0:
            # Create empty tensor if no protected attributes are specified
            protected_attr = torch.tensor([])
        else:
            protected_attr = self.attr[index, self._protected_filter_idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target, protected_attr

    def __len__(self) -> int:
        return len(self._indices)

    def _remove_corrupt_files(self):
        """Three of the files have missing labels. We remove them."""
        corrupted_files = [
            "61_1_20170109150557335.jpg.chip.jpg",
            "61_1_20170109142408075.jpg.chip.jpg",
            "39_1_20170116174525125.jpg.chip.jpg",
        ]
        for file_name in corrupted_files:
            self._image_files.remove(Path(self.root) / "utkface" / "UTKFace" / file_name)

    def _load_attrs(self):
        attr = np.empty((len(self._image_files), 3), dtype=int)

        for i, image_path in enumerate(self._image_files):
            labels = image_path.name.split("_")
            age = self._bucketize_age(int(labels[0]))
            gender = int(labels[1])
            race = int(labels[2])
            attr[i, :] = np.stack((race, age, gender))

        return attr

    def _bucketize_age(self, age: int) -> int:
        return np.digitize([age], self.age_buckets, right=False)[0]

    def get_protected_attr_indices(self) -> list[list[int]]:
        protected_attributes = self.attr[:, self._protected_filter_idx].tolist()

        indices = defaultdict(list)
        for index, protected_attr in enumerate(protected_attributes):
            indices[tuple(protected_attr)].append(index)

        return list(indices.values())
