import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)


class FairFace(VisionDataset):
    """
    FairFace is a face image dataset which is race balanced.
    It contains 108,501 images from 7 different race groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino.
    Images were collected from the YFCC-100M Flickr dataset and labeled with race, gender, and age groups.
    Source:  https://github.com/joojs/fairface
    We use the 0.25 padding one, similar to FairGRAPE - https://github.com/Bernardo1998/FairGRAPE/blob/b677eb974bf9fee2e9bccd7a31fca6bdb0858a64/util.py#L211 and compared it to the URLs in FairFace to confirm.
    """

    def __init__(
        self,
        root: str,
        target_attribute: str,
        protected_attributes: Optional[list[str]] = [],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        target_attribute: the attribute to run the classification task
        protected_attribute: the sensitive attribute over which we calculate fairness
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.attr_names = ["race", "gender"]

        attr_to_column = {"race": 0, "gender": 1}
        assert target_attribute in self.attr_names
        self._target_filter_idx = attr_to_column[target_attribute]
        self._protected_filter_idx = [attr_to_column[_] for _ in protected_attributes]

        self._data_dir = Path(self.root) / "fairface"
        self._image_dir = self._data_dir / "fairface-img-margin025-trainval"

        self.label_mapping = {
            "race": {
                "East Asian": 0,
                "Indian": 1,
                "Black": 2,
                "White": 3,
                "Middle Eastern": 4,
                "Latino_Hispanic": 5,
                "Southeast Asian": 6,
            },
            "gender": {"Male": 0, "Female": 1},
        }

        label_path = "fairface_label_train.csv" if train else "fairface_label_val.csv"
        label_csv = pd.read_csv(self._data_dir / label_path)
        label_csv.replace(self.label_mapping, inplace=True)

        self._image_files = label_csv["file"].tolist()
        self.attr = label_csv[["race", "gender"]].values

        self.num_protected_groups = [len(self.label_mapping[_]) for _ in protected_attributes]

        self.num_classes = len(self.label_mapping[target_attribute])

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        image_path = self._image_dir / self._image_files[index]
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
        return len(self._image_files)

    def get_protected_attr_indices(self) -> list[list[int]]:
        protected_attributes = self.attr[:, self._protected_filter_idx].tolist()

        indices = defaultdict(list)
        for index, protected_attr in enumerate(protected_attributes):
            indices[tuple(protected_attr)].append(index)

        return list(indices.values())
