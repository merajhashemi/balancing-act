import logging

import torch.utils.data as torch_data
import torchvision as tv

import src.datasets.utils as utils

from .cifar100 import CIFAR100
from .fairface import FairFace
from .utkface import UTKFace

logger = logging.getLogger(__name__)

# NOTE: For **all datasets** we apply normalization based on the mean and variance of ImageNet
# [Reason] the mean and variance numbers for the other datasets were very close to ImageNet statistics, and we use
# pre-trained ImageNet models to fine-tune with our dataset.


def get_num_classes_and_protected_groups(dataset):

    if isinstance(dataset, torch_data.Subset):
        return dataset.dataset.num_classes, dataset.dataset.num_protected_groups

    # Considering a regular dataset
    return dataset.num_classes, dataset.num_protected_groups


def load_dataset(
    dataset_name: str,
    augment: bool = True,
    train_batch_size: int = 256,
    test_batch_size: int = 256,
    val_batch_size: int = 256,
    val_split_ratio: float = 0.0,
    val_split_seed: int = 0,
    slurm_exec: bool = True,
    dataset_kwargs: dict = {},
):
    """Load dataset."""

    logger.debug(f"Loading {dataset_name} dataset")

    if dataset_name == "utkface":
        assert not augment, "UTKFace dataset does not apply data augmentation."
        input_shape = (3, 200, 200)
        split_datasets = build_UTKFace(val_split_ratio, val_split_seed, slurm_exec, dataset_kwargs)

    elif dataset_name == "fairface":
        assert not augment, "Fairface dataset does not apply data augmentation."
        input_shape = (3, 224, 224)
        split_datasets = build_FairFace(val_split_ratio, val_split_seed, slurm_exec, dataset_kwargs)

    elif dataset_name == "cifar100":
        assert not augment, "Current implementation of CIFAR100 does not apply data augmentation."
        input_shape = (3, 32, 32)
        split_datasets = build_CIFAR100(val_split_ratio, val_split_seed, slurm_exec, dataset_kwargs)

    else:
        raise ValueError(f"Invalid dataset {dataset_name}.")

    num_classes, num_protected_groups = get_num_classes_and_protected_groups(split_datasets["train"])

    loaders = utils.build_dataloaders_by_split(
        datasets_by_split=split_datasets,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        val_batch_size=val_batch_size,
        num_workers=utils.get_num_workers(),
    )

    return loaders, num_classes, input_shape, num_protected_groups


def build_UTKFace(
    val_split_ratio: float = 0.0, val_split_seed: int = 0, slurm_exec: bool = True, dataset_kwargs: dict = {}
):
    # The preprocessing is based on: https://arxiv.org/abs/2205.13574
    # https://pastecode.io/s/v4nueqhc (this is code from the pruning has disparate impact paper)
    # we ignore their re-sizing to (48,48) as this would be too low resolution
    transform_train = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_val = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    split_kwargs = {
        "train": {"train": True, "transform": transform_train},
        "val": {"train": True, "transform": transform_val},
        "test": {"train": False, "transform": transform_val},
    }

    datasets_by_split = utils.build_datasets_by_split(
        dataset_class=UTKFace,
        data_path=utils.get_dataset_path("utkface", slurm_exec),
        split_kwargs=split_kwargs,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        dataset_kwargs=dataset_kwargs,
    )

    return datasets_by_split


def build_FairFace(
    val_split_ratio: float = 0.0, val_split_seed: int = 0, slurm_exec: bool = True, dataset_kwargs: dict = {}
):
    # Normalization taken from original FairFace repo
    # https://github.com/dchen236/FairFace/blob/74b4f93e527f1fb2b27b2b425227c3ded0521830/predict.py#L76
    transform_train = tv.transforms.Compose(
        [
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_val = tv.transforms.Compose(
        [
            tv.transforms.Resize(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    split_kwargs = {
        "train": {"train": True, "transform": transform_train},
        "val": {"train": True, "transform": transform_val},
        "test": {"train": False, "transform": transform_val},
    }

    datasets_by_split = utils.build_datasets_by_split(
        dataset_class=FairFace,
        data_path=utils.get_dataset_path("fairface", slurm_exec),
        split_kwargs=split_kwargs,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        dataset_kwargs=dataset_kwargs,
    )

    return datasets_by_split


def build_CIFAR100(
    val_split_ratio: float = 0.0, val_split_seed: int = 0, slurm_exec: bool = True, dataset_kwargs: dict = {}
):
    # No training augmentation applied on CIFAR100
    # we use the same transforms on train and val
    transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.507, 0.4865, 0.4409], [0.2673, 0.2564, 0.2761]),
        ]
    )

    split_kwargs = {
        "train": {"train": True, "transform": transform},
        "val": {"train": True, "transform": transform},
        "test": {"train": False, "transform": transform},
    }

    datasets_by_split = utils.build_datasets_by_split(
        dataset_class=CIFAR100,
        data_path=utils.get_dataset_path("cifar100", slurm_exec),
        split_kwargs=split_kwargs,
        val_split_ratio=val_split_ratio,
        val_split_seed=val_split_seed,
        dataset_kwargs=dataset_kwargs,
    )

    return datasets_by_split
