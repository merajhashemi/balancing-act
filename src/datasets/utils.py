import logging
import math
import os
import os.path as osp
import random
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

import torch
import torch.utils.data as torch_data
from torchvision import datasets

logger = logging.getLogger(__name__)


def get_slurm_tmpdir() -> Union[Path, None]:
    """Returns the SLURM temporary directory"""

    if "SLURM_TMPDIR" in os.environ:
        return Path(os.environ["SLURM_TMPDIR"])
    elif "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
        return Path(f"/Tmp/slurm.{job_id}.0")

    return None


def get_num_workers() -> int:
    """Automatically gets the "right" number of workers for building dataloaders."""

    def get_cpus_on_node() -> int:
        if "SLURM_CPUS_PER_TASK" in os.environ:
            return int(os.environ["SLURM_CPUS_PER_TASK"])
        elif "SLURM_JOB_ID" in os.environ:
            # Use 4 workers by default for cluster interactive sessions
            return 4
        else:
            # Use all available CPUs for local (not cluster!) execution
            return cpu_count()

    return get_cpus_on_node()


def get_dataset_path(dataset_name: str, slurm_exec: bool) -> str:
    """
    Retrieves the path in which a dataset is stored from os.environ["DATA_DIR"] file,
    also contained under the utils module.
    """

    root_dir = get_slurm_tmpdir() if slurm_exec else Path(os.environ["DATA_DIR"])
    data_path = osp.join(root_dir, dataset_name)
    if osp.exists(data_path):
        logger.debug(f"Found copy of dataset at {root_dir}")
        return data_path
    else:
        raise ValueError(f"Could not find {dataset_name} dataset at {root_dir}.")


def create_loader(
    dataset: torch_data.Dataset,
    batch_size: int,
    is_train: bool,
    num_workers: int,
    pin_memory: bool,
) -> torch_data.DataLoader:
    """
    Creates a dataloader.
    """

    if is_train:
        sampler_class = torch_data.RandomSampler
    else:
        sampler_class = torch_data.SequentialSampler

    sampler = sampler_class(dataset)

    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None,
    )

    return loader


def build_datasets_by_split(
    dataset_class: type[datasets.VisionDataset],
    data_path: str,
    split_kwargs: dict,
    val_split_ratio: float = 0,
    val_split_seed: int = 0,
    dataset_kwargs: dict = {},
) -> dict[str, datasets.VisionDataset]:
    """
    Function creates the Dataset based on the splits (train/val/test) that are necessary
    """
    datasets = {}
    for split_key in split_kwargs.keys():
        datasets[split_key] = dataset_class(data_path, **split_kwargs[split_key], **dataset_kwargs)

    if val_split_ratio == -1:
        # This is a convention for a dataset that already has a validation set. No need
        # to split the training set into train/val.
        return datasets

    if val_split_ratio == 0:
        # Validation set not requested
        if "val" in datasets.keys():
            datasets.pop("val")

        return datasets

    assert "val" in datasets.keys()

    # Split the training set into train/val given the val_split_ratio. Done by randomly
    # sampling val_split_ratio of the indices for each protected attribute.
    val_indices, train_indices = [], []
    rng = random.Random(val_split_seed)
    for indices_per_protected_attr in datasets["train"].get_protected_attr_indices():
        # Using math.ceil might yield an effective split ratio different from val_split_ratio
        # E.g. in Imagenet, this might lead to having a difference of 1000 samples
        val_size = math.ceil(len(indices_per_protected_attr) * val_split_ratio)
        rng.shuffle(indices_per_protected_attr)
        val_indices.extend(indices_per_protected_attr[:val_size])
        train_indices.extend(indices_per_protected_attr[val_size:])

    datasets["val"] = torch_data.Subset(datasets["val"], val_indices)
    datasets["train"] = torch_data.Subset(datasets["train"], train_indices)

    for split_key in datasets.keys():
        logger.info(f"Creating {dataset_class.__name__} dataset for {split_key} split.")
        logger.info(f"Number of samples: {len(datasets[split_key])}")

    return datasets


def build_dataloaders_by_split(
    datasets_by_split: dict[str, datasets.VisionDataset],
    train_batch_size: int,
    test_batch_size: int,
    val_batch_size: int,
    num_workers: int,
) -> dict[str, torch_data.DataLoader]:
    """
    Function is used to create dataloaders based on the splits
    """

    batch_sizes = {"train": train_batch_size, "val": val_batch_size, "test": test_batch_size}

    loaders = {}
    for split_key, dataset in datasets_by_split.items():
        is_train = split_key in {"train"}

        logger.info(f"Creating dataloader with {split_key} batch size: {batch_sizes[split_key]}.")

        loaders[split_key] = create_loader(
            dataset=dataset,
            batch_size=batch_sizes[split_key],
            is_train=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        logger.debug(f"Created {split_key} loader with {num_workers} num workers ")

    return loaders
