import math

import pytest

from src.datasets import load_dataset


@pytest.fixture(params=[32, 64, 256])
def batch_size(request):
    return request.param


def test_cifar100(batch_size):
    loaders, num_classes, data_shape, protected_attributes = load_dataset(
        dataset_name="cifar100",
        augment=False,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        slurm_exec=False,
    )

    assert len(loaders["train"]) == math.ceil(50000 / batch_size)
    assert len(loaders["val"]) == math.ceil(10000 / batch_size)
    assert num_classes == 100

    # Test minibatch sampling
    for split, loader in loaders.items():
        x, y, z = next(iter(loader))
        assert x.shape == (batch_size, *data_shape)
        assert x.shape[1:] == data_shape
        assert y.shape == (batch_size,)
        assert z.shape == (batch_size,)


@pytest.mark.parametrize(
    "fairface_dataset_kwargs",
    [
        {"target_attribute": "race", "protected_attributes": ["gender"]},
        {"target_attribute": "gender", "protected_attributes": ["race"]},
        {"target_attribute": "gender", "protected_attributes": ["gender", "race"]},
        {"target_attribute": "gender", "protected_attributes": []},
    ],
)
def test_fairface(fairface_dataset_kwargs, batch_size):
    num_protected_attr = len(fairface_dataset_kwargs["protected_attributes"])

    loaders, num_classes, input_size = load_dataset(
        dataset_name="fairface",
        augment=False,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        dataset_kwargs=fairface_dataset_kwargs,
    )

    num_samples = sum([len(_.dataset) for _ in loaders.values()])
    # 10,954 - val; 86,744 - train
    assert num_samples == 97698

    exp_num_classes = 7 if fairface_dataset_kwargs["target_attribute"] == "race" else 2
    assert num_classes == exp_num_classes

    # Test minibatch sampling
    for _, loader in loaders.items():
        input, target, protected_attr = next(iter(loader))

        assert input.shape[0] == batch_size
        assert input.shape[1:] == input_size
        assert target.shape == (batch_size,)
        assert protected_attr.shape[1:] == (num_protected_attr,)


@pytest.mark.parametrize(
    "utk_dataset_kwargs",
    [
        {"target_attribute": "gender", "protected_attributes": []},
        {"target_attribute": "age", "protected_attributes": [], "test_ratio": 0.2},
        {"target_attribute": "race", "protected_attributes": ["gender"]},
        {"target_attribute": "age", "protected_attributes": ["gender", "race"]},
        {"target_attribute": "gender", "protected_attributes": ["age", "race"]},
    ],
)
def test_utkface(utk_dataset_kwargs, batch_size):
    num_protected_attr = len(utk_dataset_kwargs["protected_attributes"])

    age_buckets = (10, 15, 20, 25, 30, 40, 50, 60)
    dataset_kwargs = {"age_buckets": age_buckets}
    dataset_kwargs.update(utk_dataset_kwargs)

    loaders, num_classes, input_size = load_dataset(
        dataset_name="utkface",
        augment=False,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        dataset_kwargs=dataset_kwargs,
    )

    if "race" in utk_dataset_kwargs["protected_attributes"]:
        # Columns of attr are: [race, age, gender]
        race_ix = 0

        num_samples_others = 0
        for key, loader in loaders.items():
            others_label = loader.dataset.label_mapping["race"]["Others"]
            num_samples_others += sum(loader.dataset.attr[:, race_ix] == others_label)

        num_samples = sum([len(_.dataset) for _ in loaders.values()])
        # FairGRAPE paper reports 22013 samples after removing "Others"
        # https://arxiv.org/pdf/2207.10888.pdf
        # We are keeping all samples, so we should have 23705 samples
        assert num_samples == 22013 + num_samples_others

    if utk_dataset_kwargs["target_attribute"] == "race":
        assert num_classes == 5
    elif utk_dataset_kwargs["target_attribute"] == "gender":
        assert num_classes == 2
    elif utk_dataset_kwargs["target_attribute"] == "age":
        assert num_classes == len(age_buckets) + 1

    # Test minibatch sampling
    for _, loader in loaders.items():
        input, target, protected_attr = next(iter(loader))

        assert input.shape[0] == batch_size
        assert input.shape[1:] == input_size
        assert target.shape == (batch_size,)
        assert protected_attr.shape[1:] == (num_protected_attr,)
