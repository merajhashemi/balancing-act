import numpy as np
import pytest
import torch
import torch.nn as nn

import src.models.utils as model_utils
import src.sparse as _sparse
from src.datasets import load_dataset
from src.models import CifarResNet56
from src.models.utils import load_cifar100_model


@pytest.fixture(
    params=[
        {"num_classes": 100, "input_shape": (3, 32, 32)},
    ]
)
def data_kwargs(request):
    return request.param


@pytest.fixture(
    params=[
        (nn.Conv2d, nn.BatchNorm2d),
        (_sparse.MaskedConv2d, _sparse.MaskedBatchNorm2d),
    ]
)
def layer_types(request):
    """Fixture for the type of convolutional and BN layers used inside the model."""
    return {"conv_layer": request.param[0], "norm_layer": request.param[1]}


@pytest.fixture
def bloat_input(data_kwargs):
    """Generate random input with matching shape and batch size of 10."""
    x = torch.randn(10, *data_kwargs["input_shape"])
    if torch.cuda.is_available():
        x = x.cuda()

    return x


@pytest.fixture
def cifarresnet_model(data_kwargs):
    model = CifarResNet56(num_classes=data_kwargs["num_classes"])
    if torch.cuda.is_available():
        model = model.cuda()

    return model


@pytest.fixture
def dataloader():
    batch_size = 256
    dataset_kwargs = {"target_attributes": None, "protected_attributes": None, "has_protected_attributes": True}
    loaders, num_classes, input_size, num_protected_groups = load_dataset(
        dataset_name="cifar100",
        augment=False,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        dataset_kwargs=dataset_kwargs,
    )


def compute_model_accuracy(model, dataloader):
    model.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    running_correct, running_seen = 0, 0

    with torch.inference_mode():
        for batch in dataloader:
            input, target = batch[0].to(device), batch[1].to(device)
            output = model(input)
            preds = output.argmax(dim=1)

            correct_preds = torch.sum((preds == target).float()).item()
            running_correct += correct_preds
            running_seen += input.shape[0]

    accuracy = running_correct / running_seen
    print(f"Accuracy {accuracy}")

    return accuracy


@pytest.mark.parametrize("do_masked_forward", [True, False])
def test_cifarresnet_forward_pass(do_masked_forward, cifarresnet_model, data_kwargs, bloat_input):
    """ " Test that the forward pass of a MobileNetV2 model with potentially sparse
    layers has the same shape as the forward pass of a torchvision ResNet model.
    """

    model_utils.set_execution_mode_(cifarresnet_model, masked=do_masked_forward)

    cifarresnet_model.eval()
    out = cifarresnet_model(bloat_input)

    assert out.shape == (bloat_input.shape[0], data_kwargs["num_classes"])


@pytest.mark.parametrize("do_masked_forward", [True, False])
def test_cifarresnet_load_from_checkpoint(do_masked_forward, cifarresnet_model, data_kwargs, bloat_input):
    """ " Test that the forward pass of a MobileNetV2 model with potentially sparse
    layers has the same shape as the forward pass of a torchvision ResNet model.
    """
    # load model from checkpoint
    checkpoint_path = "/network/scratch/g/gallegoj/sparse_fairness/checkpoints"
    cifarresnet_model = load_cifar100_model(cifarresnet_model, checkpoint_path)
    model_utils.set_execution_mode_(cifarresnet_model, masked=do_masked_forward)
    cifarresnet_model.eval()

    batch_size = 256
    dataset_kwargs = {"target_attributes": None, "protected_attributes": None, "has_protected_attributes": True}
    loaders, num_classes, input_size, num_protected_groups = load_dataset(
        dataset_name="cifar100",
        augment=False,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        dataset_kwargs=dataset_kwargs,
    )

    target_accuracy = 72.63 / 100.0
    loaded_accuracy = compute_model_accuracy(cifarresnet_model, loaders["val"])
    assert np.abs(target_accuracy - loaded_accuracy) < (0.05 / 100.0)
