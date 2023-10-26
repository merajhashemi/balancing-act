"""
Code re-used from https://github.com/chenyaofo/pytorch-cifar-models/blob/master/pytorch_cifar_models/resnet.py
"""
from typing import Any, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

import src.sparse as sparse
from src.models.modeling_utils import apply_bn_to_conv_out, choose_correct_bn_type, create_general_conv2d

BNLayer = Union[nn.BatchNorm2d, sparse.MaskedBatchNorm2d]
Conv2dLayer = Union[
    nn.Conv2d,
    sparse.MaskedConv2d,
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Type[BNLayer] = nn.BatchNorm2d,
        conv_layer: Type[Conv2dLayer] = sparse.MaskedConv2d,
        masked_layer_kwargs: dict = {},  # include temp, detach and weight_dec
        masked_conv_ix: List[str] = [],
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = create_general_conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=stride,
            groups=groups,  # Default 1
            dilation=dilation,  # Default  1
            padding=dilation,  # Default  1
            conv_layer=conv_layer if "conv1" in masked_conv_ix else nn.Conv2d,
            use_bias=False,
            masked_layer_kwargs=masked_layer_kwargs,
        )
        bn1_type = choose_correct_bn_type(self.conv1, norm_layer)
        self.bn1 = bn1_type(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = create_general_conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=(3, 3),
            stride=1,
            groups=groups,  # Default 1
            dilation=dilation,  # Default  1
            padding=dilation,  # Default  1
            conv_layer=conv_layer if "conv1" in masked_conv_ix else nn.Conv2d,
            use_bias=False,
            masked_layer_kwargs=masked_layer_kwargs,
        )
        bn2_type = choose_correct_bn_type(self.conv2, norm_layer)
        self.bn2 = bn2_type(num_features=planes)
        # self.downsample = downsample
        self.stride = stride

        self.has_downsample = False
        if stride != 1 or inplanes != planes * self.expansion:
            self.has_downsample = True

            self.downsample_conv = create_general_conv2d(
                in_channels=inplanes,
                out_channels=planes * self.expansion,
                kernel_size=(1, 1),
                stride=stride,
                groups=1,
                conv_layer=conv_layer if "downsample_conv" in masked_conv_ix else nn.Conv2d,
                use_bias=False,
                masked_layer_kwargs=masked_layer_kwargs,
            )

            down_bn_type = choose_correct_bn_type(self.downsample_conv, norm_layer)
            self.downsample_bn = down_bn_type(num_features=planes * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = apply_bn_to_conv_out(self.bn1, out)
        out = self.relu(out)

        out = self.conv2(out)
        out = apply_bn_to_conv_out(self.bn2, out)

        if self.has_downsample:
            identity = apply_bn_to_conv_out(self.downsample_bn, self.downsample_conv(x))

        # breakpoint()
        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 100,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Type[BNLayer] = nn.BatchNorm2d,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        conv_layer: Type[Conv2dLayer] = nn.Conv2d,
        masked_conv_ix: List[str] = ["conv1", "conv2", "conv3", "downsample_conv"],
        masked_layer_kwargs: dict[str, Any] = {},
        is_first_conv_dense: bool = True,
        is_last_fc_dense: bool = True,
    ):
        super(CifarResNet, self).__init__()

        assert input_shape[0] == 3
        self.input_shape = input_shape

        self._norm_layer = norm_layer
        self._conv_layer = conv_layer
        self.inplanes = 16

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = create_general_conv2d(
            conv_layer=nn.Conv2d if is_first_conv_dense else conv_layer,
            in_channels=self.input_shape[0],  # 3 input channels
            out_channels=self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            use_bias=False,
            masked_layer_kwargs={},
        )
        bn1_class = choose_correct_bn_type(self.conv1, norm_layer)
        self.bn1 = bn1_class(num_features=self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        block_kwargs = {
            "norm_layer": self._norm_layer,
            "conv_layer": self._conv_layer,
            "masked_layer_kwargs": masked_layer_kwargs,
            "masked_conv_ix": masked_conv_ix,
        }

        self.layer1 = self._make_layer(
            block,
            planes=16,
            blocks=layers[0],
            stride=1,
            dilate=False,
            block_kwargs=block_kwargs,
        )
        self.layer2 = self._make_layer(
            block,
            planes=32,
            blocks=layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            block_kwargs=block_kwargs,
        )
        self.layer3 = self._make_layer(
            block,
            planes=64,
            blocks=layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            block_kwargs=block_kwargs,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if is_last_fc_dense or conv_layer != sparse.MaskedConv2d:
            # If the model is dense, we enforce the last layer to be dense
            last_fc_layer = nn.Linear
        else:
            last_fc_layer = sparse.MaskedLinear

        self.fc = last_fc_layer(in_features=64 * block.expansion, out_features=num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,  # out_planes
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        block_kwargs: dict[str, Any] = {},
    ) -> nn.Sequential:

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(
            block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                **block_kwargs,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    **block_kwargs,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = apply_bn_to_conv_out(self.bn1, x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if isinstance(self.fc, sparse.MaskedLinear):
            x, _ = self.fc(x)
        else:
            x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


def cifar100_resnet56(num_classes: int = 100, **kwargs: Any) -> CifarResNet:
    return CifarResNet(block=BasicBlock, layers=[9, 9, 9], num_classes=num_classes, **kwargs)
