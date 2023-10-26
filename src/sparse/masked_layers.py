from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.sparse.layer_stats import LayerStats

LayerMasks = tuple[Tensor, Optional[Tensor]]
MaskedForwardOutput = tuple[Tensor, Optional[Tensor]]


class MaskedLayer(nn.Module):
    def __init__(self, weight_mask: Tensor = None, bias_mask: Tensor = None, *args, **kwargs):
        super(MaskedLayer, self).__init__(*args, **kwargs)

        self._do_masked_forward = False

        # When initializing with None, the mask buffer would be ignored when
        # creating the state dict for this module. Methods `get_extra_state` and
        # `set_extra_state` are used to handle this correctly.
        self.register_buffer("weight_mask", weight_mask, persistent=False)
        self.register_buffer("bias_mask", bias_mask, persistent=False)

    @property
    def do_masked_forward(self):
        return self._do_masked_forward

    @do_masked_forward.setter
    def do_masked_forward(self, value: bool):
        self._do_masked_forward = value

    @torch.no_grad()
    def update_weight_mask_(self, weight_mask: Tensor):
        self.weight_mask = weight_mask

    @torch.no_grad()
    def update_bias_mask_(self, bias_mask: Optional[Tensor]):
        self.bias_mask = bias_mask

    def get_parameters(self, masked: bool = True) -> tuple[Tensor, Optional[Tensor]]:
        if self.weight_mask is None or not masked:
            return self.weight, self.bias
        else:
            weight_mask = torch.ones_like(self.weight, dtype=torch.bool) * self.weight_mask
            masked_weight = torch.where(weight_mask, self.weight, torch.zeros_like(self.weight))

            if self.bias is not None and self.bias_mask is not None:
                bias_mask = torch.ones_like(self.bias, dtype=torch.bool) * self.bias_mask
                masked_bias = torch.where(bias_mask, self.bias, torch.zeros_like(self.bias))

            else:
                # Either bias or bias_mask is None
                masked_bias = self.bias

            return masked_weight, masked_bias

    def layer_stats(self) -> LayerStats:
        num_params = self.weight.numel()
        num_params += 0 if self.bias is None else self.bias.numel()

        with torch.no_grad():
            # No need to add the computation of sparsity statistics of
            # *non-sparsifiable* parameters to the computational graph.

            # Looking directly at the weight and bias tensors allows us to avoid
            # _inferring_ the number of active parameters based on the mask.
            _weight, _bias = self.get_parameters(masked=True)

        # _weight and _bias are masked versions of self.weight and self.bias of
        # the same shape. We count the number of non-zero elements in each.
        # This way of counting relies on non-zero initialization of the
        # parameters, as well as the almost-surely impossibility of the
        # parameters becoming _exactly_ zero.
        num_active_params = torch.count_nonzero(_weight)
        num_active_params += 0 if _bias is None else torch.count_nonzero(_bias)

        # Biases are kept fully dense, so they do not count as sparsifiable
        num_sparse_params = _weight.numel()
        # See above for explanation of why we use this method of counting.
        num_active_sparse_params = torch.count_nonzero(_weight)

        sq_l2_norm = torch.linalg.norm(_weight) ** 2
        sq_l2_norm += 0 if _bias is None else torch.linalg.norm(_bias) ** 2

        return LayerStats(
            layer_type=self.__class__.__name__,
            num_params=num_params,
            num_active_params=num_active_params,
            num_sparse_params=num_sparse_params,
            num_active_sparse_params=num_active_sparse_params,
            sq_l2_norm=sq_l2_norm,
        )

    def get_io_mask(self) -> Optional[Tensor]:
        if self.weight_mask is None:
            # There is no mask
            io_mask = None
        elif self.weight_mask.shape == self.weight.shape:
            # The mask is unstructured.
            if isinstance(self, nn.Conv2d):
                # Only if all the entries associated with an output channel are
                # zero, is the output channel is masked.
                io_mask = torch.sum(self.weight_mask, dim=(1, 2, 3))
            else:
                io_mask = torch.sum(self.weight_mask, dim=1)
        else:
            # This would have size in_features or out_channels, depending on
            # the layer type.
            io_mask = torch.squeeze(self.weight_mask)

        # Ensure that io_mask is boolean
        io_mask = io_mask > 0 if io_mask is not None else None

        return io_mask

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def get_extra_state(self):
        return {"weight_mask": self.weight_mask, "bias_mask": self.bias_mask}

    def set_extra_state(self, state):
        self.weight_mask = state["weight_mask"]
        self.bias_mask = state["bias_mask"]


class MaskedLinear(MaskedLayer, nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> MaskedForwardOutput:
        weight, bias = self.get_parameters(masked=self.do_masked_forward)
        out = F.linear(x, weight, bias)
        mask = self.get_io_mask() if self.do_masked_forward else None
        return out, mask


class MaskedConv2d(MaskedLayer, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> MaskedForwardOutput:
        weight, bias = self.get_parameters(masked=self.do_masked_forward)
        out = self._conv_forward(x, weight, bias)
        mask = self.get_io_mask() if self.do_masked_forward else None
        return out, mask
