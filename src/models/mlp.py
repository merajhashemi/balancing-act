from typing import Callable, List, Type, Union

import numpy as np
from torch import Tensor, nn

import src.sparse as sparse

LinearLayer = Union[nn.Linear, sparse.MaskedLinear]


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_dims: List[int],
        num_classes: int,
        bias: bool = True,
        act_fn: Callable[..., nn.Module] = nn.ReLU,
        linear_layer: Type[LinearLayer] = nn.Linear,
        **kwargs,  # Added to skip unknown entries in config.model
    ):
        super(MLP, self).__init__()

        self.input_shape = input_shape
        self.input_dim = int(np.prod(self.input_shape))
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims

        self._act_fn = act_fn
        self._linear_layer = linear_layer

        self.main = self.construct_main(bias=bias)

    def construct_main(self, bias: bool) -> nn.ModuleList:
        modules = []

        in_dim = self.input_dim
        for h_dim in self.hidden_dims:
            layer = self._linear_layer(in_features=in_dim, out_features=h_dim, bias=bias)
            modules.append(layer)
            modules.append(self._act_fn())
            in_dim = h_dim

        last_layer = self._linear_layer(in_features=in_dim, out_features=self.num_classes, bias=bias)
        modules.append(last_layer)

        return nn.ModuleList(modules)

    def forward(self, input: Tensor) -> Tensor:
        x = input.view(-1, self.input_dim)

        for layer in self.main:
            x = layer(x)[0] if isinstance(layer, sparse.MaskedLinear) else layer(x)

        return x
