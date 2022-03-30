import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from firewood import utils
from firewood.common import backend
from firewood.common.types import STR
from firewood.layers import activations, lr_equalizers
from firewood.layers import noise as _noise
from firewood.layers import normalizations, weight_normalizations
from firewood.layers.biased_activations import BiasedActivation
from firewood.layers.upfirdn import get_upfirdn_layer


class Block(nn.Module):
    """
    Wrapper for the weight layer.

    Basic order of layers:
        (weight -> fir -> noise) -> normalization -> (bias -> activation)
        Each layers is optional, except weight layer.
    """

    def __init__(
        self,
        weight_layer: nn.Module,
        op_order: str = "WNA",
        normalization: Optional[str] = None,
        normalization_args: Optional[Dict[str, Any]] = None,
        activation: Optional[str] = None,
        activation_args: Optional[Dict[str, Any]] = None,
        fir: Optional[List[float]] = None,
        fir_args: Optional[Dict[str, Any]] = None,
        noise: Optional[str] = None,
        noise_args: Optional[Dict[str, Any]] = None,
        weight_normalization: Optional[STR] = None,
        weight_normalization_args: Optional[Dict[str, Any]] = None,
        lr_equalization: Optional[bool] = None,
        lr_equalization_args: Optional[Dict[str, Any]] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        in_features, out_features = utils.get_in_out_features(weight_layer)
        self.op_order = utils.normalize_op_order(op_order)
        if lr_equalization is None:
            lr_equalization = backend.lr_equalization()
        self._lr_equalization = lr_equalization

        self.weight_layer = weight_layer
        self.rank = getattr(weight_layer, "rank", None)
        if self.rank is None:
            self.rank = (
                getattr(getattr(weight_layer, "weight", None), "ndim", 3) - 2
            )
        # set FIR filter
        self.up_fir, self.down_fir = get_upfirdn_layer(
            rank=self.rank, kernel=fir, **fir_args or dict()
        )

        # set noise
        self.noise = _noise.get(noise, **noise_args or dict())

        # set normalization
        normalization_args = normalization_args or dict()
        if self.op_order.index("N") < self.op_order.index("W"):
            num_features = in_features
        else:
            num_features = out_features
        self.normalization = normalizations.get(
            normalization=normalization,
            num_features=num_features,
            **normalization_args,
        )

        # set activation
        self.activation_args = activation_args or dict()
        if not self.lr_equalization:
            # By default, the gain value is used when initializing the weights.
            # In runtime weight scaling, the gain value is multiplied to the
            # outputs of the activation function and not used in the
            # initialization sequence.
            # Default value of the gain is 1.0 because it is already multiplied
            # in the initialization sequence.
            self.activation_args.update(gain=1.0)
        self.activation_name = activation
        self.activation = activations.get(activation, **self.activation_args)

        self.bias: Optional[Tensor] = None
        self._transfer_bias_if_need()

        # set dropout
        if dropout is None or dropout <= 0.0 or dropout >= 1.0:
            self.dropout: Optional[nn.Module] = None
        else:
            weight: Tensor = getattr(self.weight_layer, "weight")
            rank = weight.ndim - 2
            if rank == 2:
                self.dropout = nn.Dropout2d(dropout)
            elif rank == 3:
                self.dropout = nn.Dropout3d(dropout)
            else:
                self.dropout = nn.Dropout(dropout)

        weight_normalization_args = weight_normalization_args or dict()
        self.weight_normalization = weight_normalization or ()
        if isinstance(self.weight_normalization, str):
            self.weight_normalization = (self.weight_normalization,)
        for _norm in self.weight_normalization:
            norm = weight_normalizations.get(_norm, **weight_normalization_args)
            if norm is not None:
                self.weight_layer = norm(self.weight_layer)

        # Not use nn.ModuleDict because all layers are submodules already.
        self.layers: Dict[str, List[Any]] = defaultdict(list)
        if self.up_fir is not None:
            self.layers["W"].append(self.up_fir)
        self.layers["W"].append(self.weight_layer)
        if self.down_fir is not None:
            self.layers["W"].append(self.down_fir)
        if self.noise is not None:
            self.layers["W"].append(self.noise)
        if self.normalization is not None:
            self.layers["N"].append(self.normalization)
        self._reset_activation_layers()
        if self.lr_equalization:
            lr_equalization_args = lr_equalization_args or dict()
            lr_equalization_args.update(recursive=True)
            lr_equalizers.lr_equalizer(self, **lr_equalization_args)

    @property
    def lr_equalization(self) -> bool:
        return self._lr_equalization

    @lr_equalization.setter
    def lr_equalization(self, value: bool) -> None:
        self._lr_equalization = value
        if self.activation is None:
            return
        if value and self.activation_args.get("gain") == 1.0:
            self.activation_args.update(gain=None)
        self.activation = activations.get(
            self.activation_name, **self.activation_args
        )
        self._transfer_bias_if_need()
        self._reset_activation_layers()

    def _transfer_bias_if_need(self) -> None:
        """
        If necessary, extract bias from weight layer for other operations.
        (e.g. FIR, noise, normalization, biased_activation, ...)
        """
        if getattr(self.weight_layer, "bias", None) is not None and (
            self.down_fir is not None
            or self.noise is not None
            or (
                self.normalization is not None
                and self.op_order.index("W") < self.op_order.index("N")
            )
            or (
                isinstance(self.activation, BiasedActivation)
                and self.op_order.index("W") < self.op_order.index("A")
            )
        ):
            lr_equalizers.transfer_bias_attrs(self.weight_layer, self)

    def _reset_activation_layers(self) -> None:
        self.layers["A"].clear()
        use_biased_activation = isinstance(self.activation, BiasedActivation)
        if self.bias is not None and not use_biased_activation:
            self.layers["A"].append(self._add_bias)
        if self.activation is None:
            return
        if use_biased_activation:
            self.layers["A"].append(
                functools.partial(self.activation, bias=self._get_bias)
            )
        else:
            self.layers["A"].append(self.activation)
        if getattr(self, "dropout", None) is not None:
            self.layers["A"].append(self.dropout)

    def _get_bias(self) -> Optional[Tensor]:
        return self.bias

    def _add_bias(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return input
        bias = self.bias.view([-1 if i == 1 else 1 for i in range(input.ndim)])
        return input + bias.to(dtype=input.dtype)

    def forward(
        self, input: Tensor, external_input: Optional[Tensor] = None
    ) -> Tensor:
        output = input
        for op in self.op_order:
            for layer in self.layers[op]:
                if getattr(layer, "use_external_input", False):
                    output = layer(output, external_input)
                else:
                    output = layer(output)
        return output

    def extra_repr(self) -> str:
        bias = (
            False if self.bias is None else "True (extracted from weight_layer)"
        )
        return ", ".join(
            [
                f"op_order={self.op_order}",
                f"bias={bias}",
                f"lr_equalization={self.lr_equalization}",
            ]
        )
