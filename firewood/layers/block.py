import re
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from firewood import utils
from firewood.common import backend
from firewood.common.types import STR
from firewood.hooks import lr_equalizers, weight_normalizations
from firewood.layers import activations
from firewood.layers import noise as _noise
from firewood.layers import normalizations
from firewood.layers.bias import Bias
from firewood.layers.biased_activations import ACTIVATIONS, BiasedActivation
from firewood.layers.upfirdn import get_upfirdn_layer

# If want to use other layers in Block, modify values of SUPPORT_LAYER_NAMES.
SUPPORT_LAYER_NAMES = {
    "W": ["up_fir", "weighting", "down_fir", "noise"],
    "N": ["normalization"],
    "B": ["add_bias"],
    "A": ["activation", "dropout"],
}


class Block(nn.Module):
    """
    Wrapper for the weight layer.

    Basic order of layers:
        (weight -> fir -> noise) -> normalization -> (bias -> activation)
        Each layers is optional, except weight layer.
        And if affine of normalization's attribute is True, bias layer will be
        deleted because it is included in normalization layer.
    """

    # private properties
    __lr_equalization = False
    __op_order = "WNA"

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
        self.rank = utils.get_rank(weight_layer)
        self.op_order = op_order

        # Do not modify self.layers directly, use self.update_layer_in_order().
        self.layers = nn.ModuleDict()

        # set weight layer
        self.update_layer_in_order("weighting", weight_layer)

        # set FIR filter
        up_fir_layer, down_fir_layer = get_upfirdn_layer(
            rank=self.rank, kernel=fir, **fir_args or dict()
        )
        self.update_layer_in_order("up_fir", up_fir_layer)
        self.update_layer_in_order("down_fir", down_fir_layer)

        # set noise
        self.update_layer_in_order(
            "noise", _noise.get(noise, **noise_args or dict())
        )

        # set normalization
        normalization_args = normalization_args or dict()
        in_features, out_features = utils.get_in_out_features(weight_layer)
        if self.op_order.index("N") < self.op_order.index("W"):
            num_features = in_features
        else:
            num_features = out_features
        self.update_layer_in_order(
            "normalization",
            normalizations.get(
                normalization=normalization,
                num_features=num_features,
                **normalization_args,
            ),
        )

        # set activation
        self.activation_args = activation_args or dict()
        self.activation_name = activations.normalize_activation_name(activation)
        self.update_layer_in_order(
            "activation",
            activations.get(self.activation_name, **self.activation_args),
        )

        # set dropout
        if isinstance(dropout, float) and 0 < dropout < 1:
            dropout_class = getattr(nn, f"Dropout{self.rank}d", nn.Dropout)
            self.update_layer_in_order("dropout", dropout_class(dropout))

        weight_normalization_args = weight_normalization_args or dict()
        self.weight_normalization = weight_normalization or ()
        if isinstance(self.weight_normalization, str):
            self.weight_normalization = (self.weight_normalization,)
        for _norm in self.weight_normalization:
            norm = weight_normalizations.get(_norm, **weight_normalization_args)
            if norm is not None:
                norm(self.layers.get_submodule("weighting"))

        if lr_equalization is None:
            lr_equalization = backend.lr_equalization()
        self.lr_equalization_args = lr_equalization_args or dict()
        self.lr_equalization = lr_equalization

    def forward(
        self, input: Tensor, external_input: Optional[Tensor] = None
    ) -> Tensor:
        output = input
        for name, layer in self.layers.items():
            if getattr(layer, "use_external_input", False):
                if getattr(layer, "use_external_output", False):
                    output, external_input = layer(output, external_input)
                else:
                    output = layer(output, external_input)
            else:
                output = layer(output)
        return output

    @property
    def lr_equalization(self) -> bool:
        return self.__lr_equalization

    @lr_equalization.setter
    def lr_equalization(self, value: bool) -> None:
        if self.__lr_equalization == value:
            return
        self.__lr_equalization = value
        self._check_layers()

    @property
    def op_order(self) -> str:
        return self.__op_order.replace("B", "")

    @op_order.setter
    def op_order(self, op_order: str) -> None:
        self.__op_order = normalize_op_order(op_order)

    @torch.no_grad()
    def update_layer_in_order(
        self, name: str, module: Optional[nn.Module] = None
    ) -> None:
        """
        Update layers of block with the given name in the correct order.
        If module is None, the layer will be removed from the layers.

        Used only for initialization, do not use this method for trained models.
        If want to use to trained models, use `_update_layer_in_order` method.
        """
        self._update_layer_in_order(name, module)
        self._check_layers()

    @torch.no_grad()
    def _check_layers(self) -> None:
        # extract bias -> fuse -> if not fused, move bias to optimal position.
        self.__move_bias_to_independent_layer()
        self.__remove_meaningless_bias()
        self.__fuse_layers_if_faster()
        self.__unravel_fused_layers_if_faster()
        self.__move_bias_to_optimal_position()

    @torch.no_grad()
    def _update_layer_in_order(
        self, name: str, module: Optional[nn.Module] = None
    ) -> None:
        if name not in sum(SUPPORT_LAYER_NAMES.values(), []):
            raise ValueError(
                f"Not support: {name}. If want to use this layer, add it to "
                "`firewood.layers.block.SUPPORT_LAYER_NAMES` in correct order."
            )

        if module is None:
            if name in self.layers:
                del self.layers[name]
            return

        updated_layers = nn.ModuleDict()
        for op in self.__op_order:
            for _name in SUPPORT_LAYER_NAMES[op]:
                if _name == name:
                    updated_layers.add_module(_name, module)
                elif _name in self.layers:
                    updated_layers.add_module(
                        _name, self.layers.get_submodule(_name)
                    )
        self.layers = updated_layers

    def __move_bias_to_independent_layer(self) -> None:
        if "weighting" not in self.layers:
            return
        weight_layer = self.layers.get_submodule("weighting")
        bias_attrs = lr_equalizers.pop_bias_attrs(weight_layer)
        if bias_attrs["bias"] is None:
            return
        bias_layer = Bias()
        lr_equalizers.set_bias_attrs(bias_layer, **bias_attrs)
        self._update_layer_in_order("add_bias", bias_layer)

    def __remove_meaningless_bias(self) -> None:
        if "add_bias" not in self.layers:
            return

        if "activation" in self.layers:
            activation_layer = self.layers.get_submodule("activation")
            if getattr(activation_layer, "bias", None) is not None:
                lr_equalizers.pop_bias_attrs(activation_layer)
                warnings.warn("Remove meaningless bias from activation layer.")

        # If affine is True, the normalization layer multiply weight and
        # add bias. So, no need to use bias after normalization.
        if "normalization" in self.layers and self.op_order == "WNA":
            normalization_layer = self.layers.get_submodule("normalization")
            if getattr(normalization_layer, "affine", False):
                self._update_layer_in_order("add_bias", None)
                warnings.warn("Remove meaningless bias after normalization.")

    def __update_activation(self, fuse: bool) -> None:
        if "activation" not in self.layers:
            return
        activation_layer = self.layers.get_submodule("activation")
        if fuse == isinstance(activation_layer, BiasedActivation):
            return
        activation_args = self.activation_args.copy()
        # By default, the gain value is used when initializing the weights.
        # In runtime weight scaling, the gain value is multiplied to the outputs
        # of the activation function and not used in the initialization sequence.
        # Default value of the gain is 1.0 because it is already multiplied in
        # the initialization sequence.
        if activation_args.get("gain", None) is None:
            if self.lr_equalization:
                # If default_gain is used, only `relu`, `leaky_relu` and
                # `silu` become BiasedActivation.
                gain = ACTIVATIONS[self.activation_name]["default_gain"]
            else:
                gain = 1.0
            activation_args.update(gain=gain)
        activation_layer = activations.get(
            activation=self.activation_name, **activation_args
        )
        self._update_layer_in_order("activation", activation_layer)

    def __fuse_layers_if_faster(self) -> None:
        """
        Fuse layers if the operation is faster than the original.
        Use only trainable fused layers.
        """

        def __fuse_biased_activation() -> None:
            if "activation" not in self.layers:
                return
            if "add_bias" not in self.layers:
                return
            if "BA" not in self.__op_order:
                return
            self.__update_activation(fuse=True)
            activation_layer = self.layers.get_submodule("activation")
            bias_layer = self.layers.get_submodule("add_bias")
            if isinstance(activation_layer, BiasedActivation):
                lr_equalizers.transfer_bias_attrs(bias_layer, activation_layer)
                self._update_layer_in_order("add_bias", None)

        __fuse_biased_activation()

    def __unravel_fused_layers_if_faster(self) -> None:
        """
        Unravel fused layers if the operation is faster than the fused.
        """

        def __unravel_biased_activation() -> None:
            if "activation" not in self.layers:
                return
            activation_layer = self.layers.get_submodule("activation")
            gain = getattr(self.activation_args, "gain", None)
            if self.lr_equalization and gain is None:
                return
            if (
                isinstance(activation_layer, BiasedActivation)
                and getattr(activation_layer, "bias", None) is not None
                and gain in {None, 1.0}
            ):
                bias_layer = Bias()
                lr_equalizers.transfer_bias_attrs(activation_layer, bias_layer)
                self._update_layer_in_order("add_bias", bias_layer)
            self.__update_activation(fuse=False)

        __unravel_biased_activation()

    def __move_bias_to_optimal_position(self) -> None:
        """
        Determine whether bias_layer is required.
        """
        if "add_bias" not in self.layers:
            return
        op_before_bias = self.__op_order[
            self.__op_order.index("W") : self.__op_order.index("B")
        ]
        layers: List[str] = sum(
            (SUPPORT_LAYER_NAMES[op] for op in op_before_bias), []
        )
        index = layers.index("weighting")
        for layer in layers[index + 1 :]:
            if layer in self.layers:
                return
        weight_layer = self.layers.get_submodule("weighting")
        bias_layer = self.layers.get_submodule("add_bias")
        lr_equalizers.transfer_bias_attrs(bias_layer, weight_layer)
        self._update_layer_in_order("add_bias", None)

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"op_order={self.op_order}",
                f"lr_equalization={self.lr_equalization}",
            ]
        )


def normalize_op_order(op_order: str) -> str:
    """
    A: activation layer
    N: normalization layer
    W: weight layer
    """
    op_order = op_order.upper()
    if len(op_order) != 3:
        raise ValueError("Op order must be 3 characters long.")
    op_order = re.sub("[^ANW]", "W", op_order)
    for char in "ANW":
        if op_order.count(char) != 1:
            raise ValueError(
                f"Op order must contain exactly one {char} character."
            )
    return re.sub("(WN?)", r"\1B", op_order)
