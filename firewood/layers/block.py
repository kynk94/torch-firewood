import copy
import inspect
import re
import warnings
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
from torch import Tensor

from firewood import utils
from firewood.common import backend
from firewood.common.types import INT, STR
from firewood.hooks import lr_equalizers, weight_normalizations
from firewood.layers import activations
from firewood.layers import noise as _noise
from firewood.layers import normalizations
from firewood.layers.bias import Bias
from firewood.layers.biased_activations import ACTIVATIONS, BiasedActivation
from firewood.layers.upfirdn import _UpFirDnNd, get_upfir_firdn_layers

# If want to use other layers in Block, modify values of SUPPORT_LAYER_NAMES.
# Set the name not to be confused with `nn.Module` basic attr name.
# (ex. "weight" -> "weighting", "bias" -> "add_bias")
SUPPORT_LAYER_NAMES = {
    "W": ["up_fir", "weighting", "fir_down", "noise"],
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
    __op_order = "WNA"

    def __init__(
        self,
        weight_layer: nn.Module,
        op_order: str = "WNA",
        normalization: Optional[str] = None,
        activation: Optional[str] = None,
        up: Optional[INT] = None,
        fir: Optional[List[float]] = None,
        down: Optional[INT] = None,
        noise: Optional[str] = None,
        weight_normalization: Optional[STR] = None,
        lr_equalization: Optional[bool] = None,
        dropout: Optional[float] = None,
        keep_meaningless_bias: bool = False,
        normalization_args: Optional[Dict[str, Any]] = None,
        activation_args: Optional[Dict[str, Any]] = None,
        fir_args: Optional[Dict[str, Any]] = None,
        noise_args: Optional[Dict[str, Any]] = None,
        weight_normalization_args: Optional[Dict[str, Any]] = None,
        lr_equalization_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.SUPPORT_LAYER_NAMES = copy.deepcopy(SUPPORT_LAYER_NAMES)
        self.rank = utils.get_rank(weight_layer)
        self.op_order = op_order
        self.keep_meaningless_bias = keep_meaningless_bias

        # Do not modify self.layers directly, use self.update_layer_in_order().
        self.layers = nn.ModuleDict()

        # set weight layer
        self.update_layer_in_order("weighting", weight_layer)

        # set FIR filter
        up_fir_layer, fir_down_layer = get_upfir_firdn_layers(
            rank=self.rank, kernel=fir, up=up, down=down, **fir_args or dict()
        )
        self.update_layer_in_order("up_fir", up_fir_layer)
        self.update_layer_in_order("fir_down", fir_down_layer)

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
        if lr_equalization:
            lr_equalizers.lr_equalizer(self, **self.lr_equalization_args)

    def forward(self, input: Tensor, *extra_inputs: Any) -> Tensor:
        output = input
        for layer in self.layers.values():
            if getattr(layer, "use_extra_inputs", False):
                output = layer(output, *extra_inputs)
            else:
                output = layer(output)
        return output

    @property
    def lr_equalization(self) -> bool:
        if "weighting" not in self.layers:
            return False
        weight_layer = self.layers.get_submodule("weighting")
        for hook in weight_layer._forward_pre_hooks.values():
            if isinstance(hook, lr_equalizers.WeightLREqualizer):
                return True
        return False

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
        if not self.keep_meaningless_bias:
            self.__remove_meaningless_bias()
        self.__fuse_layers_if_faster()
        self.__unravel_fused_layers_if_faster()
        self.__move_bias_to_optimal_position()
        self.__check_extra_inputs()

    @torch.no_grad()
    def _update_layer_in_order(
        self, name: str, module: Optional[nn.Module] = None
    ) -> None:
        if name not in sum(self.SUPPORT_LAYER_NAMES.values(), []):
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
            for _name in self.SUPPORT_LAYER_NAMES[op]:
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
        for name, param in weight_layer.named_parameters(recurse=False):
            if "bias" in name:
                break
        else:
            return
        bias_layer = Bias()
        lr_equalizers.transfer_bias_attrs(weight_layer, bias_layer)
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

        def __fuse_upsample_1x1_convolution() -> None:
            """
            Weighting first to not use upsampled spatial size.
            weighting -> up_fir
            """
            w_index = self.SUPPORT_LAYER_NAMES["W"].index("weighting")
            if self.SUPPORT_LAYER_NAMES["W"][w_index - 1] != "up_fir":
                return
            self.SUPPORT_LAYER_NAMES["W"][w_index - 1] = "weighting"
            self.SUPPORT_LAYER_NAMES["W"][w_index] = "up_fir"

        def __fuse_downsample_1x1_convolution() -> None:
            """
            Downsampling first to reduce the spatial size to compute.
            fir_down -> weighting
            """
            w_index = self.SUPPORT_LAYER_NAMES["W"].index("weighting")
            if self.SUPPORT_LAYER_NAMES["W"][w_index + 1] != "fir_down":
                return
            self.SUPPORT_LAYER_NAMES["W"][w_index + 1] = "weighting"
            self.SUPPORT_LAYER_NAMES["W"][w_index] = "fir_down"

        def __unravel_resample_1x1_convolution() -> None:
            """
            Unravel resample 1x1 convolution if use both upsample and downsample.
            Because it is not possible to fuse upsample and downsample.
            up_fir -> weighting -> fir_down
            """
            w_index = self.SUPPORT_LAYER_NAMES["W"].index("weighting")
            if self.SUPPORT_LAYER_NAMES["W"][w_index - 1] == "fir_down":
                self.SUPPORT_LAYER_NAMES["W"][w_index - 1] = "weighting"
                self.SUPPORT_LAYER_NAMES["W"][w_index] = "fir_down"
            if self.SUPPORT_LAYER_NAMES["W"][w_index + 1] == "up_fir":
                self.SUPPORT_LAYER_NAMES["W"][w_index + 1] = "weighting"
                self.SUPPORT_LAYER_NAMES["W"][w_index] = "up_fir"

        # TODO: Need to implement follows.
        # conv -> transposed_conv, transposed_conv -> conv
        def __fuse_upsample_convolution() -> None:
            weight_layer = self.layers.get_submodule("weighting")
            up_layer = cast(_UpFirDnNd, self.layers.get_submodule("up_fir"))

        def __fuse_downsample_convolution() -> None:
            weight_layer = self.layers.get_submodule("weighting")
            down_layer = cast(_UpFirDnNd, self.layers.get_submodule("fir_down"))

        def __fuse_resample_convolution() -> None:
            if "weighting" not in self.layers:
                return
            weight_layer = self.layers.get_submodule("weighting")
            name = utils.get_name(weight_layer).lower()
            if "conv" not in name or "sep" in name:
                return
            if "up_fir" in self.layers:
                up_layer = cast(_UpFirDnNd, self.layers.get_submodule("up_fir"))
                up_use_resample = up_layer.use_resample
                up_use_fir = up_layer.use_fir
            else:
                up_layer = None
                up_use_resample = False
                up_use_fir = False
            if "fir_down" in self.layers:
                down_layer = cast(
                    _UpFirDnNd, self.layers.get_submodule("fir_down")
                )
                down_use_resample = down_layer.use_resample
                down_use_fir = down_layer.use_fir
            else:
                down_layer = None
                down_use_resample = False
                down_use_fir = False
            # If fir only, do not fuse.
            if not up_use_resample and not down_use_resample:
                return

            is_1x1_convolution = all(
                s == 1
                for s in cast(Tensor, getattr(weight_layer, "weight")).shape[2:]
            )
            if is_1x1_convolution:
                if up_use_resample and down_layer is None:
                    __fuse_upsample_1x1_convolution()
                elif up_layer is None and down_use_resample:
                    __fuse_downsample_1x1_convolution()
                else:
                    __unravel_resample_1x1_convolution()
                return

            if set(getattr(weight_layer, "stride")) != {1}:
                return

            if up_layer is None and down_use_resample:
                __fuse_downsample_convolution()
            elif up_use_resample:
                __fuse_upsample_convolution()

        def __fuse_biased_activation() -> None:
            if "activation" not in self.layers:
                return
            if "add_bias" not in self.layers:
                return
            if "BA" not in self.__op_order:
                return
            self.__update_activation(fuse=True)
            activation_layer = self.layers.get_submodule("activation")
            if isinstance(activation_layer, BiasedActivation):
                bias_layer = self.layers.get_submodule("add_bias")
                lr_equalizers.transfer_bias_attrs(bias_layer, activation_layer)
                self._update_layer_in_order("add_bias", None)

        __fuse_resample_convolution()
        __fuse_biased_activation()

    def __unravel_fused_layers_if_faster(self) -> None:
        """
        Unravel fused layers if the operation is faster than the fused.
        """

        def __unravel_biased_activation() -> None:
            if "activation" not in self.layers:
                return
            activation_layer = self.layers.get_submodule("activation")
            gain = self.activation_args.get("gain", None)
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
        if "add_bias" not in self.layers:
            return
        op_before_bias = self.__op_order[
            self.__op_order.index("W") : self.__op_order.index("B")
        ]
        layers: List[str] = sum(
            (self.SUPPORT_LAYER_NAMES[op] for op in op_before_bias), []
        )
        index = layers.index("weighting")
        for layer in layers[index + 1 :]:
            if layer in self.layers:
                return
        weight_layer = self.layers.get_submodule("weighting")
        bias_layer = self.layers.get_submodule("add_bias")
        lr_equalizers.transfer_bias_attrs(bias_layer, weight_layer)
        self._update_layer_in_order("add_bias", None)

    def __check_extra_inputs(self) -> None:
        count_use_extra_inputs = 0
        for layer in self.layers.values():
            if getattr(layer, "use_extra_inputs", False):
                count_use_extra_inputs += 1
        if count_use_extra_inputs > 1:
            raise ValueError(
                "Only one layer can use extra inputs at the same time in Block."
            )

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


def conv_to_trans_conv(module: nn.Module) -> nn.Module:
    """
    Convert a convolutional layer to a transposed convolutional layer.
    If the input is already transposed convolutional layer, return input itself.
    The converted module's output will be the same as the original module's if
    stride is 1 and padding is 0.

    Args:
        module: A convolutional layer.
            nn.ConvNd -> nn.ConvTransposeNd
            firewood.layers.GFixConvNd -> firewood.layers.GFixConvTransposeNd
    Returns:
        A transposed convolutional layer.
    """
    for submodule in module.modules():
        name = utils.get_name(submodule).lower()
        if "conv" not in name or "sep" in name:
            continue
    params = dict(inspect.signature(module.__init__).parameters)


def trans_conv_to_conv(module: nn.Module) -> nn.Module:
    """
    Convert a transposed convolutional layer to a convolutional layer.
    If the input is already convolutional layer, return input itself.
    The converted module's output will be the same as the original module's if
    stride is 1 and padding is 0 and output_padding is 0.

    Args:
        module: A transposed convolutional layer.
            nn.ConvTransposeNd -> nn.ConvNd
            firewood.layers.GFixConvTransposeNd -> firewood.layers.GFixConvNd
    Returns:
        A convolutional layer.
    """
    params = dict(inspect.signature(module.__init__).parameters)
