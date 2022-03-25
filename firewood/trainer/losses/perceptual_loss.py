import functools
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as TM
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torchvision import transforms

from firewood.common.types import NEST_FLOAT, NEST_INT, NEST_STR

VGG = Literal["vgg", "vgg16", "vgg19"]

VGG16_TABLE: Dict[str, int] = {
    "block1_conv1": 1,
    "block1_conv2": 3,
    "block2_conv1": 6,
    "block2_conv2": 8,
    "block3_conv1": 11,
    "block3_conv2": 13,
    "block3_conv3": 15,
    "block4_conv1": 18,
    "block4_conv2": 20,
    "block4_conv3": 22,
    "block5_conv1": 25,
    "block5_conv2": 27,
    "block5_conv3": 29,
}
VGG19_TABLE: Dict[str, int] = {
    "block1_conv1": 1,
    "block1_conv2": 3,
    "block2_conv1": 6,
    "block2_conv2": 8,
    "block3_conv1": 11,
    "block3_conv2": 13,
    "block3_conv3": 15,
    "block3_conv4": 17,
    "block4_conv1": 20,
    "block4_conv2": 22,
    "block4_conv3": 24,
    "block4_conv4": 26,
    "block5_conv1": 29,
    "block5_conv2": 31,
    "block5_conv3": 33,
    "block5_conv4": 35,
}


def parse_targets(
    targets: Union[NEST_STR, NEST_INT], table: Dict[str, int] = VGG16_TABLE
) -> NEST_INT:
    if isinstance(targets, str):
        return table[targets]
    elif isinstance(targets, int):
        return targets
    parsed_targets = tuple(parse_targets(target, table) for target in targets)
    if all(isinstance(target, int) for target in parsed_targets):
        return cast(NEST_INT, tuple(sorted(parsed_targets)))
    parsed_targets = tuple(
        tuple(sorted(target)) if isinstance(target, tuple) else target
        for target in parsed_targets
    )
    return cast(NEST_INT, parsed_targets)


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        extractor: VGG = "vgg16",
        targets: Optional[Union[NEST_STR, NEST_INT]] = None,
    ):
        super().__init__()
        self.extractor = extractor.lower()
        if self.extractor in {"vgg", "vgg16"}:
            model = TM.vgg16(pretrained=True).features
            table = VGG16_TABLE
            if targets is None:
                targets = table["block5_conv3"]
        elif self.extractor in {"vgg19"}:
            model = TM.vgg19(pretrained=True).features
            table = VGG19_TABLE
            if targets is None:
                targets = table["block5_conv4"]
        else:
            raise ValueError(f"Not support extractor: {extractor}")
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        targets = parse_targets(targets, table)
        self.targets = cast(
            Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
            (targets,) if isinstance(targets, int) else targets,
        )
        self.is_single = all(isinstance(i, int) for i in self.targets)

        self.transform = transforms.Compose(
            [
                transforms.Resize(224, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, input: Tensor) -> Tensor:
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
        return self.transform(input * 0.5 + 0.5)

    def get_index(self, target: int) -> Union[int, Tuple[int, ...]]:
        if self.is_single:
            if target in self.targets:
                return 1
            return -1
        found_targets = []
        for i, target_sequence in enumerate(self.targets):
            if target in cast(Tuple[int, ...], target_sequence):
                found_targets.append(i)
        if found_targets:
            return tuple(found_targets)
        return -1

    def forward(
        self,
        input: Tensor,
    ) -> Union[List[Tensor], List[List[Tensor]]]:
        """
        input: (batch_size, 1 or 3, height, width) in range [-1, 1]
        """
        output = self.preprocess(input)
        outputs: List[Any] = []
        if not self.is_single:
            for _ in range(len(self.targets)):
                outputs.append([])
        for i, layer in enumerate(self.model):
            output = layer(output)
            index = self.get_index(i)
            if index == -1:
                continue
            if self.is_single:
                outputs.append(output)
                continue
            for target in cast(Tuple[int, ...], index):
                outputs[target].append(output)
        return outputs


class PerceptualLoss(_Loss):
    """
    Perceptual loss using VGG16 and VGG19.

    Args:
        weights: loss weights for each target.
            If `weights` specified, `targets` should be sorted in ascending order.
            Else, get wrong results.
    """

    def __init__(
        self,
        extractor: VGG = "vgg16",
        targets: Optional[Union[NEST_STR, NEST_INT]] = None,
        weights: Optional[NEST_FLOAT] = None,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.model = VGGFeatureExtractor(extractor, targets)

        if weights is not None:
            if isinstance(weights, (int, float)):
                if isinstance(self.model.targets[0], (tuple, list)):
                    weights = tuple(
                        (weights,) * len(target)
                        if isinstance(target, (tuple, list))
                        else (weights,)
                        for target in self.model.targets
                    )
                else:
                    weights = (weights,) * len(self.model.targets)
            elif isinstance(weights[0], (tuple, list)):
                weights = tuple(
                    tuple(weight)
                    if isinstance(weight, (tuple, list))
                    else (weight,)
                    for weight in weights
                )
                for i, w in enumerate(cast(Tuple[Tuple[float]], weights)):
                    target = self.model.targets[i]
                    if not isinstance(target, (tuple, list)):
                        continue
                    if len(w) != len(target):
                        raise ValueError(
                            "The length of `weights` must be equal to the number of targets"
                        )
            elif len(weights) != len(self.model.targets):
                raise ValueError(
                    "The length of `weights` must be equal to the number of targets"
                )
        self.weights = weights

        self.loss_fn = functools.partial(
            F.l1_loss,
            reduction=self.reduction,
        )

    def calculate_loss(
        self,
        features_a: Union[Tensor, List[Tensor]],
        features_b: Union[Tensor, List[Tensor]],
        weights: Optional[NEST_FLOAT] = None,
    ) -> Union[List[Tensor], List[List[Tensor]]]:
        losses = []
        if weights is None:
            for a, b in zip(features_a, features_b):
                if isinstance(a, Tensor):
                    losses.append(self.loss_fn(a, b))
                else:
                    losses.append(self.calculate_loss(a, b))  # type: ignore
        else:
            for a, b, w in zip(features_a, features_b, weights):  # type: ignore
                if isinstance(a, Tensor):
                    loss = self.loss_fn(a, b)
                    if w != 1.0:
                        loss *= w
                    losses.append(loss)
                else:
                    losses.append(self.calculate_loss(a, b, w))  # type: ignore
        return losses

    def forward(
        self, input_a: Tensor, input_b: Tensor
    ) -> Union[Tensor, List[Tensor]]:
        features_a = self.model(input_a)
        features_b = self.model(input_b)
        losses = self.calculate_loss(features_a, features_b, self.weights)
        if isinstance(losses[0], list):
            return [sum(loss) for loss in losses]  # type: ignore
        return sum(losses)  # type: ignore
