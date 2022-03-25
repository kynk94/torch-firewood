import importlib
import sys

import pytest

from firewood import models

gan = ["gan." + model for model in models.gan.__all__]
semantic_segmentation = [
    "semantic_segmentation." + model
    for model in models.semantic_segmentation.__all__
]

all_models = gan + semantic_segmentation


@pytest.mark.parametrize("model", all_models)
def test_models(model: str) -> None:
    module = importlib.import_module("firewood.models." + model)
    sys.argv = [""]
    module.main()
