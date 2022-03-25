import shutil

import pytest

from firewood.utils import data
from tests.helpers.utils import gen_params


@pytest.mark.parametrize(*gen_params("name", ["mnist"]))
def test_torchvision_train_test_val_datasets(name: str):
    (
        train_loader,
        test_loader,
        val_loader,
    ) = data.torchvision_train_test_val_datasets(name, root="./temp")
    assert train_loader is not None
    shutil.rmtree("./temp")
