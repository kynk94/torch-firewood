import shutil

import pytest

from firewood.trainer.utils import data


@pytest.mark.parametrize("name", ["mnist"])
def test_torchvision_train_val_test_datasets(name: str):
    (
        train_loader,
        val_loader,
        test_loader,
    ) = data.torchvision_train_val_test_datasets(name, root="./temp")
    assert train_loader is not None
    shutil.rmtree("./temp")
