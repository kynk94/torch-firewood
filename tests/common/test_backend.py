import pytest

from firewood.common import backend


def test_lr_equalization() -> None:
    assert backend.lr_equalization() is False
    backend.set_lr_equalization(True)
    assert backend.lr_equalization() is True
    backend.set_lr_equalization(False)
    assert backend.lr_equalization() is False


@pytest.mark.xfail(raises=TypeError)
def test_lr_equalization_failed() -> None:
    backend.set_lr_equalization("invalid")


def test_conv_weight_gradients_disabled() -> None:
    assert backend.weight_gradients_disabled() is False
    backend.set_conv_weight_gradients_disabled(True)
    assert backend.weight_gradients_disabled() is True
    backend.set_conv_weight_gradients_disabled(False)
    assert backend.weight_gradients_disabled() is False


@pytest.mark.xfail(raises=TypeError)
def test_conv_weight_gradients_disabled_failed() -> None:
    backend.set_conv_weight_gradients_disabled("invalid")


def test_runtime_build() -> None:
    assert backend.runtime_build() is False
    backend.set_runtime_build(True)
    assert backend.runtime_build() is True
    backend.set_runtime_build(False)
    assert backend.runtime_build() is False


@pytest.mark.xfail(raises=TypeError)
def test_runtime_build_failed() -> None:
    backend.set_runtime_build("invalid")
