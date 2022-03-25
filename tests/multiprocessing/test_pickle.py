import os
import pickle
import shutil

import pytest
import torch

from firewood import layers
from firewood.layers.biased_activations import ACTIVATIONS


def dump_pickle(obj, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_biased_activation_cpu(activation):
    bias_act = layers.BiasedActivation(activation)
    assert bias_act.activation == activation

    pickle_path = f"./temp/{activation}.pickle"
    dump_pickle(bias_act, pickle_path)
    bias_act_loaded = load_pickle(pickle_path)

    shutil.rmtree("./temp")
    assert (
        bias_act_loaded.activation == activation
    ), "Biased activation not loaded correctly"


def test_gfix_conv():
    gfix_conv = layers.GFixConv2d(1, 1, 1)

    pickle_path = f"./temp/gfix_conv.pickle"
    dump_pickle(gfix_conv, pickle_path)
    gfix_conv_loaded = load_pickle(pickle_path)

    shutil.rmtree("./temp")
    assert (
        gfix_conv_loaded.in_channels == gfix_conv.in_channels
        and gfix_conv_loaded.out_channels == gfix_conv.out_channels
        and gfix_conv_loaded.kernel_size == gfix_conv.kernel_size
    ), "GFixConv2d not loaded correctly"


def test_upfirdn():
    upfirdn = layers.UpFirDn2d([1, 3, 3, 1], up=2, down=2)

    pickle_path = f"./temp/upfirdn.pickle"
    dump_pickle(upfirdn, pickle_path)
    upfirdn_loaded = load_pickle(pickle_path)

    shutil.rmtree("./temp")
    assert (
        torch.allclose(upfirdn_loaded.kernel, upfirdn.kernel)
        and upfirdn_loaded.up == upfirdn.up
        and upfirdn_loaded.down == upfirdn.down
    ), "UpFirDn2d not loaded correctly"
