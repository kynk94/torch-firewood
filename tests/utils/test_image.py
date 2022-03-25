import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional_tensor as TFT
from PIL import Image

from firewood.utils import image as im_utils
from tests.helpers.raiseif import expect_raise


def test_make_grid():
    images = torch.randn(100, 3, 10, 10)
    grid = im_utils.make_grid(images.numpy())
    assert grid.shape == (3, 100, 100)

    nrow = random.randint(1, 100)
    grid = im_utils.make_grid(images, nrow=nrow)
    assert grid.shape == (3, math.ceil(100 / nrow) * 10, nrow * 10)

    grid = im_utils.make_grid(images.permute(0, 2, 3, 1), dataformats="NHWC")
    assert grid.shape == (100, 100, 3)

    with expect_raise(ValueError):
        im_utils.make_grid(images, dataformats="invalid")


def test_image_auto_permute():
    images = torch.randn(3, 10, 10)
    permuted_images = im_utils.image_auto_permute(images)
    assert permuted_images.shape == (10, 10, 3)

    images = torch.randn(100, 10, 10, 3)
    permuted_images = im_utils.image_auto_permute(images)
    assert permuted_images.shape == (100, 10, 10, 3)

    permuted_images = im_utils.image_auto_permute(images, target_format="NCHW")
    assert permuted_images.shape == (100, 3, 10, 10)

    images = torch.randn(4, 4, 4)
    permuted_images = im_utils.image_auto_permute(images, ignore_error=True)
    assert permuted_images is images

    images = torch.randn(1, 3, 6)
    permuted_images = im_utils.image_auto_permute(images, ignore_error=True)
    assert permuted_images is images

    images = torch.randn(3, 10, 20)
    permuted_images = im_utils.image_auto_permute(images)
    assert permuted_images.shape == (10, 20, 3)

    with expect_raise(ValueError):
        im_utils.image_auto_permute(images, target_format="invalid")

    images = torch.randn(1, 3, 6)
    with expect_raise(ValueError):
        im_utils.image_auto_permute(images, ignore_error=False)


def test_tensor_to_image():
    tensor = torch.randn(3, 10, 10).clamp_(-1, 1)

    image = im_utils.tensor_to_image(tensor, auto_permute=True)
    assert image.shape == (10, 10, 3)
    assert image.dtype == np.uint8
    assert image.min() >= 0 and image.max() <= 255


def test_normalize_image_array():
    array = np.random.uniform(0, 255, (3, 10, 10)).astype(np.uint8)
    tensor = torch.tensor(array, dtype=torch.uint8)
    normalized_array = im_utils.normalize_image_array(array, 0, 1)
    normalized_tensor = im_utils.normalize_image_array(tensor, 0, 1)
    array = array / 255.0
    tensor = tensor / 255.0
    assert np.allclose(
        array, normalized_array, atol=1e-5
    ), f"Case 1 array mismatch. l1: {np.abs(array - normalized_array).mean()}"
    assert torch.allclose(
        tensor, normalized_tensor, atol=1e-5
    ), f"Case 1 tensor mismatch. l1: {F.l1_loss(tensor, normalized_tensor)}"

    array = np.random.uniform(0, 1, (3, 10, 10))
    tensor = torch.tensor(array, dtype=torch.float32)
    normalized_array = im_utils.normalize_image_array(array, -1, 1)
    normalized_tensor = im_utils.normalize_image_array(tensor, -1, 1)
    array = array * 2.0 - 1.0
    tensor = tensor * 2.0 - 1.0
    assert np.allclose(
        array, normalized_array, atol=1e-5
    ), f"Case 2 array mismatch. l1: {np.abs(array - normalized_array).mean()}"
    assert torch.allclose(
        tensor, normalized_tensor, atol=1e-5
    ), f"Case 2 tensor mismatch. l1: {F.l1_loss(tensor, normalized_tensor)}"

    array = np.random.uniform(-1, 1, (3, 10, 10))
    tensor = torch.tensor(array, dtype=torch.float32)
    normalized_array = im_utils.normalize_image_array(array, 0, 255)
    normalized_tensor = im_utils.normalize_image_array(tensor, 0, 255)
    array = array * 127.5 + 127.5
    tensor = tensor * 127.5 + 127.5
    assert np.allclose(
        array, normalized_array, atol=1e-5
    ), f"Case 3 array mismatch. l1: {np.abs(array - normalized_array).mean()}"
    assert torch.allclose(
        tensor, normalized_tensor, atol=1e-5
    ), f"Case 3 tensor mismatch. l1: {F.l1_loss(tensor, normalized_tensor)}"

    array = np.random.uniform(-2, 2, (3, 10, 10))
    tensor = torch.tensor(array, dtype=torch.float32)
    array_min, array_max = array.min(), array.max()
    tensor_min, tensor_max = tensor.min(), tensor.max()
    normalized_array = im_utils.normalize_image_array(array, 0, 1)
    normalized_tensor = im_utils.normalize_image_array(tensor, 0, 1)
    manual_normalized_array = (array - array_min) / (array_max - array_min)
    manual_noramlized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    assert np.allclose(
        manual_normalized_array, normalized_array, atol=1e-5
    ), f"Case 4 array mismatch. l1: {np.abs(manual_normalized_array - normalized_array).mean()}"
    assert torch.allclose(
        manual_noramlized_tensor, normalized_tensor, atol=1e-5
    ), f"Case 4 tensor mismatch. l1: {F.l1_loss(manual_noramlized_tensor, normalized_tensor)}"


def test_batch_flat_to_square():
    for _ in range(10):
        channel = random.choice([5, 7, 11, 13, 17, 19])
        embedding = random.randint(20, 30)
        batch_array = np.random.normal(size=(2, channel * embedding**2))
        batch_tensor = torch.tensor(batch_array, dtype=torch.float32)
        squared_array = im_utils.batch_flat_to_square(batch_array, channel)
        squared_tensor = im_utils.batch_flat_to_square(batch_tensor, channel)
        assert squared_array.shape == (2, channel, embedding, embedding)
        assert squared_tensor.shape == (2, channel, embedding, embedding)

    batch_array = np.random.normal(size=(2, 100))
    batch_tensor = torch.tensor(batch_array, dtype=torch.float32)
    squared_array = im_utils.batch_flat_to_square(batch_array)
    squared_tensor = im_utils.batch_flat_to_square(batch_tensor)
    assert squared_array.shape == (2, 1, 10, 10)
    assert squared_tensor.shape == (2, 1, 10, 10)

    batch_array = np.random.normal(size=(2, 300))
    batch_tensor = torch.tensor(batch_array, dtype=torch.float32)
    squared_array = im_utils.batch_flat_to_square(batch_array)
    squared_tensor = im_utils.batch_flat_to_square(batch_tensor)
    assert squared_array.shape == (2, 3, 10, 10)
    assert squared_tensor.shape == (2, 3, 10, 10)

    with expect_raise(ValueError):
        im_utils.batch_flat_to_square(torch.randn(2, 500))


def test_match_channels():
    a = torch.randn(2, 3, 10, 10)
    new_a, new_a_ = im_utils.match_channels_between_NCHW_tensor(a, a)
    assert new_a.shape == new_a_.shape
    assert new_a.size(1) == 3

    b = torch.randn(2, 4, 10, 10)
    new_a, new_b = im_utils.match_channels_between_NCHW_tensor(a, b)
    assert new_a.shape == new_b.shape
    assert new_a.size(1) == 4

    a = torch.randn(2, 1, 10, 10)
    b = torch.randn(2, 4, 10, 10)
    new_a, new_b = im_utils.match_channels_between_NCHW_tensor(a, b)
    assert new_a.shape == new_b.shape
    assert new_a.size(1) == 4
    new_b, new_a = im_utils.match_channels_between_NCHW_tensor(b, a)
    assert new_a.shape == new_b.shape
    assert new_a.size(1) == 4

    a = torch.randn(2, 1, 10, 10)
    b = torch.randn(2, 10, 10, 10)
    new_a, new_b = im_utils.match_channels_between_NCHW_tensor(a, b)
    assert new_a.shape == new_b.shape
    assert new_a.size(1) == 10
    new_b, new_a = im_utils.match_channels_between_NCHW_tensor(b, a)
    assert new_a.shape == new_b.shape
    assert new_a.size(1) == 10

    with expect_raise(ValueError):
        a = torch.randn(2, 4, 10, 10)
        b = torch.randn(2, 5, 10, 10)
        im_utils.match_channels_between_NCHW_tensor(a, b)
    with expect_raise(ValueError):
        a = torch.randn(2, 4, 10, 10)
        b = torch.randn(2, 4, 20, 20)
        im_utils.match_channels_between_NCHW_tensor(a, b)


def test_RGB_alpha_smoothing() -> None:
    input = np.random.uniform(0, 255, (128, 128, 3)).astype(np.uint8)
    alpha = np.random.uniform(0, 255, (128, 128)).astype(np.uint8)
    background_color = (0, 128, 255)

    input_image = Image.fromarray(input)
    alpha_image = Image.fromarray(alpha)

    output = im_utils.alpha_smoothing(input, alpha, background_color)
    alpha = np.expand_dims(alpha.astype(np.float32), -1) / 255
    manual_output = input * alpha + (1 - alpha) * np.array(background_color)

    assert np.allclose(
        output, manual_output
    ), f"Array output value mismatch. l1: {np.abs(output - manual_output).mean()}"

    output = im_utils.alpha_smoothing(
        input_image, alpha_image, np.array(background_color)
    )
    assert np.allclose(
        output, manual_output
    ), f"Image output value mismatch. l1: {np.abs(output - manual_output).mean()}"

    alpha_image = alpha_image.resize((256, 256))
    output = im_utils.alpha_smoothing(
        input_image, alpha_image, background_color
    )
    alpha = np.expand_dims(np.array(alpha_image.resize((128, 128))), -1) / 255
    manual_output = input * alpha + (1 - alpha) * np.array(background_color)
    assert np.allclose(
        output, manual_output
    ), f"Resized Image output value mismatch. l1: {np.abs(output - manual_output).mean()}"

    input = np.random.uniform(0, 255, (128, 128, 4)).astype(np.uint8)
    alpha = input.astype(np.float32)[..., -1:] / 255

    background_color = 128
    output = im_utils.alpha_smoothing(input, None, background_color)
    manual_output = input[..., :-1] * alpha + (1 - alpha) * np.array(
        background_color
    )
    assert np.allclose(
        output, manual_output
    ), f"Array output value mismatch. l1: {np.abs(output - manual_output).mean()}"


def test_GRAY_alpha_smoothing() -> None:
    input = np.random.uniform(0, 255, (128, 128)).astype(np.uint8)
    alpha = np.random.uniform(0, 255, (128, 128)).astype(np.uint8)
    alpha_stacked_input = np.stack([input, alpha], axis=-1)
    background_color = (0, 128, 255)

    input_image = Image.fromarray(input)
    alpha_image = Image.fromarray(alpha)

    output = im_utils.alpha_smoothing(input, alpha, background_color)
    alpha = alpha.astype(np.float32) / 255
    gray_background_color = np.matmul(
        np.array(background_color, dtype=np.float32),
        np.array([0.299, 0.587, 0.114], dtype=np.float32),
    )
    manual_output = input * alpha + (1 - alpha) * gray_background_color

    assert np.allclose(
        output, manual_output
    ), f"Array output value mismatch. l1: {np.abs(output - manual_output).mean()}"

    output = im_utils.alpha_smoothing(
        alpha_stacked_input, None, background_color
    )[..., 0]
    assert np.allclose(
        output, manual_output
    ), f"Stacked Array output value mismatch. l1: {np.abs(output - manual_output).mean()}"

    output = im_utils.alpha_smoothing(
        input_image, alpha_image, background_color
    )
    assert np.allclose(
        output, manual_output
    ), f"Image output value mismatch. l1: {np.abs(output - manual_output).mean()}"


def test_invalid_alpha_smoothing() -> None:
    input = np.random.uniform(0, 255, (128, 128, 5)).astype(np.uint8)
    alpha = np.random.uniform(0, 255, (128, 128)).astype(np.uint8)
    background_color = (0, 128, 255)

    alpha_image = Image.fromarray(alpha)
    with expect_raise(ValueError):
        im_utils.alpha_smoothing(input, None)
    with expect_raise(ValueError):
        im_utils.alpha_smoothing(input, alpha_image)
    with expect_raise(ValueError):
        im_utils.alpha_smoothing(alpha, alpha, (0, 128))


def test_RGB_alpha_smoothing_NCHW_tensor():
    int_input = (
        (torch.randn(1, 3, 128, 128) * 127.5 + 127.5)
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
    )
    int_alpha = (
        (torch.randn(1, 1, 128, 128) * 127.5 + 127.5)
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
    )
    float_input = int_input.float()
    float_alpha = int_alpha.float()
    background_color = (0, 128, 255)
    background_color_tensor = torch.tensor(
        background_color, dtype=torch.float32
    ).view(1, -1, 1, 1)

    output = im_utils.alpha_smoothing_NCHW_tensor(
        int_input, int_alpha, background_color
    )
    manual_output = (
        float_input * float_alpha
        + (255 - float_alpha) * background_color_tensor
    ) / 255
    assert torch.allclose(
        output, manual_output
    ), f"Array output value mismatch. l1: {torch.abs(output - manual_output).mean()}"

    output = im_utils.alpha_smoothing_NCHW_tensor(
        torch.cat([int_input, int_alpha], dim=1), None, background_color_tensor
    )
    assert torch.allclose(
        output, manual_output
    ), f"Concat output value mismatch. l1: {torch.abs(output - manual_output).mean()}"

    upscaled_alpha = TFT.resize(
        float_alpha, size=256, interpolation="bilinear", antialias=True
    )
    downscaled_alpha = TFT.resize(
        upscaled_alpha, size=128, interpolation="bilinear", antialias=True
    )
    output = im_utils.alpha_smoothing_NCHW_tensor(
        float_input, upscaled_alpha, 128
    )
    manual_output = (
        float_input * downscaled_alpha + (255 - downscaled_alpha) * 128
    ) / 255
    assert torch.allclose(
        output, manual_output, atol=1e-5
    ), f"Resized output value mismatch. l1: {torch.abs(output - manual_output).mean()}"


def test_GRAY_alpha_smoothing_NCHW_tensor():
    int_input = (
        (torch.randn(1, 1, 128, 128) * 127.5 + 127.5)
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
    )
    int_alpha = (
        (torch.randn(1, 1, 128, 128) * 127.5 + 127.5)
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
    )
    float_input = int_input.float()
    float_alpha = int_alpha.float()
    background_color = (0, 128, 255)
    background_color_tensor = torch.tensor(
        background_color, dtype=torch.float32
    ).view(1, -1, 1, 1)
    gray_background_color = np.matmul(
        np.array(background_color, dtype=np.float32),
        np.array([0.299, 0.587, 0.114], dtype=np.float32),
    )
    gray_background_color_tensor = torch.tensor(
        gray_background_color, dtype=torch.float32
    ).view(1, -1, 1, 1)

    output = im_utils.alpha_smoothing_NCHW_tensor(
        int_input, int_alpha, background_color
    )
    manual_output = (
        float_input * float_alpha
        + (255 - float_alpha) * gray_background_color_tensor
    ) / 255

    assert np.allclose(
        output, manual_output
    ), f"Array output value mismatch. l1: {F.l1_loss(output, manual_output)}"

    output = im_utils.alpha_smoothing_NCHW_tensor(
        torch.cat([int_input, int_alpha], dim=1), None, background_color_tensor
    )
    assert np.allclose(
        output, manual_output
    ), f"Concat output value mismatch. l1: {F.l1_loss(output, manual_output)}"


def test_invalid_alpha_smoothing_NCHW_tensor() -> None:
    input = np.random.uniform(0, 255, (1, 5, 128, 128)).astype(np.uint8)
    alpha = np.random.uniform(0, 255, (1, 1, 128, 128)).astype(np.uint8)
    int_input = torch.tensor(input, dtype=torch.uint8)
    int_alpha = torch.tensor(alpha, dtype=torch.uint8)
    background_color = (0, 128, 255)

    with expect_raise(ValueError):
        im_utils.alpha_smoothing_NCHW_tensor(int_input, None)
    with expect_raise(ValueError):
        im_utils.alpha_smoothing_NCHW_tensor(int_input, int_alpha)
    with expect_raise(ValueError):
        im_utils.alpha_smoothing_NCHW_tensor(int_alpha, int_alpha, (0, 128))
