"""Run a small equivalence check for the complex backpropagation examples."""

from __future__ import annotations

import numpy as np

from studies.backprop import (
    activation_backward_wirtinger,
    activation_forward,
    complex_conv1d_forward,
    complex_conv2d_forward,
    complex_conv_transpose1d_forward,
    complex_conv_transpose2d_forward,
    complex_transformer_encoder_forward,
    conv1d_backward_split_r2,
    conv1d_backward_wirtinger,
    conv2d_backward_split_r2,
    conv2d_backward_wirtinger,
    conv_transpose1d_backward_split_r2,
    conv_transpose1d_backward_wirtinger,
    conv_transpose2d_backward_split_r2,
    conv_transpose2d_backward_wirtinger,
    dense_backward_split_r2,
    dense_backward_wirtinger,
    dense_forward,
    transformer_encoder_backward_wirtinger,
)
from studies.activations import ACTIVATIONS
from studies.loss_functions import reconstruction_mse_split_r2, reconstruction_mse_wirtinger
from studies.maps import TransformerParams


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _dense_demo(rng: np.random.Generator) -> None:
    x = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    weights = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    bias = rng.normal(size=2) + 1j * rng.normal(size=2)
    target = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))

    y, cache = dense_forward(x, weights, bias)

    loss_r2, grad_y_real, grad_y_imag = reconstruction_mse_split_r2(y, target)
    grad_x_r2, grad_weights_r2, grad_bias_r2 = dense_backward_split_r2(
        grad_y_real,
        grad_y_imag,
        cache,
    )

    loss_w, grad_y_conj = reconstruction_mse_wirtinger(y, target)
    grad_x_w, grad_weights_w, grad_bias_w = dense_backward_wirtinger(grad_y_conj, cache)

    print(f"dense split R2 loss: {loss_r2:.12f}")
    print(f"dense Wirtinger loss: {loss_w:.12f}")
    print("dense grad_x match:", np.allclose(0.5 * grad_x_r2, grad_x_w))
    print("dense grad_weights match:", np.allclose(0.5 * grad_weights_r2, grad_weights_w))
    print("dense grad_bias match:", np.allclose(0.5 * grad_bias_r2, grad_bias_w))


def _conv1d_demo(rng: np.random.Generator) -> None:
    x = _complex_normal(rng, (2, 6, 3))
    kernel = _complex_normal(rng, (4, 3, 3))
    bias = _complex_normal(rng, (4,))
    target = _complex_normal(rng, (2, 6, 4))
    y, cache = complex_conv1d_forward(x, kernel, bias, padding="same")

    _, grad_y_real, grad_y_imag = reconstruction_mse_split_r2(y, target)
    grad_x_r2, grad_kernel_r2, grad_bias_r2 = conv1d_backward_split_r2(grad_y_real, grad_y_imag, cache)
    _, grad_y_conj = reconstruction_mse_wirtinger(y, target)
    grad_x_w, grad_kernel_w, grad_bias_w = conv1d_backward_wirtinger(grad_y_conj, cache)

    print("conv1d grad_x match:", np.allclose(0.5 * grad_x_r2, grad_x_w))
    print("conv1d grad_kernel match:", np.allclose(0.5 * grad_kernel_r2, grad_kernel_w))
    print("conv1d grad_bias match:", np.allclose(0.5 * grad_bias_r2, grad_bias_w))


def _conv2d_demo(rng: np.random.Generator) -> None:
    x = _complex_normal(rng, (2, 5, 4, 2))
    kernel = _complex_normal(rng, (3, 3, 2, 2))
    bias = _complex_normal(rng, (3,))
    y, cache = complex_conv2d_forward(x, kernel, bias, padding="same")
    target = _complex_normal(rng, y.shape)

    _, grad_y_real, grad_y_imag = reconstruction_mse_split_r2(y, target)
    grad_x_r2, grad_kernel_r2, grad_bias_r2 = conv2d_backward_split_r2(grad_y_real, grad_y_imag, cache)
    _, grad_y_conj = reconstruction_mse_wirtinger(y, target)
    grad_x_w, grad_kernel_w, grad_bias_w = conv2d_backward_wirtinger(grad_y_conj, cache)

    print("conv2d grad_x match:", np.allclose(0.5 * grad_x_r2, grad_x_w))
    print("conv2d grad_kernel match:", np.allclose(0.5 * grad_kernel_r2, grad_kernel_w))
    print("conv2d grad_bias match:", np.allclose(0.5 * grad_bias_r2, grad_bias_w))


def _conv_transpose_demo(rng: np.random.Generator) -> None:
    x1 = _complex_normal(rng, (2, 5, 3))
    kernel1 = _complex_normal(rng, (3, 3, 4))
    bias1 = _complex_normal(rng, (4,))
    y1, cache1 = complex_conv_transpose1d_forward(x1, kernel1, bias1, stride=2, padding=1, output_padding=1)
    target1 = _complex_normal(rng, y1.shape)
    _, grad_y1_real, grad_y1_imag = reconstruction_mse_split_r2(y1, target1)
    grad_x1_r2, grad_kernel1_r2, grad_bias1_r2 = conv_transpose1d_backward_split_r2(grad_y1_real, grad_y1_imag, cache1)
    _, grad_y1_conj = reconstruction_mse_wirtinger(y1, target1)
    grad_x1_w, grad_kernel1_w, grad_bias1_w = conv_transpose1d_backward_wirtinger(grad_y1_conj, cache1)
    print("conv_transpose1d grad_x match:", np.allclose(0.5 * grad_x1_r2, grad_x1_w))
    print("conv_transpose1d grad_kernel match:", np.allclose(0.5 * grad_kernel1_r2, grad_kernel1_w))
    print("conv_transpose1d grad_bias match:", np.allclose(0.5 * grad_bias1_r2, grad_bias1_w))

    x2 = _complex_normal(rng, (2, 4, 3, 2))
    kernel2 = _complex_normal(rng, (2, 3, 2, 5))
    bias2 = _complex_normal(rng, (5,))
    y2, cache2 = complex_conv_transpose2d_forward(x2, kernel2, bias2, stride=(2, 2), padding=(1, 0), output_padding=(1, 0))
    target2 = _complex_normal(rng, y2.shape)
    _, grad_y2_real, grad_y2_imag = reconstruction_mse_split_r2(y2, target2)
    grad_x2_r2, grad_kernel2_r2, grad_bias2_r2 = conv_transpose2d_backward_split_r2(grad_y2_real, grad_y2_imag, cache2)
    _, grad_y2_conj = reconstruction_mse_wirtinger(y2, target2)
    grad_x2_w, grad_kernel2_w, grad_bias2_w = conv_transpose2d_backward_wirtinger(grad_y2_conj, cache2)
    print("conv_transpose2d grad_x match:", np.allclose(0.5 * grad_x2_r2, grad_x2_w))
    print("conv_transpose2d grad_kernel match:", np.allclose(0.5 * grad_kernel2_r2, grad_kernel2_w))
    print("conv_transpose2d grad_bias match:", np.allclose(0.5 * grad_bias2_r2, grad_bias2_w))


def _transformer_demo(rng: np.random.Generator) -> None:
    model_dim = 4
    x = _complex_normal(rng, (2, 5, model_dim))
    params = TransformerParams(
        query_weights=_complex_normal(rng, (model_dim, model_dim)),
        key_weights=_complex_normal(rng, (model_dim, model_dim)),
        value_weights=_complex_normal(rng, (model_dim, model_dim)),
        output_weights=_complex_normal(rng, (model_dim, model_dim)),
        feedforward_in_weights=_complex_normal(rng, (8, model_dim)),
        feedforward_out_weights=_complex_normal(rng, (model_dim, 8)),
    )
    y, cache = complex_transformer_encoder_forward(x, params, num_heads=2, use_layer_norm=True)
    target = _complex_normal(rng, y.shape)
    _, grad_y_conj = reconstruction_mse_wirtinger(y, target)
    grad_x, grad_params = transformer_encoder_backward_wirtinger(grad_y_conj, cache)

    print("transformer grad_x:", grad_x.shape)
    print("transformer grad_query_weights:", grad_params.query_weights.shape)
    print("transformer grad_feedforward_in_weights:", grad_params.feedforward_in_weights.shape)


def _activation_demo(rng: np.random.Generator) -> None:
    z = _complex_normal(rng, (3, 4))
    grad = _complex_normal(rng, (3, 4))
    checked = []
    for name in sorted(ACTIVATIONS):
        params = {"bias": -0.2} if name == "modrelu" else {}
        y, cache = activation_forward(name, z, **params)
        grad_z = activation_backward_wirtinger(grad, cache)
        checked.append(name)
        if y.shape != z.shape or grad_z.shape != z.shape:
            raise AssertionError(f"activation {name} produced an unexpected shape")
    print("activation backwards:", len(checked))


def main() -> None:
    rng = np.random.default_rng(7)
    _dense_demo(rng)
    _conv1d_demo(rng)
    _conv2d_demo(rng)
    _conv_transpose_demo(rng)
    _transformer_demo(rng)
    _activation_demo(rng)


if __name__ == "__main__":
    main()
