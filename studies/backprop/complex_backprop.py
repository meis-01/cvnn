"""Complex backpropagation written two equivalent ways.

The code uses a dense complex layer

    y = x @ W.T + b

and a real-valued mean-squared error loss

    L = mean_batch(sum_outputs(|y - target|**2)).

Two gradient conventions are shown:

1. split R2 backprop treats real and imaginary parts as independent real
   coordinates.
2. Wirtinger backprop propagates dL / d(conj(z)), which is the gradient used
   for steepest descent on real-valued losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = complex | float | int | np.ndarray


@dataclass(frozen=True)
class DenseCache:
    """Values needed by the dense-layer backward passes."""

    x: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    y: np.ndarray


def dense_forward(x: ArrayLike, weights: ArrayLike, bias: ArrayLike) -> tuple[np.ndarray, DenseCache]:
    """Run a complex dense layer.

    Args:
        x: Input with shape ``(batch, in_features)``.
        weights: Complex weight matrix with shape ``(out_features, in_features)``.
        bias: Complex bias vector with shape ``(out_features,)``.

    Returns:
        The layer output and a cache for backpropagation.
    """

    x_array = np.asarray(x, dtype=np.complex128)
    weights_array = np.asarray(weights, dtype=np.complex128)
    bias_array = np.asarray(bias, dtype=np.complex128)

    y = x_array @ weights_array.T + bias_array
    return y, DenseCache(x=x_array, weights=weights_array, bias=bias_array, y=y)


def mse_loss_split_r2(y: ArrayLike, target: ArrayLike) -> tuple[float, np.ndarray, np.ndarray]:
    """MSE loss and upstream gradients in split real/imaginary coordinates.

    Returns ``(loss, dL/dRe(y), dL/dIm(y))``.
    """

    y_array = np.asarray(y, dtype=np.complex128)
    target_array = np.asarray(target, dtype=np.complex128)
    error = y_array - target_array
    batch_size = y_array.shape[0]

    loss = float(np.sum(np.abs(error) ** 2) / batch_size)
    grad_y_real = 2.0 * error.real / batch_size
    grad_y_imag = 2.0 * error.imag / batch_size
    return loss, grad_y_real, grad_y_imag


def dense_backward_split_r2(
    grad_y_real: ArrayLike,
    grad_y_imag: ArrayLike,
    cache: DenseCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backprop through the dense layer by expanding C into R2.

    If ``x = a + i b``, ``W = A + i B``, and ``y = u + i v``, then

        u = a @ A.T - b @ B.T + Re(bias)
        v = a @ B.T + b @ A.T + Im(bias)

    This function differentiates those two real equations directly and returns
    complex-valued gradients for ``x``, ``weights``, and ``bias``.
    """

    grad_u = np.asarray(grad_y_real, dtype=np.float64)
    grad_v = np.asarray(grad_y_imag, dtype=np.float64)

    a = cache.x.real
    b = cache.x.imag
    weights_real = cache.weights.real
    weights_imag = cache.weights.imag

    grad_a = grad_u @ weights_real + grad_v @ weights_imag
    grad_b = -grad_u @ weights_imag + grad_v @ weights_real

    grad_weights_real = grad_u.T @ a + grad_v.T @ b
    grad_weights_imag = -grad_u.T @ b + grad_v.T @ a

    grad_bias_real = np.sum(grad_u, axis=0)
    grad_bias_imag = np.sum(grad_v, axis=0)

    grad_x = grad_a + 1j * grad_b
    grad_weights = grad_weights_real + 1j * grad_weights_imag
    grad_bias = grad_bias_real + 1j * grad_bias_imag
    return grad_x, grad_weights, grad_bias


def mse_loss_wirtinger(y: ArrayLike, target: ArrayLike) -> tuple[float, np.ndarray]:
    """MSE loss and upstream Wirtinger gradient ``dL/dconj(y)``."""

    y_array = np.asarray(y, dtype=np.complex128)
    target_array = np.asarray(target, dtype=np.complex128)
    error = y_array - target_array
    batch_size = y_array.shape[0]

    loss = float(np.sum(np.abs(error) ** 2) / batch_size)
    grad_y_conj = error / batch_size
    return loss, grad_y_conj


def dense_backward_wirtinger(
    grad_y_conj: ArrayLike,
    cache: DenseCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backprop through the dense layer with Wirtinger calculus.

    The returned gradients are ``dL/dconj(x)``, ``dL/dconj(weights)``, and
    ``dL/dconj(bias)``. For a real-valued loss, gradient descent updates use
    ``parameter -= learning_rate * dL/dconj(parameter)``.
    """

    grad_y = np.asarray(grad_y_conj, dtype=np.complex128)

    grad_x_conj = grad_y @ cache.weights.conj()
    grad_weights_conj = grad_y.T @ cache.x.conj()
    grad_bias_conj = np.sum(grad_y, axis=0)
    return grad_x_conj, grad_weights_conj, grad_bias_conj


def split_r2_to_wirtinger_gradient(grad_real: ArrayLike, grad_imag: ArrayLike) -> np.ndarray:
    """Convert R2 gradients to ``dL/dconj(z)``.

    For ``z = x + i y`` and a real-valued loss ``L``,

        dL/dconj(z) = 0.5 * (dL/dx + i dL/dy).
    """

    return 0.5 * (np.asarray(grad_real, dtype=np.float64) + 1j * np.asarray(grad_imag, dtype=np.float64))
