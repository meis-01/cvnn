"""Backpropagation for ``studies.activations`` functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from studies.activations import ACTIVATIONS, get_activation
from studies.activations.complex_valued import SELU_ALPHA, SELU_SCALE


ArrayLike = complex | float | int | np.ndarray
ActivationBackward = Callable[[np.ndarray, "ActivationCache"], np.ndarray]
EPSILON = 1e-12


@dataclass(frozen=True)
class ActivationCache:
    """Values needed by activation backward passes."""

    name: str
    z: np.ndarray
    y: np.ndarray
    params: dict[str, Any]


def _as_complex_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.complex128)


def _as_real_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _gelu_derivative(x: np.ndarray) -> np.ndarray:
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    inner_derivative = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * inner_derivative


def _split_gradient(grad_y_real: ArrayLike, grad_y_imag: ArrayLike) -> np.ndarray:
    return _as_real_array(grad_y_real) + 1j * _as_real_array(grad_y_imag)


def _split_to_wirtinger(grad_split: np.ndarray) -> np.ndarray:
    return 0.5 * _as_complex_array(grad_split)


def _wirtinger_to_split(grad_conj: ArrayLike) -> np.ndarray:
    return 2.0 * _as_complex_array(grad_conj)


def activation_forward(name: str, z: ArrayLike, **params: Any) -> tuple[np.ndarray, ActivationCache]:
    """Run an activation from ``studies.activations`` and keep a cache."""

    normalized = name.lower().replace("-", "_")
    activation = get_activation(normalized)
    z_array = _as_complex_array(z)
    y = _as_complex_array(activation(z_array, **params))
    return y, ActivationCache(name=normalized, z=z_array, y=y, params=params)


def _holomorphic_backward(grad_y: np.ndarray, derivative: np.ndarray) -> np.ndarray:
    return grad_y * derivative.conj()


def _split_backward(grad_y: np.ndarray, z: np.ndarray, derivative: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    return grad_y.real * derivative(z.real) + 1j * grad_y.imag * derivative(z.imag)


def _radial_backward(
    grad_y: np.ndarray,
    z: np.ndarray,
    amplitude: np.ndarray,
    amplitude_derivative: np.ndarray,
) -> np.ndarray:
    """VJP for ``f(z) = amplitude(abs(z)) * z / abs(z)`` in split R2."""

    x = z.real
    y = z.imag
    radius = np.abs(z)
    safe = radius > EPSILON

    scale = np.divide(amplitude, radius, out=np.zeros_like(radius), where=safe)
    radial_term = np.divide(amplitude_derivative * radius - amplitude, radius**3, out=np.zeros_like(radius), where=safe)

    dot = grad_y.real * x + grad_y.imag * y
    grad_x = scale * grad_y.real + radial_term * x * dot
    grad_y_imag = scale * grad_y.imag + radial_term * y * dot
    return grad_x + 1j * grad_y_imag


def _identity_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    return grad_y


def _complex_tanh_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    return _holomorphic_backward(grad_y, 1.0 - np.tanh(cache.z) ** 2)


def _complex_sigmoid_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    sigmoid_z = 1.0 / (1.0 + np.exp(-cache.z))
    return _holomorphic_backward(grad_y, sigmoid_z * (1.0 - sigmoid_z))


def _split_relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def _split_leaky_relu_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    negative_slope = float(cache.params.get("negative_slope", 0.01))
    return _split_backward(grad_y, cache.z, lambda x: np.where(x >= 0.0, 1.0, negative_slope))


def _split_elu_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    alpha = float(cache.params.get("alpha", 1.0))
    return _split_backward(grad_y, cache.z, lambda x: np.where(x >= 0.0, 1.0, alpha * np.exp(x)))


def _split_selu_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    alpha = float(cache.params.get("alpha", SELU_ALPHA))
    scale = float(cache.params.get("scale", SELU_SCALE))
    return _split_backward(grad_y, cache.z, lambda x: scale * np.where(x >= 0.0, 1.0, alpha * np.exp(x)))


def _split_sigmoid_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    return _split_backward(grad_y, cache.z, lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)))


def _split_tanh_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    return _split_backward(grad_y, cache.z, lambda x: 1.0 - np.tanh(x) ** 2)


def _split_softplus_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    return _split_backward(grad_y, cache.z, _sigmoid)


def _split_swish_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    beta = float(cache.params.get("beta", 1.0))

    def derivative(x: np.ndarray) -> np.ndarray:
        probability = _sigmoid(beta * x)
        return probability + beta * x * probability * (1.0 - probability)

    return _split_backward(grad_y, cache.z, derivative)


def _split_gelu_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    return _split_backward(grad_y, cache.z, _gelu_derivative)


def _zrelu_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    mask = (cache.z.real > 0.0) & (cache.z.imag > 0.0)
    return np.where(mask, grad_y, 0.0 + 0.0j)


def _modrelu_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    bias = float(cache.params.get("bias", 0.0))
    radius = np.abs(cache.z)
    preactivation = radius + bias
    amplitude = np.maximum(preactivation, 0.0)
    amplitude_derivative = (preactivation > 0.0).astype(np.float64)
    return _radial_backward(grad_y, cache.z, amplitude, amplitude_derivative)


def _cardioid_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    x = cache.z.real
    y = cache.z.imag
    radius = np.abs(cache.z)
    safe = radius > EPSILON

    scale = 0.5 * (1.0 + np.divide(x, radius, out=np.zeros_like(radius), where=safe))
    dscale_dx = 0.5 * np.divide(y**2, radius**3, out=np.zeros_like(radius), where=safe)
    dscale_dy = -0.5 * np.divide(x * y, radius**3, out=np.zeros_like(radius), where=safe)

    du_dx = scale + x * dscale_dx
    du_dy = x * dscale_dy
    dv_dx = y * dscale_dx
    dv_dy = scale + y * dscale_dy

    grad_x = grad_y.real * du_dx + grad_y.imag * dv_dx
    grad_y_imag = grad_y.real * du_dy + grad_y.imag * dv_dy
    return grad_x + 1j * grad_y_imag


def _amplitude_tanh_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    radius = np.abs(cache.z)
    amplitude = np.tanh(radius)
    amplitude_derivative = 1.0 - amplitude**2
    return _radial_backward(grad_y, cache.z, amplitude, amplitude_derivative)


def _amplitude_sigmoid_backward(grad_y: np.ndarray, cache: ActivationCache) -> np.ndarray:
    radius = np.abs(cache.z)
    amplitude = _sigmoid(radius)
    amplitude_derivative = amplitude * (1.0 - amplitude)
    return _radial_backward(grad_y, cache.z, amplitude, amplitude_derivative)


ACTIVATION_BACKWARD: dict[str, ActivationBackward] = {
    "identity": _identity_backward,
    "complex_tanh": _complex_tanh_backward,
    "complex_sigmoid": _complex_sigmoid_backward,
    "split_relu": lambda grad_y, cache: _split_backward(grad_y, cache.z, _split_relu_derivative),
    "crelu": lambda grad_y, cache: _split_backward(grad_y, cache.z, _split_relu_derivative),
    "split_leaky_relu": _split_leaky_relu_backward,
    "split_elu": _split_elu_backward,
    "split_selu": _split_selu_backward,
    "split_sigmoid": _split_sigmoid_backward,
    "split_tanh": _split_tanh_backward,
    "split_softplus": _split_softplus_backward,
    "split_swish": _split_swish_backward,
    "split_gelu": _split_gelu_backward,
    "zrelu": _zrelu_backward,
    "modrelu": _modrelu_backward,
    "cardioid": _cardioid_backward,
    "amplitude_tanh": _amplitude_tanh_backward,
    "amplitude_sigmoid": _amplitude_sigmoid_backward,
}


def activation_backward_split_r2(
    grad_y_real: ArrayLike,
    grad_y_imag: ArrayLike,
    cache: ActivationCache,
) -> np.ndarray:
    """Backprop through a cached activation in split R2 form."""

    try:
        backward = ACTIVATION_BACKWARD[cache.name]
    except KeyError as error:
        available = ", ".join(sorted(ACTIVATION_BACKWARD))
        raise ValueError(f"Unknown activation backward '{cache.name}'. Available: {available}") from error
    return backward(_split_gradient(grad_y_real, grad_y_imag), cache)


def activation_backward_wirtinger(grad_y_conj: ArrayLike, cache: ActivationCache) -> np.ndarray:
    """Backprop through a cached activation with Wirtinger gradients."""

    grad_y_split = _wirtinger_to_split(grad_y_conj)
    grad_z_split = activation_backward_split_r2(grad_y_split.real, grad_y_split.imag, cache)
    return _split_to_wirtinger(grad_z_split)


def missing_activation_backwards() -> set[str]:
    """Return activation registry names that do not have a backward function."""

    return set(ACTIVATIONS) - set(ACTIVATION_BACKWARD)
