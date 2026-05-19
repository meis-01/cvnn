"""Common complex-valued activation functions.

The functions in this module accept Python complex scalars or NumPy arrays and
return NumPy arrays/scalars following NumPy broadcasting rules. The collection
groups the complex activations most often used in complex-valued neural network
studies:

- holomorphic functions, such as complex tanh and sigmoid
- split functions, which apply a real activation to real and imaginary parts
- phase/amplitude functions, such as modReLU, zReLU, and cardioid
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


ArrayLike = complex | float | int | np.ndarray
Activation = Callable[..., np.ndarray]

EPSILON = 1e-12
SELU_ALPHA = 1.6732632423543772
SELU_SCALE = 1.0507009873554805


def _asarray(z: ArrayLike) -> np.ndarray:
    return np.asarray(z)


def _restore_scalar(original: ArrayLike, value: np.ndarray) -> np.ndarray | np.generic:
    if np.isscalar(original):
        return np.asarray(value).item()
    return value


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _safe_unit(z: np.ndarray) -> np.ndarray:
    magnitude = np.abs(z)
    return np.divide(z, magnitude, out=np.zeros_like(z, dtype=np.complex128), where=magnitude > EPSILON)


def apply_split(z: ArrayLike, real_activation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a real activation independently to real and imaginary components."""

    z_array = _asarray(z)
    value = real_activation(np.real(z_array)) + 1j * real_activation(np.imag(z_array))
    return _restore_scalar(z, value)


def identity(z: ArrayLike) -> np.ndarray:
    """Return the input unchanged."""

    value = _asarray(z)
    return _restore_scalar(z, value)


def complex_tanh(z: ArrayLike) -> np.ndarray:
    """Holomorphic complex hyperbolic tangent."""

    value = np.tanh(_asarray(z))
    return _restore_scalar(z, value)


def complex_sigmoid(z: ArrayLike) -> np.ndarray:
    """Holomorphic logistic sigmoid: 1 / (1 + exp(-z))."""

    z_array = _asarray(z)
    value = 1.0 / (1.0 + np.exp(-z_array))
    return _restore_scalar(z, value)


def split_relu(z: ArrayLike) -> np.ndarray:
    """CReLU/split ReLU: ReLU(real(z)) + i ReLU(imag(z))."""

    return apply_split(z, _relu)


def crelu(z: ArrayLike) -> np.ndarray:
    """Alias for split ReLU."""

    return split_relu(z)


def split_leaky_relu(z: ArrayLike, negative_slope: float = 0.01) -> np.ndarray:
    """Split leaky ReLU applied to real and imaginary parts."""

    def leaky_relu(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, x, negative_slope * x)

    return apply_split(z, leaky_relu)


def split_elu(z: ArrayLike, alpha: float = 1.0) -> np.ndarray:
    """Split ELU applied to real and imaginary parts."""

    def elu(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, x, alpha * np.expm1(x))

    return apply_split(z, elu)


def split_selu(z: ArrayLike, alpha: float = SELU_ALPHA, scale: float = SELU_SCALE) -> np.ndarray:
    """Split SELU applied to real and imaginary parts."""

    def selu(x: np.ndarray) -> np.ndarray:
        return scale * np.where(x >= 0.0, x, alpha * np.expm1(x))

    return apply_split(z, selu)


def split_sigmoid(z: ArrayLike) -> np.ndarray:
    """Split sigmoid applied to real and imaginary parts."""

    return apply_split(z, _sigmoid)


def split_tanh(z: ArrayLike) -> np.ndarray:
    """Split tanh applied to real and imaginary parts."""

    return apply_split(z, np.tanh)


def split_softplus(z: ArrayLike) -> np.ndarray:
    """Split softplus applied to real and imaginary parts."""

    return apply_split(z, _softplus)


def split_swish(z: ArrayLike, beta: float = 1.0) -> np.ndarray:
    """Split Swish: x * sigmoid(beta * x), applied component-wise."""

    def swish(x: np.ndarray) -> np.ndarray:
        return x * _sigmoid(beta * x)

    return apply_split(z, swish)


def split_gelu(z: ArrayLike) -> np.ndarray:
    """Split GELU approximation applied to real and imaginary parts."""

    return apply_split(z, _gelu)


def zrelu(z: ArrayLike) -> np.ndarray:
    """zReLU: keep values whose phase lies in the first quadrant."""

    z_array = _asarray(z)
    mask = (np.real(z_array) >= 0.0) & (np.imag(z_array) >= 0.0)
    value = np.where(mask, z_array, 0.0 + 0.0j)
    return _restore_scalar(z, value)


def modrelu(z: ArrayLike, bias: float = 0.0) -> np.ndarray:
    """modReLU: ReLU(abs(z) + bias) * z / abs(z)."""

    z_array = _asarray(z)
    value = _relu(np.abs(z_array) + bias) * _safe_unit(z_array)
    return _restore_scalar(z, value)


def cardioid(z: ArrayLike) -> np.ndarray:
    """Complex cardioid activation: 0.5 * (1 + cos(angle(z))) * z."""

    z_array = _asarray(z)
    value = 0.5 * (1.0 + np.cos(np.angle(z_array))) * z_array
    return _restore_scalar(z, value)


def phase_amplitude(z: ArrayLike, amplitude_activation: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply a real activation to magnitude while preserving phase."""

    z_array = _asarray(z)
    value = amplitude_activation(np.abs(z_array)) * _safe_unit(z_array)
    return _restore_scalar(z, value)


def amplitude_tanh(z: ArrayLike) -> np.ndarray:
    """Preserve phase and apply tanh to magnitude."""

    return phase_amplitude(z, np.tanh)


def amplitude_sigmoid(z: ArrayLike) -> np.ndarray:
    """Preserve phase and apply sigmoid to magnitude."""

    return phase_amplitude(z, _sigmoid)


ACTIVATIONS: dict[str, Activation] = {
    "identity": identity,
    "complex_tanh": complex_tanh,
    "complex_sigmoid": complex_sigmoid,
    "split_relu": split_relu,
    "crelu": crelu,
    "split_leaky_relu": split_leaky_relu,
    "split_elu": split_elu,
    "split_selu": split_selu,
    "split_sigmoid": split_sigmoid,
    "split_tanh": split_tanh,
    "split_softplus": split_softplus,
    "split_swish": split_swish,
    "split_gelu": split_gelu,
    "zrelu": zrelu,
    "modrelu": modrelu,
    "cardioid": cardioid,
    "amplitude_tanh": amplitude_tanh,
    "amplitude_sigmoid": amplitude_sigmoid,
}


def get_activation(name: str) -> Activation:
    """Return an activation function by registry name."""

    normalized = name.lower().replace("-", "_")
    try:
        return ACTIVATIONS[normalized]
    except KeyError as error:
        available = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"Unknown activation '{name}'. Available: {available}") from error
