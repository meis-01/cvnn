"""Literature-standard complex weight initializers.

The main initializer follows the polar construction used for complex-valued
neural networks: sample a phase uniformly from ``[-pi, pi]`` and a modulus from
a Rayleigh distribution, then form ``w = r * exp(i * theta)``.

For a Rayleigh scale ``sigma``, ``E[|w|^2] = 2 * sigma**2``. Choosing ``sigma``
therefore controls the complex variance directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Layout = Literal["dense", "conv", "conv_transpose"]
Criterion = Literal["glorot", "he", "lecun"]


@dataclass(frozen=True)
class FanInOut:
    """Fan-in and fan-out for an affine or convolutional weight tensor."""

    fan_in: int
    fan_out: int


def calculate_fans(shape: tuple[int, ...], layout: Layout = "dense") -> FanInOut:
    """Calculate ``fan_in`` and ``fan_out`` for project weight layouts.

    Supported layouts:

    - ``dense``: ``(out_features, in_features)``
    - ``conv``: ``(out_channels, *kernel_size, in_channels)``
    - ``conv_transpose``: ``(in_channels, *kernel_size, out_channels)``
    """

    if len(shape) < 2:
        raise ValueError("shape must have at least two dimensions")

    if layout == "dense":
        if len(shape) != 2:
            raise ValueError("dense weights must be shaped (out_features, in_features)")
        return FanInOut(fan_in=shape[1], fan_out=shape[0])

    receptive_field = int(np.prod(shape[1:-1]))
    if layout == "conv":
        return FanInOut(fan_in=shape[-1] * receptive_field, fan_out=shape[0] * receptive_field)
    if layout == "conv_transpose":
        return FanInOut(fan_in=shape[0] * receptive_field, fan_out=shape[-1] * receptive_field)
    raise ValueError("layout must be 'dense', 'conv', or 'conv_transpose'")


def _variance_from_criterion(fans: FanInOut, criterion: Criterion) -> float:
    if criterion == "glorot":
        return 2.0 / (fans.fan_in + fans.fan_out)
    if criterion == "he":
        return 2.0 / fans.fan_in
    if criterion == "lecun":
        return 1.0 / fans.fan_in
    raise ValueError("criterion must be 'glorot', 'he', or 'lecun'")


def complex_rayleigh(
    shape: tuple[int, ...],
    criterion: Criterion = "glorot",
    layout: Layout = "dense",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Initialize complex weights with Rayleigh modulus and uniform phase.

    ``criterion`` chooses the target complex variance:

    - ``glorot``: ``2 / (fan_in + fan_out)``
    - ``he``: ``2 / fan_in``
    - ``lecun``: ``1 / fan_in``
    """

    generator = np.random.default_rng() if rng is None else rng
    fans = calculate_fans(shape, layout=layout)
    target_variance = _variance_from_criterion(fans, criterion)
    rayleigh_scale = np.sqrt(target_variance / 2.0)
    modulus = generator.rayleigh(scale=rayleigh_scale, size=shape)
    phase = generator.uniform(-np.pi, np.pi, size=shape)
    return modulus * np.exp(1j * phase)


def complex_glorot(
    shape: tuple[int, ...],
    layout: Layout = "dense",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Complex Glorot/Xavier initializer for tanh/sigmoid-like maps."""

    return complex_rayleigh(shape, criterion="glorot", layout=layout, rng=rng)


def complex_he(
    shape: tuple[int, ...],
    layout: Layout = "dense",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Complex He initializer for ReLU-like split activations."""

    return complex_rayleigh(shape, criterion="he", layout=layout, rng=rng)


def complex_lecun(
    shape: tuple[int, ...],
    layout: Layout = "dense",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Complex LeCun initializer for self-normalizing or linear maps."""

    return complex_rayleigh(shape, criterion="lecun", layout=layout, rng=rng)


def zeros_complex(shape: tuple[int, ...]) -> np.ndarray:
    """Return a complex-valued zero tensor for biases."""

    return np.zeros(shape, dtype=np.complex128)


def complex_identity_gamma(channels: int) -> np.ndarray:
    """Return identity ``2x2`` affine matrices for complex normalizations.

    The normalization layers represent each complex channel as ``[real, imag]``.
    Their affine ``gamma`` therefore has shape ``(channels, 2, 2)`` and should
    start as the identity transform.
    """

    if channels <= 0:
        raise ValueError("channels must be positive")
    return np.broadcast_to(np.eye(2, dtype=np.float64), (channels, 2, 2)).copy()


def complex_normalization_affine(channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gamma, beta)`` defaults for complex normalization layers."""

    return complex_identity_gamma(channels), zeros_complex((channels,))
