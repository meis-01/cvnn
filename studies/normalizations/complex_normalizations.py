"""Complex-valued normalization layers with real/imaginary covariance.

The whitening operations treat each complex value ``z = x + i y`` as a
two-dimensional real vector ``[x, y]`` and normalize with the full 2x2
real/imaginary covariance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = complex | float | int | np.ndarray
EPSILON = 1e-5


@dataclass(frozen=True)
class ComplexBatchNormStats:
    """Batch statistics useful for inference or inspection."""

    mean: np.ndarray
    covariance: np.ndarray


def _as_complex_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.complex128)


def _canonical_axes(ndim: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(axis if axis >= 0 else ndim + axis for axis in axes)


def _inverse_square_root_2x2(covariance: np.ndarray, epsilon: float) -> np.ndarray:
    """Return ``(covariance + epsilon * I)^-1/2`` for trailing 2x2 matrices."""

    regularized = covariance.copy()
    regularized[..., 0, 0] += epsilon
    regularized[..., 1, 1] += epsilon
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    inv_sqrt = np.einsum(
        "...ij,...j,...kj->...ik",
        eigenvectors,
        1.0 / np.sqrt(np.maximum(eigenvalues, epsilon)),
        eigenvectors,
    )
    return inv_sqrt


def _covariance(centered: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    real = centered.real
    imag = centered.imag
    cov_rr = np.mean(real * real, axis=axes)
    cov_ii = np.mean(imag * imag, axis=axes)
    cov_ri = np.mean(real * imag, axis=axes)
    return np.stack(
        (
            np.stack((cov_rr, cov_ri), axis=-1),
            np.stack((cov_ri, cov_ii), axis=-1),
        ),
        axis=-2,
    )


def _whiten(centered: np.ndarray, inv_sqrt: np.ndarray) -> np.ndarray:
    vector = np.stack((centered.real, centered.imag), axis=-1)
    whitened = np.einsum("...ij,...j->...i", inv_sqrt, vector)
    return whitened[..., 0] + 1j * whitened[..., 1]


def _apply_affine(z: np.ndarray, gamma: ArrayLike | None, beta: ArrayLike | None) -> np.ndarray:
    output = z
    if gamma is not None:
        gamma_array = np.asarray(gamma, dtype=np.float64)
        vector = np.stack((output.real, output.imag), axis=-1)
        transformed = np.einsum("...ij,...j->...i", gamma_array, vector)
        output = transformed[..., 0] + 1j * transformed[..., 1]
    if beta is not None:
        output = output + _as_complex_array(beta)
    return output


def complex_batch_norm(
    x: ArrayLike,
    gamma: ArrayLike | None = None,
    beta: ArrayLike | None = None,
    running_stats: ComplexBatchNormStats | None = None,
    training: bool = True,
    epsilon: float = EPSILON,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, ComplexBatchNormStats]:
    """Complex batch normalization with per-channel 2x2 covariance whitening.

    Inputs are channels-last, for example ``(batch, features)``,
    ``(batch, length, channels)``, or ``(batch, height, width, channels)``.
    Statistics are computed over every axis except the final channel axis.

    ``gamma`` may be shaped ``(channels, 2, 2)`` and ``beta`` may be shaped
    ``(channels,)``.
    """

    x_array = _as_complex_array(x)
    reduction_axes = tuple(range(x_array.ndim - 1))
    broadcast_shape = (1,) * (x_array.ndim - 1) + (x_array.shape[-1],)

    if training:
        mean = np.mean(x_array, axis=reduction_axes)
        centered = x_array - mean.reshape(broadcast_shape)
        covariance = _covariance(centered, reduction_axes)
        stats = ComplexBatchNormStats(mean=mean, covariance=covariance)
    elif running_stats is not None:
        stats = running_stats
        centered = x_array - stats.mean.reshape(broadcast_shape)
    else:
        raise ValueError("running_stats must be provided when training=False")

    inv_sqrt = _inverse_square_root_2x2(stats.covariance, epsilon).reshape(broadcast_shape + (2, 2))
    normalized = _whiten(centered, inv_sqrt)
    gamma_broadcast = None if gamma is None else np.asarray(gamma, dtype=np.float64).reshape(broadcast_shape + (2, 2))
    beta_broadcast = None if beta is None else _as_complex_array(beta).reshape(broadcast_shape)
    output = _apply_affine(normalized, gamma_broadcast, beta_broadcast)
    if return_stats:
        return output, stats
    return output


def complex_layer_norm(
    x: ArrayLike,
    gamma: ArrayLike | None = None,
    beta: ArrayLike | None = None,
    epsilon: float = EPSILON,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Complex layer normalization with per-sample covariance whitening.

    Statistics are computed across the final feature axis for each sample. For
    sequence inputs shaped ``(batch, sequence, features)``, each token is
    normalized independently across its features.

    ``gamma`` may be shaped ``(features, 2, 2)`` and ``beta`` may be shaped
    ``(features,)``.
    """

    x_array = _as_complex_array(x)
    mean = np.mean(x_array, axis=-1, keepdims=True)
    centered = x_array - mean
    covariance = _covariance(centered, (-1,))
    inv_sqrt = _inverse_square_root_2x2(covariance, epsilon)[..., None, :, :]
    normalized = _whiten(centered, inv_sqrt)

    feature_shape = (1,) * (x_array.ndim - 1) + (x_array.shape[-1],)
    gamma_broadcast = None if gamma is None else np.asarray(gamma, dtype=np.float64).reshape(feature_shape + (2, 2))
    beta_broadcast = None if beta is None else _as_complex_array(beta).reshape(feature_shape)
    output = _apply_affine(normalized, gamma_broadcast, beta_broadcast)
    if return_stats:
        return output, (np.squeeze(mean, axis=-1), covariance)
    return output


def complex_group_norm(
    x: ArrayLike,
    num_groups: int,
    gamma: ArrayLike | None = None,
    beta: ArrayLike | None = None,
    epsilon: float = EPSILON,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Complex group normalization with covariance whitening per sample/group.

    Inputs are channels-last. Channels must be divisible by ``num_groups``.
    Statistics are computed over each group of channels and any spatial axes.

    ``gamma`` may be shaped ``(channels, 2, 2)`` and ``beta`` may be shaped
    ``(channels,)``.
    """

    if num_groups <= 0:
        raise ValueError("num_groups must be positive")

    x_array = _as_complex_array(x)
    channels = x_array.shape[-1]
    if channels % num_groups != 0:
        raise ValueError("channels must be divisible by num_groups")

    group_size = channels // num_groups
    grouped_shape = x_array.shape[:-1] + (num_groups, group_size)
    grouped = x_array.reshape(grouped_shape)
    reduction_axes = tuple(range(1, grouped.ndim - 2)) + (grouped.ndim - 1,)
    mean = np.mean(grouped, axis=reduction_axes, keepdims=True)
    centered = grouped - mean
    covariance = _covariance(centered, reduction_axes)

    inv_sqrt_shape = (x_array.shape[0],) + (1,) * (grouped.ndim - 3) + (num_groups, 1, 2, 2)
    inv_sqrt = _inverse_square_root_2x2(covariance, epsilon).reshape(inv_sqrt_shape)
    normalized = _whiten(centered, inv_sqrt).reshape(x_array.shape)

    channel_shape = (1,) * (x_array.ndim - 1) + (channels,)
    gamma_broadcast = None if gamma is None else np.asarray(gamma, dtype=np.float64).reshape(channel_shape + (2, 2))
    beta_broadcast = None if beta is None else _as_complex_array(beta).reshape(channel_shape)
    output = _apply_affine(normalized, gamma_broadcast, beta_broadcast)
    if return_stats:
        return output, (np.squeeze(mean, axis=tuple(reduction_axes)), covariance)
    return output
