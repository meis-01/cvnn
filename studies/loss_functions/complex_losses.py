"""Loss functions and output gradients for complex-valued models.

All losses are real-valued. Reconstruction losses accept complex predictions.
Binary classification losses use a real score derived from the model output and
return upstream gradients either in split R2 coordinates or Wirtinger form.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


ArrayLike = complex | float | int | np.ndarray
BinaryScoreMode = Literal["real", "magnitude"]
EPSILON = 1e-12


def sigmoid(x: ArrayLike) -> np.ndarray:
    """Numerically stable logistic sigmoid."""

    x_array = np.asarray(x, dtype=np.float64)
    return np.where(x_array >= 0.0, 1.0 / (1.0 + np.exp(-x_array)), np.exp(x_array) / (1.0 + np.exp(x_array)))


def split_r2_to_wirtinger_gradient(grad_real: ArrayLike, grad_imag: ArrayLike) -> np.ndarray:
    """Convert split R2 gradients to ``dL/dconj(z)``.

    For ``z = x + i y`` and real-valued ``L``:

        dL/dconj(z) = 0.5 * (dL/dx + i dL/dy).
    """

    return 0.5 * (np.asarray(grad_real, dtype=np.float64) + 1j * np.asarray(grad_imag, dtype=np.float64))


def reconstruction_mse_split_r2(prediction: ArrayLike, target: ArrayLike) -> tuple[float, np.ndarray, np.ndarray]:
    """Complex reconstruction MSE and split R2 upstream gradients.

    The loss is ``mean_batch(sum_features(abs(prediction - target) ** 2))``.
    Returns ``(loss, dL/dRe(prediction), dL/dIm(prediction))``.
    """

    prediction_array = np.asarray(prediction, dtype=np.complex128)
    target_array = np.asarray(target, dtype=np.complex128)
    error = prediction_array - target_array
    batch_size = prediction_array.shape[0]

    loss = float(np.sum(np.abs(error) ** 2) / batch_size)
    grad_real = 2.0 * error.real / batch_size
    grad_imag = 2.0 * error.imag / batch_size
    return loss, grad_real, grad_imag


def reconstruction_mse_wirtinger(prediction: ArrayLike, target: ArrayLike) -> tuple[float, np.ndarray]:
    """Complex reconstruction MSE and upstream ``dL/dconj(prediction)``."""

    loss, grad_real, grad_imag = reconstruction_mse_split_r2(prediction, target)
    return loss, split_r2_to_wirtinger_gradient(grad_real, grad_imag)


def binary_cross_entropy(probability: ArrayLike, target: ArrayLike) -> tuple[float, np.ndarray]:
    """Binary cross entropy from probabilities and ``dL/dprobability``.

    ``probability`` and ``target`` may be shaped ``(batch,)`` or ``(batch, 1)``.
    """

    probability_array = np.asarray(probability, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    clipped = np.clip(probability_array, EPSILON, 1.0 - EPSILON)
    batch_size = clipped.shape[0]

    loss = -float(np.sum(target_array * np.log(clipped) + (1.0 - target_array) * np.log(1.0 - clipped)) / batch_size)
    grad_probability = (clipped - target_array) / (clipped * (1.0 - clipped) * batch_size)
    return loss, grad_probability


def binary_cross_entropy_with_logits_split_r2(
    logits: ArrayLike,
    target: ArrayLike,
    score_mode: BinaryScoreMode = "real",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Binary cross entropy from logits and split R2 upstream gradients.

    Args:
        logits: Real or complex model outputs shaped ``(batch,)`` or
            ``(batch, 1)``.
        target: Binary labels with the same shape as the selected score.
        score_mode: How to turn complex logits into a real classification
            score. ``"real"`` uses ``Re(logit)`` and gives zero imaginary
            gradient. ``"magnitude"`` uses ``abs(logit)`` and backpropagates
            through magnitude.

    Returns:
        ``(loss, dL/dRe(logits), dL/dIm(logits))``.
    """

    logits_array = np.asarray(logits, dtype=np.complex128)
    target_array = np.asarray(target, dtype=np.float64)

    if score_mode == "real":
        score = logits_array.real
    elif score_mode == "magnitude":
        score = np.abs(logits_array)
    else:
        raise ValueError("score_mode must be 'real' or 'magnitude'")

    probability = sigmoid(score)
    batch_size = score.shape[0]
    loss_terms = np.maximum(score, 0.0) - score * target_array + np.log1p(np.exp(-np.abs(score)))
    loss = float(np.sum(loss_terms) / batch_size)
    grad_score = (probability - target_array) / batch_size

    if score_mode == "real":
        grad_real = grad_score
        grad_imag = np.zeros_like(grad_score)
    else:
        magnitude = np.abs(logits_array)
        grad_real = grad_score * np.divide(logits_array.real, magnitude, out=np.zeros_like(score), where=magnitude > EPSILON)
        grad_imag = grad_score * np.divide(logits_array.imag, magnitude, out=np.zeros_like(score), where=magnitude > EPSILON)

    return loss, grad_real, grad_imag


def binary_cross_entropy_with_logits_wirtinger(
    logits: ArrayLike,
    target: ArrayLike,
    score_mode: BinaryScoreMode = "real",
) -> tuple[float, np.ndarray]:
    """Binary cross entropy from logits and upstream ``dL/dconj(logits)``."""

    loss, grad_real, grad_imag = binary_cross_entropy_with_logits_split_r2(logits, target, score_mode=score_mode)
    return loss, split_r2_to_wirtinger_gradient(grad_real, grad_imag)
