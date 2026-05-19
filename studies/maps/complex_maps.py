"""Complex-valued neural network maps implemented with NumPy.

The convolution functions use deep-learning cross-correlation semantics rather
than signal-processing kernel reversal. Inputs are channels-last:

- 1D convolution: ``(batch, length, channels)``
- 2D convolution: ``(batch, height, width, channels)``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ArrayLike = complex | float | int | np.ndarray
Padding = Literal["valid", "same"]
EPSILON = 1e-12


@dataclass(frozen=True)
class TransformerParams:
    """Weights for a small complex transformer encoder block.

    All projection matrices are shaped ``(out_features, in_features)`` because
    the dense map computes ``x @ weights.T + bias``.
    """

    query_weights: np.ndarray
    key_weights: np.ndarray
    value_weights: np.ndarray
    output_weights: np.ndarray
    query_bias: np.ndarray | None = None
    key_bias: np.ndarray | None = None
    value_bias: np.ndarray | None = None
    output_bias: np.ndarray | None = None
    feedforward_in_weights: np.ndarray | None = None
    feedforward_out_weights: np.ndarray | None = None
    feedforward_in_bias: np.ndarray | None = None
    feedforward_out_bias: np.ndarray | None = None


def _as_complex_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.complex128)


def _normalize_stride(stride: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(stride, int):
        return stride, stride
    return stride


def _padding_1d(length: int, kernel_size: int, stride: int, padding: Padding) -> tuple[int, int, int]:
    if padding == "valid":
        output_length = (length - kernel_size) // stride + 1
        return 0, 0, output_length
    if padding != "same":
        raise ValueError("padding must be 'valid' or 'same'")

    output_length = int(np.ceil(length / stride))
    total_padding = max((output_length - 1) * stride + kernel_size - length, 0)
    left = total_padding // 2
    right = total_padding - left
    return left, right, output_length


def _padding_2d(
    height: int,
    width: int,
    kernel_height: int,
    kernel_width: int,
    stride: tuple[int, int],
    padding: Padding,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pad_top, pad_bottom, output_height = _padding_1d(height, kernel_height, stride[0], padding)
    pad_left, pad_right, output_width = _padding_1d(width, kernel_width, stride[1], padding)
    return (pad_top, pad_bottom), (pad_left, pad_right), (output_height, output_width)


def complex_dense(x: ArrayLike, weights: ArrayLike, bias: ArrayLike | None = None) -> np.ndarray:
    """Apply a complex dense layer ``x @ weights.T + bias``.

    ``x`` may have any leading dimensions as long as its final dimension is the
    input feature axis.
    """

    x_array = _as_complex_array(x)
    weights_array = _as_complex_array(weights)
    y = x_array @ weights_array.T
    if bias is not None:
        y = y + _as_complex_array(bias)
    return y


def complex_conv1d(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int = 1,
    padding: Padding = "valid",
) -> np.ndarray:
    """Apply a channels-last complex 1D convolution/cross-correlation.

    Args:
        x: Input shaped ``(batch, length, in_channels)``.
        kernel: Kernel shaped ``(out_channels, kernel_size, in_channels)``.
        bias: Optional bias shaped ``(out_channels,)``.
        stride: Positive integer stride.
        padding: ``"valid"`` or ``"same"``.
    """

    if stride <= 0:
        raise ValueError("stride must be positive")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    batch_size, length, in_channels = x_array.shape
    out_channels, kernel_size, kernel_channels = kernel_array.shape
    if in_channels != kernel_channels:
        raise ValueError("input channels must match kernel channels")

    pad_left, pad_right, output_length = _padding_1d(length, kernel_size, stride, padding)
    padded = np.pad(x_array, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")
    output = np.empty((batch_size, output_length, out_channels), dtype=np.complex128)

    for index in range(output_length):
        start = index * stride
        window = padded[:, start : start + kernel_size, :]
        output[:, index, :] = np.einsum("bkc,okc->bo", window, kernel_array)

    if bias is not None:
        output = output + _as_complex_array(bias)
    return output


def complex_conv_transpose1d(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """Apply a channels-last complex 1D transposed convolution.

    This follows PyTorch-style output sizing:

        ``(length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1``

    Args:
        x: Input shaped ``(batch, length, in_channels)``.
        kernel: Kernel shaped ``(in_channels, kernel_size, out_channels)``.
        bias: Optional bias shaped ``(out_channels,)``.
    """

    if stride <= 0 or dilation <= 0:
        raise ValueError("stride and dilation must be positive")
    if padding < 0 or output_padding < 0:
        raise ValueError("padding and output_padding must be non-negative")
    if output_padding >= stride:
        raise ValueError("output_padding must be smaller than stride")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    batch_size, length, in_channels = x_array.shape
    kernel_channels, kernel_size, out_channels = kernel_array.shape
    if in_channels != kernel_channels:
        raise ValueError("input channels must match kernel channels")

    output_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    if output_length <= 0:
        raise ValueError("computed output length must be positive")

    output = np.zeros((batch_size, output_length, out_channels), dtype=np.complex128)
    for index in range(length):
        base = index * stride - padding
        for kernel_index in range(kernel_size):
            output_index = base + kernel_index * dilation
            if 0 <= output_index < output_length:
                output[:, output_index, :] += x_array[:, index, :] @ kernel_array[:, kernel_index, :]

    if bias is not None:
        output = output + _as_complex_array(bias)
    return output


def complex_conv2d(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int | tuple[int, int] = 1,
    padding: Padding = "valid",
) -> np.ndarray:
    """Apply a channels-last complex 2D convolution/cross-correlation.

    Args:
        x: Input shaped ``(batch, height, width, in_channels)``.
        kernel: Kernel shaped
            ``(out_channels, kernel_height, kernel_width, in_channels)``.
        bias: Optional bias shaped ``(out_channels,)``.
        stride: Positive integer or ``(height_stride, width_stride)``.
        padding: ``"valid"`` or ``"same"``.
    """

    stride_hw = _normalize_stride(stride)
    if stride_hw[0] <= 0 or stride_hw[1] <= 0:
        raise ValueError("stride values must be positive")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    batch_size, height, width, in_channels = x_array.shape
    out_channels, kernel_height, kernel_width, kernel_channels = kernel_array.shape
    if in_channels != kernel_channels:
        raise ValueError("input channels must match kernel channels")

    pad_h, pad_w, output_hw = _padding_2d(height, width, kernel_height, kernel_width, stride_hw, padding)
    padded = np.pad(x_array, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant")
    output_height, output_width = output_hw
    output = np.empty((batch_size, output_height, output_width, out_channels), dtype=np.complex128)

    for row in range(output_height):
        row_start = row * stride_hw[0]
        for col in range(output_width):
            col_start = col * stride_hw[1]
            window = padded[:, row_start : row_start + kernel_height, col_start : col_start + kernel_width, :]
            output[:, row, col, :] = np.einsum("bhwc,ohwc->bo", window, kernel_array)

    if bias is not None:
        output = output + _as_complex_array(bias)
    return output


def complex_conv_transpose2d(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> np.ndarray:
    """Apply a channels-last complex 2D transposed convolution.

    Args:
        x: Input shaped ``(batch, height, width, in_channels)``.
        kernel: Kernel shaped
            ``(in_channels, kernel_height, kernel_width, out_channels)``.
        bias: Optional bias shaped ``(out_channels,)``.
    """

    stride_hw = _normalize_stride(stride)
    padding_hw = _normalize_stride(padding)
    output_padding_hw = _normalize_stride(output_padding)
    dilation_hw = _normalize_stride(dilation)
    if stride_hw[0] <= 0 or stride_hw[1] <= 0 or dilation_hw[0] <= 0 or dilation_hw[1] <= 0:
        raise ValueError("stride and dilation values must be positive")
    if padding_hw[0] < 0 or padding_hw[1] < 0 or output_padding_hw[0] < 0 or output_padding_hw[1] < 0:
        raise ValueError("padding and output_padding values must be non-negative")
    if output_padding_hw[0] >= stride_hw[0] or output_padding_hw[1] >= stride_hw[1]:
        raise ValueError("output_padding values must be smaller than stride values")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    batch_size, height, width, in_channels = x_array.shape
    kernel_channels, kernel_height, kernel_width, out_channels = kernel_array.shape
    if in_channels != kernel_channels:
        raise ValueError("input channels must match kernel channels")

    output_height = (
        (height - 1) * stride_hw[0]
        - 2 * padding_hw[0]
        + dilation_hw[0] * (kernel_height - 1)
        + output_padding_hw[0]
        + 1
    )
    output_width = (
        (width - 1) * stride_hw[1]
        - 2 * padding_hw[1]
        + dilation_hw[1] * (kernel_width - 1)
        + output_padding_hw[1]
        + 1
    )
    if output_height <= 0 or output_width <= 0:
        raise ValueError("computed output spatial size must be positive")

    output = np.zeros((batch_size, output_height, output_width, out_channels), dtype=np.complex128)
    for row in range(height):
        base_row = row * stride_hw[0] - padding_hw[0]
        for col in range(width):
            base_col = col * stride_hw[1] - padding_hw[1]
            for kernel_row in range(kernel_height):
                output_row = base_row + kernel_row * dilation_hw[0]
                if not 0 <= output_row < output_height:
                    continue
                for kernel_col in range(kernel_width):
                    output_col = base_col + kernel_col * dilation_hw[1]
                    if 0 <= output_col < output_width:
                        output[:, output_row, output_col, :] += x_array[:, row, col, :] @ kernel_array[:, kernel_row, kernel_col, :]

    if bias is not None:
        output = output + _as_complex_array(bias)
    return output


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, sequence_length, model_dim = x.shape
    if model_dim % num_heads != 0:
        raise ValueError("model dimension must be divisible by num_heads")
    head_dim = model_dim // num_heads
    return x.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    batch_size, num_heads, sequence_length, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, num_heads * head_dim)


def complex_multi_head_attention(
    x: ArrayLike,
    params: TransformerParams,
    num_heads: int = 1,
    attention_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply complex multi-head self-attention.

    Attention scores are real-valued ``Re(Q @ K.H) / sqrt(head_dim)``. The
    resulting real probabilities mix complex value vectors.
    """

    x_array = _as_complex_array(x)
    query = _split_heads(complex_dense(x_array, params.query_weights, params.query_bias), num_heads)
    key = _split_heads(complex_dense(x_array, params.key_weights, params.key_bias), num_heads)
    value = _split_heads(complex_dense(x_array, params.value_weights, params.value_bias), num_heads)
    head_dim = query.shape[-1]

    scores = np.einsum("bhqd,bhkd->bhqk", query, key.conj()).real / np.sqrt(head_dim)
    if attention_mask is not None:
        scores = scores + np.asarray(attention_mask, dtype=np.float64)

    attention = _softmax(scores, axis=-1)
    context = np.einsum("bhqk,bhkd->bhqd", attention, value)
    merged = _merge_heads(context)
    output = complex_dense(merged, params.output_weights, params.output_bias)
    return output, attention


def complex_layer_norm(x: ArrayLike, epsilon: float = EPSILON) -> np.ndarray:
    """Layer-normalize complex features by centering and RMS magnitude."""

    x_array = _as_complex_array(x)
    mean = np.mean(x_array, axis=-1, keepdims=True)
    centered = x_array - mean
    variance = np.mean(np.abs(centered) ** 2, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + epsilon)


def _complex_gelu(z: np.ndarray) -> np.ndarray:
    real = 0.5 * z.real * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z.real + 0.044715 * z.real**3)))
    imag = 0.5 * z.imag * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z.imag + 0.044715 * z.imag**3)))
    return real + 1j * imag


def complex_transformer_encoder(
    x: ArrayLike,
    params: TransformerParams,
    num_heads: int = 1,
    attention_mask: np.ndarray | None = None,
    use_layer_norm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a compact complex transformer encoder block.

    The block is self-attention plus residual, followed by an optional complex
    feed-forward map plus residual. Feed-forward weights are optional so the
    same params object can represent attention-only experiments.
    """

    x_array = _as_complex_array(x)
    attention_output, attention = complex_multi_head_attention(
        x_array,
        params=params,
        num_heads=num_heads,
        attention_mask=attention_mask,
    )
    hidden = x_array + attention_output
    if use_layer_norm:
        hidden = complex_layer_norm(hidden)

    if params.feedforward_in_weights is None or params.feedforward_out_weights is None:
        return hidden, attention

    feedforward = complex_dense(hidden, params.feedforward_in_weights, params.feedforward_in_bias)
    feedforward = _complex_gelu(feedforward)
    feedforward = complex_dense(feedforward, params.feedforward_out_weights, params.feedforward_out_bias)
    output = hidden + feedforward
    if use_layer_norm:
        output = complex_layer_norm(output)
    return output, attention
