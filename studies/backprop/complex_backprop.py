"""Backpropagation for the complex maps in ``studies.maps``.

The module keeps two views of complex differentiation close together:

- split R2 gradients store ``dL/dRe(z) + i dL/dIm(z)``.
- Wirtinger gradients store ``dL/dconj(z)``, the update direction for
  real-valued losses.

For real-valued losses, ``split = 2 * wirtinger``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from studies.maps import TransformerParams
from studies.maps.complex_maps import _complex_gelu, _padding_1d, _padding_2d, _softmax, complex_layer_norm


ArrayLike = complex | float | int | np.ndarray
Padding = str
EPSILON = 1e-12


@dataclass(frozen=True)
class DenseCache:
    """Values needed by dense-layer backward passes."""

    x: np.ndarray
    weights: np.ndarray
    bias: np.ndarray | None
    y: np.ndarray


@dataclass(frozen=True)
class Conv1DCache:
    """Values needed by 1D convolution backward passes."""

    x: np.ndarray
    padded_x: np.ndarray
    kernel: np.ndarray
    bias: np.ndarray | None
    stride: int
    padding: Padding
    pad_left: int
    pad_right: int
    y: np.ndarray


@dataclass(frozen=True)
class Conv2DCache:
    """Values needed by 2D convolution backward passes."""

    x: np.ndarray
    padded_x: np.ndarray
    kernel: np.ndarray
    bias: np.ndarray | None
    stride: tuple[int, int]
    padding: Padding
    pad_h: tuple[int, int]
    pad_w: tuple[int, int]
    y: np.ndarray


@dataclass(frozen=True)
class ConvTranspose1DCache:
    """Values needed by 1D transposed convolution backward passes."""

    x: np.ndarray
    kernel: np.ndarray
    bias: np.ndarray | None
    stride: int
    padding: int
    output_padding: int
    dilation: int
    y: np.ndarray


@dataclass(frozen=True)
class ConvTranspose2DCache:
    """Values needed by 2D transposed convolution backward passes."""

    x: np.ndarray
    kernel: np.ndarray
    bias: np.ndarray | None
    stride: tuple[int, int]
    padding: tuple[int, int]
    output_padding: tuple[int, int]
    dilation: tuple[int, int]
    y: np.ndarray


@dataclass(frozen=True)
class AttentionCache:
    """Values needed by complex self-attention backward passes."""

    x: np.ndarray
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    attention: np.ndarray
    context: np.ndarray
    merged: np.ndarray
    output: np.ndarray
    params: TransformerParams
    num_heads: int


@dataclass(frozen=True)
class TransformerCache:
    """Values needed by the supported transformer encoder backward pass."""

    x: np.ndarray
    attention_cache: AttentionCache
    hidden: np.ndarray
    normalized_hidden: np.ndarray
    feedforward_in_cache: DenseCache | None
    feedforward_activation: np.ndarray | None
    feedforward_out_cache: DenseCache | None
    final_norm_input: np.ndarray
    output: np.ndarray
    use_layer_norm: bool


def _as_complex_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.complex128)


def _split_to_wirtinger(grad_split: np.ndarray) -> np.ndarray:
    return 0.5 * _as_complex_array(grad_split)


def _wirtinger_to_split(grad_conj: ArrayLike) -> np.ndarray:
    return 2.0 * _as_complex_array(grad_conj)


def _split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, sequence_length, model_dim = x.shape
    if model_dim % num_heads != 0:
        raise ValueError("model dimension must be divisible by num_heads")
    head_dim = model_dim // num_heads
    return x.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    batch_size, num_heads, sequence_length, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, num_heads * head_dim)


def _unmerge_heads_gradient(grad_merged: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, sequence_length, model_dim = grad_merged.shape
    head_dim = model_dim // num_heads
    return grad_merged.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(0, 2, 1, 3)


def _sum_bias_gradient(grad_y: np.ndarray) -> np.ndarray:
    axes = tuple(range(grad_y.ndim - 1))
    return np.sum(grad_y, axis=axes)


def _complex_gelu_derivative(z: np.ndarray) -> np.ndarray:
    def real_gelu_derivative(x: np.ndarray) -> np.ndarray:
        inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        inner_derivative = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
        return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * inner_derivative

    return real_gelu_derivative(z.real) + 1j * real_gelu_derivative(z.imag)


def complex_layer_norm_backward_split_r2(grad_y: ArrayLike, x: ArrayLike, epsilon: float = EPSILON) -> np.ndarray:
    """Backprop through ``complex_layer_norm`` in split R2 form."""

    grad = _as_complex_array(grad_y)
    x_array = _as_complex_array(x)
    feature_count = x_array.shape[-1]
    mean = np.mean(x_array, axis=-1, keepdims=True)
    centered = x_array - mean
    variance = np.mean(np.abs(centered) ** 2, axis=-1, keepdims=True)
    sigma = np.sqrt(variance + epsilon)

    projection = np.sum((grad * centered.conj()).real, axis=-1, keepdims=True)
    grad_centered = grad / sigma - centered * projection / (feature_count * sigma**3)
    return grad_centered - np.mean(grad_centered, axis=-1, keepdims=True)


def dense_forward(x: ArrayLike, weights: ArrayLike, bias: ArrayLike | None = None) -> tuple[np.ndarray, DenseCache]:
    """Run ``y = x @ weights.T + bias`` and keep a cache."""

    x_array = _as_complex_array(x)
    weights_array = _as_complex_array(weights)
    bias_array = None if bias is None else _as_complex_array(bias)

    y = x_array @ weights_array.T
    if bias_array is not None:
        y = y + bias_array
    return y, DenseCache(x=x_array, weights=weights_array, bias=bias_array, y=y)


def dense_backward_split_r2(
    grad_y_real: ArrayLike,
    grad_y_imag: ArrayLike,
    cache: DenseCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through dense by expanding complex multiplication into R2."""

    grad_u = np.asarray(grad_y_real, dtype=np.float64)
    grad_v = np.asarray(grad_y_imag, dtype=np.float64)
    grad_shape = grad_u.shape

    grad_u_2d = grad_u.reshape(-1, grad_shape[-1])
    grad_v_2d = grad_v.reshape(-1, grad_shape[-1])
    x_2d = cache.x.reshape(-1, cache.x.shape[-1])

    a = x_2d.real
    b = x_2d.imag
    weights_real = cache.weights.real
    weights_imag = cache.weights.imag

    grad_a = grad_u_2d @ weights_real + grad_v_2d @ weights_imag
    grad_b = -grad_u_2d @ weights_imag + grad_v_2d @ weights_real

    grad_weights_real = grad_u_2d.T @ a + grad_v_2d.T @ b
    grad_weights_imag = -grad_u_2d.T @ b + grad_v_2d.T @ a

    grad_x = (grad_a + 1j * grad_b).reshape(cache.x.shape)
    grad_weights = grad_weights_real + 1j * grad_weights_imag
    grad_bias = None if cache.bias is None else _sum_bias_gradient(grad_u + 1j * grad_v)
    return grad_x, grad_weights, grad_bias


def dense_backward_wirtinger(
    grad_y_conj: ArrayLike,
    cache: DenseCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through dense with Wirtinger gradients."""

    grad_split = _wirtinger_to_split(grad_y_conj)
    grad_x, grad_weights, grad_bias = dense_backward_split_r2(grad_split.real, grad_split.imag, cache)
    return _split_to_wirtinger(grad_x), _split_to_wirtinger(grad_weights), None if grad_bias is None else _split_to_wirtinger(grad_bias)


def complex_conv1d_forward(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int = 1,
    padding: Padding = "valid",
) -> tuple[np.ndarray, Conv1DCache]:
    """Run channels-last complex 1D cross-correlation and keep a cache."""

    if stride <= 0:
        raise ValueError("stride must be positive")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    bias_array = None if bias is None else _as_complex_array(bias)
    _, length, in_channels = x_array.shape
    out_channels, kernel_size, kernel_channels = kernel_array.shape
    if in_channels != kernel_channels:
        raise ValueError("input channels must match kernel channels")

    pad_left, pad_right, output_length = _padding_1d(length, kernel_size, stride, padding)  # type: ignore[arg-type]
    padded = np.pad(x_array, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")
    output = np.empty((x_array.shape[0], output_length, out_channels), dtype=np.complex128)
    for index in range(output_length):
        start = index * stride
        window = padded[:, start : start + kernel_size, :]
        output[:, index, :] = np.einsum("bkc,okc->bo", window, kernel_array)
    if bias_array is not None:
        output = output + bias_array
    return output, Conv1DCache(x_array, padded, kernel_array, bias_array, stride, padding, pad_left, pad_right, output)


def conv1d_backward_split_r2(
    grad_y_real: ArrayLike,
    grad_y_imag: ArrayLike,
    cache: Conv1DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 1D cross-correlation in split R2 form."""

    grad = np.asarray(grad_y_real, dtype=np.float64) + 1j * np.asarray(grad_y_imag, dtype=np.float64)
    grad_padded = np.zeros_like(cache.padded_x)
    grad_kernel = np.zeros_like(cache.kernel)

    for index in range(cache.y.shape[1]):
        start = index * cache.stride
        window = cache.padded_x[:, start : start + cache.kernel.shape[1], :]
        for out_channel in range(cache.kernel.shape[0]):
            grad_out = grad[:, index, out_channel]
            grad_padded[:, start : start + cache.kernel.shape[1], :] += grad_out[:, None, None] * cache.kernel[out_channel].conj()
            grad_kernel[out_channel] += np.sum(grad_out[:, None, None] * window.conj(), axis=0)

    end = grad_padded.shape[1] - cache.pad_right if cache.pad_right else grad_padded.shape[1]
    grad_x = grad_padded[:, cache.pad_left:end, :]
    grad_bias = None if cache.bias is None else np.sum(grad, axis=(0, 1))
    return grad_x, grad_kernel, grad_bias


def conv1d_backward_wirtinger(
    grad_y_conj: ArrayLike,
    cache: Conv1DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 1D cross-correlation with Wirtinger gradients."""

    grad_split = _wirtinger_to_split(grad_y_conj)
    grad_x, grad_kernel, grad_bias = conv1d_backward_split_r2(grad_split.real, grad_split.imag, cache)
    return _split_to_wirtinger(grad_x), _split_to_wirtinger(grad_kernel), None if grad_bias is None else _split_to_wirtinger(grad_bias)


def complex_conv2d_forward(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int | tuple[int, int] = 1,
    padding: Padding = "valid",
) -> tuple[np.ndarray, Conv2DCache]:
    """Run channels-last complex 2D cross-correlation and keep a cache."""

    stride_hw = (stride, stride) if isinstance(stride, int) else stride
    if stride_hw[0] <= 0 or stride_hw[1] <= 0:
        raise ValueError("stride values must be positive")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    bias_array = None if bias is None else _as_complex_array(bias)
    _, height, width, in_channels = x_array.shape
    out_channels, kernel_height, kernel_width, kernel_channels = kernel_array.shape
    if in_channels != kernel_channels:
        raise ValueError("input channels must match kernel channels")

    pad_h, pad_w, output_hw = _padding_2d(height, width, kernel_height, kernel_width, stride_hw, padding)  # type: ignore[arg-type]
    padded = np.pad(x_array, ((0, 0), pad_h, pad_w, (0, 0)), mode="constant")
    output = np.empty((x_array.shape[0], output_hw[0], output_hw[1], out_channels), dtype=np.complex128)
    for row in range(output_hw[0]):
        row_start = row * stride_hw[0]
        for col in range(output_hw[1]):
            col_start = col * stride_hw[1]
            window = padded[:, row_start : row_start + kernel_height, col_start : col_start + kernel_width, :]
            output[:, row, col, :] = np.einsum("bhwc,ohwc->bo", window, kernel_array)
    if bias_array is not None:
        output = output + bias_array
    return output, Conv2DCache(x_array, padded, kernel_array, bias_array, stride_hw, padding, pad_h, pad_w, output)


def conv2d_backward_split_r2(
    grad_y_real: ArrayLike,
    grad_y_imag: ArrayLike,
    cache: Conv2DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 2D cross-correlation in split R2 form."""

    grad = np.asarray(grad_y_real, dtype=np.float64) + 1j * np.asarray(grad_y_imag, dtype=np.float64)
    grad_padded = np.zeros_like(cache.padded_x)
    grad_kernel = np.zeros_like(cache.kernel)
    kernel_height, kernel_width = cache.kernel.shape[1:3]

    for row in range(cache.y.shape[1]):
        row_start = row * cache.stride[0]
        for col in range(cache.y.shape[2]):
            col_start = col * cache.stride[1]
            window = cache.padded_x[:, row_start : row_start + kernel_height, col_start : col_start + kernel_width, :]
            for out_channel in range(cache.kernel.shape[0]):
                grad_out = grad[:, row, col, out_channel]
                grad_padded[:, row_start : row_start + kernel_height, col_start : col_start + kernel_width, :] += (
                    grad_out[:, None, None, None] * cache.kernel[out_channel].conj()
                )
                grad_kernel[out_channel] += np.sum(grad_out[:, None, None, None] * window.conj(), axis=0)

    row_end = grad_padded.shape[1] - cache.pad_h[1] if cache.pad_h[1] else grad_padded.shape[1]
    col_end = grad_padded.shape[2] - cache.pad_w[1] if cache.pad_w[1] else grad_padded.shape[2]
    grad_x = grad_padded[:, cache.pad_h[0]:row_end, cache.pad_w[0]:col_end, :]
    grad_bias = None if cache.bias is None else np.sum(grad, axis=(0, 1, 2))
    return grad_x, grad_kernel, grad_bias


def conv2d_backward_wirtinger(
    grad_y_conj: ArrayLike,
    cache: Conv2DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 2D cross-correlation with Wirtinger gradients."""

    grad_split = _wirtinger_to_split(grad_y_conj)
    grad_x, grad_kernel, grad_bias = conv2d_backward_split_r2(grad_split.real, grad_split.imag, cache)
    return _split_to_wirtinger(grad_x), _split_to_wirtinger(grad_kernel), None if grad_bias is None else _split_to_wirtinger(grad_bias)


def complex_conv_transpose1d_forward(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
) -> tuple[np.ndarray, ConvTranspose1DCache]:
    """Run channels-last complex 1D transposed convolution and keep a cache."""

    if stride <= 0 or dilation <= 0:
        raise ValueError("stride and dilation must be positive")
    if padding < 0 or output_padding < 0:
        raise ValueError("padding and output_padding must be non-negative")
    if output_padding >= stride:
        raise ValueError("output_padding must be smaller than stride")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    bias_array = None if bias is None else _as_complex_array(bias)
    _, length, in_channels = x_array.shape
    kernel_channels, kernel_size, out_channels = kernel_array.shape
    if in_channels != kernel_channels:
        raise ValueError("input channels must match kernel channels")

    output_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    if output_length <= 0:
        raise ValueError("computed output length must be positive")

    output = np.zeros((x_array.shape[0], output_length, out_channels), dtype=np.complex128)
    for index in range(length):
        base = index * stride - padding
        for kernel_index in range(kernel_size):
            output_index = base + kernel_index * dilation
            if 0 <= output_index < output_length:
                output[:, output_index, :] += x_array[:, index, :] @ kernel_array[:, kernel_index, :]
    if bias_array is not None:
        output = output + bias_array
    return output, ConvTranspose1DCache(x_array, kernel_array, bias_array, stride, padding, output_padding, dilation, output)


def conv_transpose1d_backward_split_r2(
    grad_y_real: ArrayLike,
    grad_y_imag: ArrayLike,
    cache: ConvTranspose1DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 1D transposed convolution in split R2 form."""

    grad = np.asarray(grad_y_real, dtype=np.float64) + 1j * np.asarray(grad_y_imag, dtype=np.float64)
    grad_x = np.zeros_like(cache.x)
    grad_kernel = np.zeros_like(cache.kernel)
    kernel_size = cache.kernel.shape[1]

    for index in range(cache.x.shape[1]):
        base = index * cache.stride - cache.padding
        for kernel_index in range(kernel_size):
            output_index = base + kernel_index * cache.dilation
            if 0 <= output_index < cache.y.shape[1]:
                grad_out = grad[:, output_index, :]
                grad_x[:, index, :] += grad_out @ cache.kernel[:, kernel_index, :].conj().T
                grad_kernel[:, kernel_index, :] += cache.x[:, index, :].conj().T @ grad_out

    grad_bias = None if cache.bias is None else np.sum(grad, axis=(0, 1))
    return grad_x, grad_kernel, grad_bias


def conv_transpose1d_backward_wirtinger(
    grad_y_conj: ArrayLike,
    cache: ConvTranspose1DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 1D transposed convolution with Wirtinger gradients."""

    grad_split = _wirtinger_to_split(grad_y_conj)
    grad_x, grad_kernel, grad_bias = conv_transpose1d_backward_split_r2(grad_split.real, grad_split.imag, cache)
    return _split_to_wirtinger(grad_x), _split_to_wirtinger(grad_kernel), None if grad_bias is None else _split_to_wirtinger(grad_bias)


def complex_conv_transpose2d_forward(
    x: ArrayLike,
    kernel: ArrayLike,
    bias: ArrayLike | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[np.ndarray, ConvTranspose2DCache]:
    """Run channels-last complex 2D transposed convolution and keep a cache."""

    stride_hw = (stride, stride) if isinstance(stride, int) else stride
    padding_hw = (padding, padding) if isinstance(padding, int) else padding
    output_padding_hw = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
    dilation_hw = (dilation, dilation) if isinstance(dilation, int) else dilation
    if stride_hw[0] <= 0 or stride_hw[1] <= 0 or dilation_hw[0] <= 0 or dilation_hw[1] <= 0:
        raise ValueError("stride and dilation values must be positive")
    if padding_hw[0] < 0 or padding_hw[1] < 0 or output_padding_hw[0] < 0 or output_padding_hw[1] < 0:
        raise ValueError("padding and output_padding values must be non-negative")
    if output_padding_hw[0] >= stride_hw[0] or output_padding_hw[1] >= stride_hw[1]:
        raise ValueError("output_padding values must be smaller than stride values")

    x_array = _as_complex_array(x)
    kernel_array = _as_complex_array(kernel)
    bias_array = None if bias is None else _as_complex_array(bias)
    _, height, width, in_channels = x_array.shape
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

    output = np.zeros((x_array.shape[0], output_height, output_width, out_channels), dtype=np.complex128)
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
    if bias_array is not None:
        output = output + bias_array
    return output, ConvTranspose2DCache(x_array, kernel_array, bias_array, stride_hw, padding_hw, output_padding_hw, dilation_hw, output)


def conv_transpose2d_backward_split_r2(
    grad_y_real: ArrayLike,
    grad_y_imag: ArrayLike,
    cache: ConvTranspose2DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 2D transposed convolution in split R2 form."""

    grad = np.asarray(grad_y_real, dtype=np.float64) + 1j * np.asarray(grad_y_imag, dtype=np.float64)
    grad_x = np.zeros_like(cache.x)
    grad_kernel = np.zeros_like(cache.kernel)
    kernel_height, kernel_width = cache.kernel.shape[1:3]

    for row in range(cache.x.shape[1]):
        base_row = row * cache.stride[0] - cache.padding[0]
        for col in range(cache.x.shape[2]):
            base_col = col * cache.stride[1] - cache.padding[1]
            for kernel_row in range(kernel_height):
                output_row = base_row + kernel_row * cache.dilation[0]
                if not 0 <= output_row < cache.y.shape[1]:
                    continue
                for kernel_col in range(kernel_width):
                    output_col = base_col + kernel_col * cache.dilation[1]
                    if 0 <= output_col < cache.y.shape[2]:
                        grad_out = grad[:, output_row, output_col, :]
                        grad_x[:, row, col, :] += grad_out @ cache.kernel[:, kernel_row, kernel_col, :].conj().T
                        grad_kernel[:, kernel_row, kernel_col, :] += cache.x[:, row, col, :].conj().T @ grad_out

    grad_bias = None if cache.bias is None else np.sum(grad, axis=(0, 1, 2))
    return grad_x, grad_kernel, grad_bias


def conv_transpose2d_backward_wirtinger(
    grad_y_conj: ArrayLike,
    cache: ConvTranspose2DCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Backprop through complex 2D transposed convolution with Wirtinger gradients."""

    grad_split = _wirtinger_to_split(grad_y_conj)
    grad_x, grad_kernel, grad_bias = conv_transpose2d_backward_split_r2(grad_split.real, grad_split.imag, cache)
    return _split_to_wirtinger(grad_x), _split_to_wirtinger(grad_kernel), None if grad_bias is None else _split_to_wirtinger(grad_bias)


def complex_multi_head_attention_forward(
    x: ArrayLike,
    params: TransformerParams,
    num_heads: int = 1,
    attention_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, AttentionCache]:
    """Run complex multi-head self-attention and keep a cache."""

    x_array = _as_complex_array(x)
    query_flat, _ = dense_forward(x_array, params.query_weights, params.query_bias)
    key_flat, _ = dense_forward(x_array, params.key_weights, params.key_bias)
    value_flat, _ = dense_forward(x_array, params.value_weights, params.value_bias)
    query = _split_heads(query_flat, num_heads)
    key = _split_heads(key_flat, num_heads)
    value = _split_heads(value_flat, num_heads)
    scores = np.einsum("bhqd,bhkd->bhqk", query, key.conj()).real / np.sqrt(query.shape[-1])
    if attention_mask is not None:
        scores = scores + np.asarray(attention_mask, dtype=np.float64)
    attention = _softmax(scores, axis=-1)
    context = np.einsum("bhqk,bhkd->bhqd", attention, value)
    merged = _merge_heads(context)
    output, _ = dense_forward(merged, params.output_weights, params.output_bias)
    return output, AttentionCache(x_array, query, key, value, attention, context, merged, output, params, num_heads)


def multi_head_attention_backward_wirtinger(
    grad_output_conj: ArrayLike,
    cache: AttentionCache,
) -> tuple[np.ndarray, TransformerParams]:
    """Backprop through complex self-attention with Wirtinger gradients."""

    grad_output_split = _wirtinger_to_split(grad_output_conj)
    output_dense_cache = DenseCache(cache.merged, cache.params.output_weights, cache.params.output_bias, cache.output)
    grad_merged_split, grad_output_weights_split, grad_output_bias_split = dense_backward_split_r2(
        grad_output_split.real,
        grad_output_split.imag,
        output_dense_cache,
    )

    grad_context = _unmerge_heads_gradient(grad_merged_split, cache.num_heads)
    grad_attention = np.einsum("bhqd,bhkd->bhqk", grad_context, cache.value.conj()).real
    grad_value = np.einsum("bhqk,bhqd->bhkd", cache.attention, grad_context)

    dot = np.sum(grad_attention * cache.attention, axis=-1, keepdims=True)
    grad_scores = cache.attention * (grad_attention - dot)
    scale = 1.0 / np.sqrt(cache.query.shape[-1])
    grad_query = scale * np.einsum("bhqk,bhkd->bhqd", grad_scores, cache.key)
    grad_key = scale * np.einsum("bhqk,bhqd->bhkd", grad_scores, cache.query)

    grad_query_flat = _merge_heads(grad_query)
    grad_key_flat = _merge_heads(grad_key)
    grad_value_flat = _merge_heads(grad_value)

    q_cache = DenseCache(cache.x, cache.params.query_weights, cache.params.query_bias, _merge_heads(cache.query))
    k_cache = DenseCache(cache.x, cache.params.key_weights, cache.params.key_bias, _merge_heads(cache.key))
    v_cache = DenseCache(cache.x, cache.params.value_weights, cache.params.value_bias, _merge_heads(cache.value))

    grad_x_q, grad_q_weights, grad_q_bias = dense_backward_split_r2(grad_query_flat.real, grad_query_flat.imag, q_cache)
    grad_x_k, grad_k_weights, grad_k_bias = dense_backward_split_r2(grad_key_flat.real, grad_key_flat.imag, k_cache)
    grad_x_v, grad_v_weights, grad_v_bias = dense_backward_split_r2(grad_value_flat.real, grad_value_flat.imag, v_cache)

    grad_x_split = grad_x_q + grad_x_k + grad_x_v
    grad_params = TransformerParams(
        query_weights=_split_to_wirtinger(grad_q_weights),
        key_weights=_split_to_wirtinger(grad_k_weights),
        value_weights=_split_to_wirtinger(grad_v_weights),
        output_weights=_split_to_wirtinger(grad_output_weights_split),
        query_bias=None if grad_q_bias is None else _split_to_wirtinger(grad_q_bias),
        key_bias=None if grad_k_bias is None else _split_to_wirtinger(grad_k_bias),
        value_bias=None if grad_v_bias is None else _split_to_wirtinger(grad_v_bias),
        output_bias=None if grad_output_bias_split is None else _split_to_wirtinger(grad_output_bias_split),
    )
    return _split_to_wirtinger(grad_x_split), grad_params


def complex_transformer_encoder_forward(
    x: ArrayLike,
    params: TransformerParams,
    num_heads: int = 1,
    attention_mask: np.ndarray | None = None,
    use_layer_norm: bool = False,
) -> tuple[np.ndarray, TransformerCache]:
    """Run a complex transformer encoder path and keep a cache."""

    attention_output, attention_cache = complex_multi_head_attention_forward(x, params, num_heads, attention_mask)
    x_array = _as_complex_array(x)
    hidden = x_array + attention_output
    normalized_hidden = complex_layer_norm(hidden) if use_layer_norm else hidden
    feedforward_in_cache = None
    feedforward_activation = None
    feedforward_out_cache = None
    output = normalized_hidden

    if params.feedforward_in_weights is not None and params.feedforward_out_weights is not None:
        feedforward_in, feedforward_in_cache = dense_forward(
            normalized_hidden,
            params.feedforward_in_weights,
            params.feedforward_in_bias,
        )
        feedforward_activation = _complex_gelu(feedforward_in)
        feedforward_out, feedforward_out_cache = dense_forward(
            feedforward_activation,
            params.feedforward_out_weights,
            params.feedforward_out_bias,
        )
        output = normalized_hidden + feedforward_out

    final_norm_input = output
    if use_layer_norm:
        output = complex_layer_norm(final_norm_input)
    return output, TransformerCache(
        x=x_array,
        attention_cache=attention_cache,
        hidden=hidden,
        normalized_hidden=normalized_hidden,
        feedforward_in_cache=feedforward_in_cache,
        feedforward_activation=feedforward_activation,
        feedforward_out_cache=feedforward_out_cache,
        final_norm_input=final_norm_input,
        output=output,
        use_layer_norm=use_layer_norm,
    )


def transformer_encoder_backward_wirtinger(
    grad_output_conj: ArrayLike,
    cache: TransformerCache,
) -> tuple[np.ndarray, TransformerParams]:
    """Backprop through the complex transformer encoder with Wirtinger gradients."""

    grad_output_split = _wirtinger_to_split(grad_output_conj)
    if cache.use_layer_norm:
        grad_output_split = complex_layer_norm_backward_split_r2(grad_output_split, cache.final_norm_input)

    grad_hidden_split = grad_output_split
    feedforward_in_weights = None
    feedforward_out_weights = None
    feedforward_in_bias = None
    feedforward_out_bias = None

    if cache.feedforward_in_cache is not None and cache.feedforward_out_cache is not None:
        grad_residual_split = grad_output_split
        grad_ff_activation, grad_ff_out_weights, grad_ff_out_bias = dense_backward_split_r2(
            grad_output_split.real,
            grad_output_split.imag,
            cache.feedforward_out_cache,
        )
        gelu_derivative = _complex_gelu_derivative(cache.feedforward_in_cache.y)
        grad_ff_in = grad_ff_activation.real * gelu_derivative.real + 1j * grad_ff_activation.imag * gelu_derivative.imag
        grad_ff_input, grad_ff_in_weights, grad_ff_in_bias = dense_backward_split_r2(
            grad_ff_in.real,
            grad_ff_in.imag,
            cache.feedforward_in_cache,
        )
        grad_hidden_split = grad_residual_split + grad_ff_input
        feedforward_in_weights = _split_to_wirtinger(grad_ff_in_weights)
        feedforward_out_weights = _split_to_wirtinger(grad_ff_out_weights)
        feedforward_in_bias = None if grad_ff_in_bias is None else _split_to_wirtinger(grad_ff_in_bias)
        feedforward_out_bias = None if grad_ff_out_bias is None else _split_to_wirtinger(grad_ff_out_bias)

    if cache.use_layer_norm:
        grad_hidden_split = complex_layer_norm_backward_split_r2(grad_hidden_split, cache.hidden)

    grad_attention_x, attention_grad_params = multi_head_attention_backward_wirtinger(
        _split_to_wirtinger(grad_hidden_split),
        cache.attention_cache,
    )
    grad_x = _split_to_wirtinger(grad_hidden_split) + grad_attention_x
    grad_params = TransformerParams(
        query_weights=attention_grad_params.query_weights,
        key_weights=attention_grad_params.key_weights,
        value_weights=attention_grad_params.value_weights,
        output_weights=attention_grad_params.output_weights,
        query_bias=attention_grad_params.query_bias,
        key_bias=attention_grad_params.key_bias,
        value_bias=attention_grad_params.value_bias,
        output_bias=attention_grad_params.output_bias,
        feedforward_in_weights=feedforward_in_weights,
        feedforward_out_weights=feedforward_out_weights,
        feedforward_in_bias=feedforward_in_bias,
        feedforward_out_bias=feedforward_out_bias,
    )
    return grad_x, grad_params
