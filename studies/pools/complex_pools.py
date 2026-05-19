"""Complex-valued pooling layers implemented with NumPy.

Inputs are channels-last:

- 1D pooling: ``(batch, length, channels)``
- 2D pooling: ``(batch, height, width, channels)``

Average pooling averages complex values directly. Max pooling follows the
common complex-valued convention of selecting the value with largest magnitude
and returning that original complex value.
"""

from __future__ import annotations

import math

import numpy as np


ArrayLike = complex | float | int | np.ndarray


def _as_complex_array(value: ArrayLike) -> np.ndarray:
    return np.asarray(value, dtype=np.complex128)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return value


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _pool_output_size(input_size: int, kernel_size: int, stride: int, padding: int, ceil_mode: bool) -> int:
    numerator = input_size + 2 * padding - kernel_size
    if ceil_mode:
        return max(math.floor((numerator + stride - 1) / stride) + 1, 0)
    return max(math.floor(numerator / stride) + 1, 0)


def _adaptive_window(index: int, input_size: int, output_size: int) -> tuple[int, int]:
    start = math.floor(index * input_size / output_size)
    end = math.ceil((index + 1) * input_size / output_size)
    return start, end


def _pad_1d_for_windows(
    padded: np.ndarray,
    valid: np.ndarray,
    output_length: int,
    kernel_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    needed = (output_length - 1) * stride + kernel_size
    extra = max(needed - padded.shape[1], 0)
    if extra:
        padded = np.pad(padded, ((0, 0), (0, extra), (0, 0)), mode="constant")
        valid = np.pad(valid, ((0, 0), (0, extra), (0, 0)), mode="constant")
    return padded, valid


def _pad_2d_for_windows(
    padded: np.ndarray,
    valid: np.ndarray,
    output_h: int,
    output_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    needed_h = (output_h - 1) * stride_h + kernel_h
    needed_w = (output_w - 1) * stride_w + kernel_w
    extra_h = max(needed_h - padded.shape[1], 0)
    extra_w = max(needed_w - padded.shape[2], 0)
    if extra_h or extra_w:
        padded = np.pad(padded, ((0, 0), (0, extra_h), (0, extra_w), (0, 0)), mode="constant")
        valid = np.pad(valid, ((0, 0), (0, extra_h), (0, extra_w), (0, 0)), mode="constant")
    return padded, valid


def complex_avg_pool1d(
    x: ArrayLike,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> np.ndarray:
    """Complex counterpart of PyTorch-style ``AvgPool1d``.

    The average is the arithmetic mean of complex values in each window.
    """

    _positive("kernel_size", kernel_size)
    stride = kernel_size if stride is None else stride
    _positive("stride", stride)
    if padding < 0:
        raise ValueError("padding must be non-negative")

    x_array = _as_complex_array(x)
    batch_size, length, channels = x_array.shape
    output_length = _pool_output_size(length, kernel_size, stride, padding, ceil_mode)
    padded = np.pad(x_array, ((0, 0), (padding, padding), (0, 0)), mode="constant")
    valid = np.pad(np.ones((batch_size, length, channels), dtype=bool), ((0, 0), (padding, padding), (0, 0)), mode="constant")
    padded, valid = _pad_1d_for_windows(padded, valid, output_length, kernel_size, stride)
    output = np.empty((batch_size, output_length, channels), dtype=np.complex128)

    for index in range(output_length):
        start = index * stride
        window = padded[:, start : start + kernel_size, :]
        valid_window = valid[:, start : start + kernel_size, :]
        total = np.sum(window, axis=1)
        if count_include_pad:
            divisor = kernel_size
        else:
            divisor = np.maximum(np.sum(valid_window, axis=1), 1)
        output[:, index, :] = total / divisor
    return output


def complex_max_pool1d(
    x: ArrayLike,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Complex counterpart of PyTorch-style ``MaxPool1d``.

    Max selection is by ``abs(z)``; the returned value keeps its complex phase.
    """

    _positive("kernel_size", kernel_size)
    stride = kernel_size if stride is None else stride
    _positive("stride", stride)
    if padding < 0:
        raise ValueError("padding must be non-negative")

    x_array = _as_complex_array(x)
    batch_size, length, channels = x_array.shape
    output_length = _pool_output_size(length, kernel_size, stride, padding, ceil_mode)
    padded = np.pad(x_array, ((0, 0), (padding, padding), (0, 0)), mode="constant")
    valid = np.pad(np.ones((batch_size, length, channels), dtype=bool), ((0, 0), (padding, padding), (0, 0)), mode="constant")
    padded, valid = _pad_1d_for_windows(padded, valid, output_length, kernel_size, stride)
    output = np.empty((batch_size, output_length, channels), dtype=np.complex128)
    indices = np.empty((batch_size, output_length, channels), dtype=np.int64)

    for index in range(output_length):
        start = index * stride
        window = padded[:, start : start + kernel_size, :]
        valid_window = valid[:, start : start + kernel_size, :]
        scores = np.where(valid_window, np.abs(window), -np.inf)
        local_indices = np.argmax(scores, axis=1)
        batch_indices = np.arange(batch_size)[:, None]
        channel_indices = np.arange(channels)[None, :]
        output[:, index, :] = window[batch_indices, local_indices, channel_indices]
        indices[:, index, :] = local_indices + start - padding

    if return_indices:
        return output, indices
    return output


def complex_avg_pool2d(
    x: ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> np.ndarray:
    """Complex counterpart of PyTorch-style ``AvgPool2d``."""

    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(kernel_size if stride is None else stride)
    pad_h, pad_w = _pair(padding)
    for name, value in (("kernel_h", kernel_h), ("kernel_w", kernel_w), ("stride_h", stride_h), ("stride_w", stride_w)):
        _positive(name, value)
    if pad_h < 0 or pad_w < 0:
        raise ValueError("padding must be non-negative")

    x_array = _as_complex_array(x)
    batch_size, height, width, channels = x_array.shape
    output_h = _pool_output_size(height, kernel_h, stride_h, pad_h, ceil_mode)
    output_w = _pool_output_size(width, kernel_w, stride_w, pad_w, ceil_mode)
    padded = np.pad(x_array, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    valid = np.pad(np.ones((batch_size, height, width, channels), dtype=bool), ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    padded, valid = _pad_2d_for_windows(padded, valid, output_h, output_w, kernel_h, kernel_w, stride_h, stride_w)
    output = np.empty((batch_size, output_h, output_w, channels), dtype=np.complex128)

    for row in range(output_h):
        row_start = row * stride_h
        for col in range(output_w):
            col_start = col * stride_w
            window = padded[:, row_start : row_start + kernel_h, col_start : col_start + kernel_w, :]
            valid_window = valid[:, row_start : row_start + kernel_h, col_start : col_start + kernel_w, :]
            total = np.sum(window, axis=(1, 2))
            if count_include_pad:
                divisor = kernel_h * kernel_w
            else:
                divisor = np.maximum(np.sum(valid_window, axis=(1, 2)), 1)
            output[:, row, col, :] = total / divisor
    return output


def complex_max_pool2d(
    x: ArrayLike,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Complex counterpart of PyTorch-style ``MaxPool2d``."""

    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(kernel_size if stride is None else stride)
    pad_h, pad_w = _pair(padding)
    for name, value in (("kernel_h", kernel_h), ("kernel_w", kernel_w), ("stride_h", stride_h), ("stride_w", stride_w)):
        _positive(name, value)
    if pad_h < 0 or pad_w < 0:
        raise ValueError("padding must be non-negative")

    x_array = _as_complex_array(x)
    batch_size, height, width, channels = x_array.shape
    output_h = _pool_output_size(height, kernel_h, stride_h, pad_h, ceil_mode)
    output_w = _pool_output_size(width, kernel_w, stride_w, pad_w, ceil_mode)
    padded = np.pad(x_array, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    valid = np.pad(np.ones((batch_size, height, width, channels), dtype=bool), ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    padded, valid = _pad_2d_for_windows(padded, valid, output_h, output_w, kernel_h, kernel_w, stride_h, stride_w)
    output = np.empty((batch_size, output_h, output_w, channels), dtype=np.complex128)
    indices = np.empty((batch_size, output_h, output_w, channels), dtype=np.int64)

    for row in range(output_h):
        row_start = row * stride_h
        for col in range(output_w):
            col_start = col * stride_w
            window = padded[:, row_start : row_start + kernel_h, col_start : col_start + kernel_w, :]
            valid_window = valid[:, row_start : row_start + kernel_h, col_start : col_start + kernel_w, :]
            flat_window = window.reshape(batch_size, kernel_h * kernel_w, channels)
            flat_valid = valid_window.reshape(batch_size, kernel_h * kernel_w, channels)
            scores = np.where(flat_valid, np.abs(flat_window), -np.inf)
            local_indices = np.argmax(scores, axis=1)
            batch_indices = np.arange(batch_size)[:, None]
            channel_indices = np.arange(channels)[None, :]
            output[:, row, col, :] = flat_window[batch_indices, local_indices, channel_indices]
            local_rows = local_indices // kernel_w
            local_cols = local_indices % kernel_w
            source_rows = local_rows + row_start - pad_h
            source_cols = local_cols + col_start - pad_w
            indices[:, row, col, :] = source_rows * width + source_cols

    if return_indices:
        return output, indices
    return output


def complex_adaptive_avg_pool1d(x: ArrayLike, output_size: int) -> np.ndarray:
    """Complex counterpart of PyTorch-style ``AdaptiveAvgPool1d``."""

    _positive("output_size", output_size)
    x_array = _as_complex_array(x)
    batch_size, length, channels = x_array.shape
    output = np.empty((batch_size, output_size, channels), dtype=np.complex128)
    for index in range(output_size):
        start, end = _adaptive_window(index, length, output_size)
        output[:, index, :] = np.mean(x_array[:, start:end, :], axis=1)
    return output


def complex_adaptive_max_pool1d(x: ArrayLike, output_size: int, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Complex counterpart of PyTorch-style ``AdaptiveMaxPool1d``."""

    _positive("output_size", output_size)
    x_array = _as_complex_array(x)
    batch_size, length, channels = x_array.shape
    output = np.empty((batch_size, output_size, channels), dtype=np.complex128)
    indices = np.empty((batch_size, output_size, channels), dtype=np.int64)
    for index in range(output_size):
        start, end = _adaptive_window(index, length, output_size)
        window = x_array[:, start:end, :]
        local_indices = np.argmax(np.abs(window), axis=1)
        batch_indices = np.arange(batch_size)[:, None]
        channel_indices = np.arange(channels)[None, :]
        output[:, index, :] = window[batch_indices, local_indices, channel_indices]
        indices[:, index, :] = local_indices + start
    if return_indices:
        return output, indices
    return output


def complex_adaptive_avg_pool2d(x: ArrayLike, output_size: int | tuple[int, int]) -> np.ndarray:
    """Complex counterpart of PyTorch-style ``AdaptiveAvgPool2d``."""

    output_h, output_w = _pair(output_size)
    _positive("output_h", output_h)
    _positive("output_w", output_w)
    x_array = _as_complex_array(x)
    batch_size, height, width, channels = x_array.shape
    output = np.empty((batch_size, output_h, output_w, channels), dtype=np.complex128)
    for row in range(output_h):
        row_start, row_end = _adaptive_window(row, height, output_h)
        for col in range(output_w):
            col_start, col_end = _adaptive_window(col, width, output_w)
            output[:, row, col, :] = np.mean(x_array[:, row_start:row_end, col_start:col_end, :], axis=(1, 2))
    return output


def complex_adaptive_max_pool2d(
    x: ArrayLike,
    output_size: int | tuple[int, int],
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Complex counterpart of PyTorch-style ``AdaptiveMaxPool2d``."""

    output_h, output_w = _pair(output_size)
    _positive("output_h", output_h)
    _positive("output_w", output_w)
    x_array = _as_complex_array(x)
    batch_size, height, width, channels = x_array.shape
    output = np.empty((batch_size, output_h, output_w, channels), dtype=np.complex128)
    indices = np.empty((batch_size, output_h, output_w, channels), dtype=np.int64)
    for row in range(output_h):
        row_start, row_end = _adaptive_window(row, height, output_h)
        for col in range(output_w):
            col_start, col_end = _adaptive_window(col, width, output_w)
            window = x_array[:, row_start:row_end, col_start:col_end, :]
            window_h = row_end - row_start
            window_w = col_end - col_start
            flat_window = window.reshape(batch_size, window_h * window_w, channels)
            local_indices = np.argmax(np.abs(flat_window), axis=1)
            batch_indices = np.arange(batch_size)[:, None]
            channel_indices = np.arange(channels)[None, :]
            output[:, row, col, :] = flat_window[batch_indices, local_indices, channel_indices]
            local_rows = local_indices // window_w
            local_cols = local_indices % window_w
            indices[:, row, col, :] = (local_rows + row_start) * width + (local_cols + col_start)
    if return_indices:
        return output, indices
    return output


def complex_global_avg_pool1d(x: ArrayLike) -> np.ndarray:
    """Average-pool over the full 1D length axis."""

    return np.mean(_as_complex_array(x), axis=1)


def complex_global_max_pool1d(x: ArrayLike) -> np.ndarray:
    """Max-pool by magnitude over the full 1D length axis."""

    x_array = _as_complex_array(x)
    indices = np.argmax(np.abs(x_array), axis=1)
    return x_array[np.arange(x_array.shape[0])[:, None], indices, np.arange(x_array.shape[-1])[None, :]]


def complex_global_avg_pool2d(x: ArrayLike) -> np.ndarray:
    """Average-pool over the full 2D spatial axes."""

    return np.mean(_as_complex_array(x), axis=(1, 2))


def complex_global_max_pool2d(x: ArrayLike) -> np.ndarray:
    """Max-pool by magnitude over the full 2D spatial axes."""

    x_array = _as_complex_array(x)
    batch_size, height, width, channels = x_array.shape
    flat = x_array.reshape(batch_size, height * width, channels)
    indices = np.argmax(np.abs(flat), axis=1)
    return flat[np.arange(batch_size)[:, None], indices, np.arange(channels)[None, :]]
