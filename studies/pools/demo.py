"""Run shape checks for complex pooling layers."""

from __future__ import annotations

import numpy as np

from studies.pools import (
    complex_adaptive_avg_pool1d,
    complex_adaptive_avg_pool2d,
    complex_adaptive_max_pool1d,
    complex_adaptive_max_pool2d,
    complex_avg_pool1d,
    complex_avg_pool2d,
    complex_global_avg_pool2d,
    complex_global_max_pool2d,
    complex_max_pool1d,
    complex_max_pool2d,
)


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def main() -> None:
    rng = np.random.default_rng(31)
    x1 = _complex_normal(rng, (2, 9, 3))
    x2 = _complex_normal(rng, (2, 7, 8, 3))

    print("avg_pool1d:", complex_avg_pool1d(x1, kernel_size=3, stride=2, padding=1).shape)
    max1, idx1 = complex_max_pool1d(x1, kernel_size=3, stride=2, padding=1, return_indices=True)
    print("max_pool1d:", max1.shape, idx1.shape)
    print("adaptive_avg_pool1d:", complex_adaptive_avg_pool1d(x1, output_size=4).shape)
    print("adaptive_max_pool1d:", complex_adaptive_max_pool1d(x1, output_size=4).shape)

    print("avg_pool2d:", complex_avg_pool2d(x2, kernel_size=(3, 2), stride=(2, 2), padding=(1, 0)).shape)
    max2, idx2 = complex_max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
    print("max_pool2d:", max2.shape, idx2.shape)
    print("adaptive_avg_pool2d:", complex_adaptive_avg_pool2d(x2, output_size=(3, 4)).shape)
    print("adaptive_max_pool2d:", complex_adaptive_max_pool2d(x2, output_size=(3, 4)).shape)
    print("global_avg_pool2d:", complex_global_avg_pool2d(x2).shape)
    print("global_max_pool2d:", complex_global_max_pool2d(x2).shape)


if __name__ == "__main__":
    main()
