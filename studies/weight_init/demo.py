"""Run empirical checks for complex initializers."""

from __future__ import annotations

import numpy as np

from studies.weight_init import calculate_fans, complex_glorot, complex_he, complex_normalization_affine


def _complex_variance(weights: np.ndarray) -> float:
    return float(np.mean(np.abs(weights) ** 2))


def main() -> None:
    rng = np.random.default_rng(47)

    dense_shape = (1024, 512)
    dense = complex_glorot(dense_shape, rng=rng)
    dense_fans = calculate_fans(dense_shape)
    dense_target = 2.0 / (dense_fans.fan_in + dense_fans.fan_out)
    print("dense glorot variance:", round(_complex_variance(dense), 6), "target:", round(dense_target, 6))

    conv_shape = (256, 3, 3, 128)
    conv = complex_he(conv_shape, layout="conv", rng=rng)
    conv_fans = calculate_fans(conv_shape, layout="conv")
    conv_target = 2.0 / conv_fans.fan_in
    print("conv he variance:", round(_complex_variance(conv), 6), "target:", round(conv_target, 6))

    gamma, beta = complex_normalization_affine(4)
    print("normalization gamma:", gamma.shape, "beta:", beta.shape)
    print("gamma channel 0:", gamma[0].astype(int).tolist())


if __name__ == "__main__":
    main()
