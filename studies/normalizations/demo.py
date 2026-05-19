"""Run shape and covariance checks for complex normalizations."""

from __future__ import annotations

import numpy as np

from studies.normalizations import complex_batch_norm, complex_group_norm, complex_layer_norm


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _channel_covariance(x: np.ndarray) -> np.ndarray:
    real = x.real
    imag = x.imag
    return np.array(
        [
            [np.mean(real * real), np.mean(real * imag)],
            [np.mean(real * imag), np.mean(imag * imag)],
        ]
    )


def main() -> None:
    rng = np.random.default_rng(23)

    batch_x = _complex_normal(rng, (8, 5, 4))
    batch_y, batch_stats = complex_batch_norm(batch_x, return_stats=True)
    print("batch_norm:", batch_y.shape, batch_stats.covariance.shape)
    print("batch_norm channel 0 covariance:", np.round(_channel_covariance(batch_y[..., 0]), 3))

    layer_x = _complex_normal(rng, (3, 6))
    layer_y = complex_layer_norm(layer_x)
    print("layer_norm:", layer_y.shape)
    print("layer_norm sample 0 covariance:", np.round(_channel_covariance(layer_y[0]), 3))

    group_x = _complex_normal(rng, (2, 4, 4, 8))
    group_y = complex_group_norm(group_x, num_groups=4)
    print("group_norm:", group_y.shape)
    print("group_norm sample 0 group 0 covariance:", np.round(_channel_covariance(group_y[0, :, :, :2]), 3))


if __name__ == "__main__":
    main()
