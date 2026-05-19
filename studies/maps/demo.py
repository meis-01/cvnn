"""Run shape checks for complex map implementations."""

from __future__ import annotations

import numpy as np

from studies.maps import (
    TransformerParams,
    complex_conv1d,
    complex_conv2d,
    complex_conv_transpose1d,
    complex_conv_transpose2d,
    complex_dense,
    complex_transformer_encoder,
)


def main() -> None:
    rng = np.random.default_rng(11)

    dense_x = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    dense_w = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    dense_b = rng.normal(size=4) + 1j * rng.normal(size=4)
    print("dense:", complex_dense(dense_x, dense_w, dense_b).shape)

    conv1_x = rng.normal(size=(2, 8, 3)) + 1j * rng.normal(size=(2, 8, 3))
    conv1_k = rng.normal(size=(5, 3, 3)) + 1j * rng.normal(size=(5, 3, 3))
    print("conv1d:", complex_conv1d(conv1_x, conv1_k, padding="same").shape)
    convt1_k = rng.normal(size=(3, 3, 5)) + 1j * rng.normal(size=(3, 3, 5))
    print("conv_transpose1d:", complex_conv_transpose1d(conv1_x, convt1_k, stride=2, padding=1, output_padding=1).shape)

    conv2_x = rng.normal(size=(2, 8, 7, 3)) + 1j * rng.normal(size=(2, 8, 7, 3))
    conv2_k = rng.normal(size=(6, 3, 3, 3)) + 1j * rng.normal(size=(6, 3, 3, 3))
    print("conv2d:", complex_conv2d(conv2_x, conv2_k, stride=(2, 2), padding="same").shape)
    convt2_k = rng.normal(size=(3, 3, 3, 6)) + 1j * rng.normal(size=(3, 3, 3, 6))
    print("conv_transpose2d:", complex_conv_transpose2d(conv2_x, convt2_k, stride=(2, 2), padding=1, output_padding=(1, 0)).shape)

    model_dim = 4
    ff_dim = 8
    transformer_x = rng.normal(size=(2, 5, model_dim)) + 1j * rng.normal(size=(2, 5, model_dim))
    params = TransformerParams(
        query_weights=rng.normal(size=(model_dim, model_dim)) + 1j * rng.normal(size=(model_dim, model_dim)),
        key_weights=rng.normal(size=(model_dim, model_dim)) + 1j * rng.normal(size=(model_dim, model_dim)),
        value_weights=rng.normal(size=(model_dim, model_dim)) + 1j * rng.normal(size=(model_dim, model_dim)),
        output_weights=rng.normal(size=(model_dim, model_dim)) + 1j * rng.normal(size=(model_dim, model_dim)),
        feedforward_in_weights=rng.normal(size=(ff_dim, model_dim)) + 1j * rng.normal(size=(ff_dim, model_dim)),
        feedforward_out_weights=rng.normal(size=(model_dim, ff_dim)) + 1j * rng.normal(size=(model_dim, ff_dim)),
    )
    y, attention = complex_transformer_encoder(transformer_x, params, num_heads=2)
    print("transformer:", y.shape)
    print("attention:", attention.shape)


if __name__ == "__main__":
    main()
