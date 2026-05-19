"""Run a small equivalence check for the complex backpropagation examples."""

from __future__ import annotations

import numpy as np

from studies.backprop import (
    dense_backward_split_r2,
    dense_backward_wirtinger,
    dense_forward,
    mse_loss_split_r2,
    mse_loss_wirtinger,
)


def main() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    weights = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    bias = rng.normal(size=2) + 1j * rng.normal(size=2)
    target = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))

    y, cache = dense_forward(x, weights, bias)

    loss_r2, grad_y_real, grad_y_imag = mse_loss_split_r2(y, target)
    grad_x_r2, grad_weights_r2, grad_bias_r2 = dense_backward_split_r2(
        grad_y_real,
        grad_y_imag,
        cache,
    )

    loss_w, grad_y_conj = mse_loss_wirtinger(y, target)
    grad_x_w, grad_weights_w, grad_bias_w = dense_backward_wirtinger(grad_y_conj, cache)

    print(f"split R2 loss: {loss_r2:.12f}")
    print(f"Wirtinger loss: {loss_w:.12f}")
    print("grad_x match:", np.allclose(0.5 * grad_x_r2, grad_x_w))
    print("grad_weights match:", np.allclose(0.5 * grad_weights_r2, grad_weights_w))
    print("grad_bias match:", np.allclose(0.5 * grad_bias_r2, grad_bias_w))


if __name__ == "__main__":
    main()
