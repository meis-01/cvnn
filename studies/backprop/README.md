# Complex Backpropagation

This package contains two NumPy implementations of the same complex dense-layer
backpropagation step.

The example layer is:

```python
y = x @ weights.T + bias
loss = mean_batch(sum_outputs(abs(y - target) ** 2))
```

## Split Real/Imaginary Backprop

`dense_backward_split_r2()` treats a complex value as a point in `R2`. If:

```text
x = a + i b
W = A + i B
y = u + i v
```

then:

```text
u = a @ A.T - b @ B.T + Re(bias)
v = a @ B.T + b @ A.T + Im(bias)
```

The function differentiates those real equations and returns complex gradients
whose real and imaginary parts are the `R2` gradients.

## Wirtinger Backprop

`dense_backward_wirtinger()` propagates `dL / dconj(z)`. For real-valued losses,
this is the gradient used by steepest descent:

```python
weights -= learning_rate * grad_weights_conj
bias -= learning_rate * grad_bias_conj
```

For the MSE used here:

```python
grad_y_conj = (y - target) / batch_size
```

The relation between both implementations is:

```python
grad_wirtinger = 0.5 * (grad_real + 1j * grad_imag)
```

## Usage

```python
import numpy as np

from studies.backprop import (
    dense_backward_split_r2,
    dense_backward_wirtinger,
    dense_forward,
    mse_loss_split_r2,
    mse_loss_wirtinger,
)

x = np.array([[1 + 2j, -0.5 + 0.25j]])
weights = np.array([[0.25 - 0.75j, 0.5 + 0.1j]])
bias = np.array([0.05 + 0.2j])
target = np.array([[0.2 - 0.1j]])

y, cache = dense_forward(x, weights, bias)

loss_r2, grad_y_real, grad_y_imag = mse_loss_split_r2(y, target)
grad_x_r2, grad_weights_r2, grad_bias_r2 = dense_backward_split_r2(
    grad_y_real,
    grad_y_imag,
    cache,
)

loss_w, grad_y_conj = mse_loss_wirtinger(y, target)
grad_x_w, grad_weights_w, grad_bias_w = dense_backward_wirtinger(grad_y_conj, cache)

assert np.allclose(loss_r2, loss_w)
assert np.allclose(0.5 * grad_x_r2, grad_x_w)
assert np.allclose(0.5 * grad_weights_r2, grad_weights_w)
assert np.allclose(0.5 * grad_bias_r2, grad_bias_w)
```
