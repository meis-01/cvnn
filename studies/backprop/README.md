# Complex Backpropagation

This package contains NumPy backpropagation code for the complex maps in
`studies.maps` and activations in `studies.activations`. Loss functions live in
`studies.loss_functions` and provide the upstream gradients consumed here.

Two gradient conventions are used:

- Split `R2`: stores `dL/dRe(z) + i dL/dIm(z)`.
- Wirtinger: stores `dL/dconj(z)`, the update direction for real-valued losses.

For real-valued losses:

```python
grad_wirtinger = 0.5 * grad_split_r2
```

## Covered Maps

- Activations: `activation_forward`, `activation_backward_split_r2`,
  `activation_backward_wirtinger`.
- Dense: `dense_forward`, `dense_backward_split_r2`,
  `dense_backward_wirtinger`.
- 1D convolution: `complex_conv1d_forward`, `conv1d_backward_split_r2`,
  `conv1d_backward_wirtinger`.
- 2D convolution: `complex_conv2d_forward`, `conv2d_backward_split_r2`,
  `conv2d_backward_wirtinger`.
- 1D transposed convolution: `complex_conv_transpose1d_forward`,
  `conv_transpose1d_backward_split_r2`,
  `conv_transpose1d_backward_wirtinger`.
- 2D transposed convolution: `complex_conv_transpose2d_forward`,
  `conv_transpose2d_backward_split_r2`,
  `conv_transpose2d_backward_wirtinger`.
- Transformer encoder: `complex_transformer_encoder_forward`,
  `transformer_encoder_backward_wirtinger`.

The convolution functions use channels-last cross-correlation semantics, matching
`studies.maps`. Transposed convolution kernels use channels-last-adapted
PyTorch-style shapes: `(in_channels, kernel_size, out_channels)` for 1D and
`(in_channels, kernel_height, kernel_width, out_channels)` for 2D.

Activation backward coverage follows the public `studies.activations.ACTIVATIONS`
registry. `missing_activation_backwards()` should return an empty set.

Piecewise activations use the standard zero subgradient at nondifferentiable
boundaries.

## Dense Example

```python
import numpy as np

from studies.backprop import dense_backward_wirtinger, dense_forward
from studies.loss_functions import reconstruction_mse_wirtinger

x = np.array([[1 + 2j, -0.5 + 0.25j]])
weights = np.array([[0.25 - 0.75j, 0.5 + 0.1j]])
bias = np.array([0.05 + 0.2j])
target = np.array([[0.2 - 0.1j]])

y, cache = dense_forward(x, weights, bias)
loss, grad_y_conj = reconstruction_mse_wirtinger(y, target)
grad_x, grad_weights, grad_bias = dense_backward_wirtinger(grad_y_conj, cache)
```

## Activation Example

```python
from studies.backprop import activation_backward_wirtinger, activation_forward
from studies.loss_functions import reconstruction_mse_wirtinger

y, cache = activation_forward("modrelu", z, bias=-0.2)
loss, grad_y_conj = reconstruction_mse_wirtinger(y, target)
grad_z = activation_backward_wirtinger(grad_y_conj, cache)
```

## Transformer Example

```python
from studies.backprop import (
    complex_transformer_encoder_forward,
    transformer_encoder_backward_wirtinger,
)
from studies.loss_functions import reconstruction_mse_wirtinger
from studies.maps import TransformerParams

params = TransformerParams(
    query_weights=wq,
    key_weights=wk,
    value_weights=wv,
    output_weights=wo,
    feedforward_in_weights=w1,
    feedforward_out_weights=w2,
)

y, cache = complex_transformer_encoder_forward(x, params, num_heads=2)
loss, grad_y_conj = reconstruction_mse_wirtinger(y, target)
grad_x, grad_params = transformer_encoder_backward_wirtinger(grad_y_conj, cache)
```
