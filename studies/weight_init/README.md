# Complex Weight Initialization

This package implements the polar complex initialization used in the complex
neural network literature, notably the approach popularized by Trabelsi et al.,
*Deep Complex Networks*.

The initializer samples:

```text
theta ~ Uniform(-pi, pi)
r     ~ Rayleigh(sigma)
w     = r * exp(i theta)
```

For a Rayleigh scale `sigma`, `E[abs(w) ** 2] = 2 * sigma ** 2`, so the
initializer chooses `sigma` from the target complex variance.

## Initializers

```python
from studies.weight_init import complex_glorot, complex_he, complex_lecun

w_dense = complex_glorot((out_features, in_features), layout="dense")
w_conv = complex_he((out_channels, kernel_size, in_channels), layout="conv")
w_deconv = complex_glorot((in_channels, kernel_size, out_channels), layout="conv_transpose")
```

Supported variance criteria:

- `complex_glorot`: `2 / (fan_in + fan_out)`
- `complex_he`: `2 / fan_in`
- `complex_lecun`: `1 / fan_in`

Biases should usually start at zero:

```python
from studies.weight_init import zeros_complex

bias = zeros_complex((out_features,))
```

## Coverage

Parameterized maps are covered by `layout`:

- Dense and transformer projections/feed-forward weights: `layout="dense"`
- `complex_conv1d` and `complex_conv2d`: `layout="conv"`
- `complex_conv_transpose1d` and `complex_conv_transpose2d`:
  `layout="conv_transpose"`

Pooling, activation, and loss modules do not have learned weights.

## Normalization Affine Parameters

Complex batch/layer/group normalization layers use real `2x2` affine matrices
for the real/imaginary channel vector. Initialize these as identity matrices and
use zero complex offsets:

```python
from studies.weight_init import complex_identity_gamma, complex_normalization_affine

gamma = complex_identity_gamma(channels)
gamma, beta = complex_normalization_affine(channels)
```
