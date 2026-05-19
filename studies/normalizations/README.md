# Complex Normalizations

This package contains NumPy implementations of complex-valued normalization
layers. Each layer treats `z = x + i y` as the real vector `[x, y]` and whitens
with the full real/imaginary covariance matrix.

## Batch Norm

```python
from studies.normalizations import complex_batch_norm

y, stats = complex_batch_norm(x, return_stats=True)
```

Inputs are channels-last. Statistics are computed over every axis except the
final channel axis, so the same function works for dense, 1D, and 2D outputs.

Optional affine parameters:

- `gamma`: `(channels, 2, 2)` real matrices
- `beta`: `(channels,)` complex offsets

## Layer Norm

```python
from studies.normalizations import complex_layer_norm

y = complex_layer_norm(x)
```

Layer norm computes statistics across the final feature axis for each sample or
token independently.

## Group Norm

```python
from studies.normalizations import complex_group_norm

y = complex_group_norm(x, num_groups=4)
```

Group norm expects channels-last inputs and computes covariance statistics per
sample and per group, over spatial positions and grouped channels.
