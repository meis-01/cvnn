# Complex Pools

This package contains complex counterparts for common PyTorch pooling layers.
Inputs are channels-last to match the rest of `studies`.

Average pooling computes the arithmetic mean of complex values. Max pooling
selects by largest magnitude, then returns the original complex value.

## Fixed-Window Pools

```python
from studies.pools import complex_avg_pool1d, complex_max_pool1d
from studies.pools import complex_avg_pool2d, complex_max_pool2d

y1 = complex_avg_pool1d(x1, kernel_size=2, stride=2)
y2 = complex_max_pool2d(x2, kernel_size=(2, 2), stride=(2, 2), return_indices=True)
```

Supported PyTorch-style options include `kernel_size`, `stride`, `padding`,
`ceil_mode`, `count_include_pad` for average pooling, and `return_indices` for
max pooling.

## Adaptive Pools

```python
from studies.pools import complex_adaptive_avg_pool2d, complex_adaptive_max_pool2d

y = complex_adaptive_avg_pool2d(x, output_size=(4, 4))
```

Adaptive pooling uses the same window partitioning rule as PyTorch.

## Global Pools

```python
from studies.pools import complex_global_avg_pool2d, complex_global_max_pool2d

features = complex_global_avg_pool2d(x)
```
