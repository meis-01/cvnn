# Complex Maps

This package contains NumPy implementations of common complex-valued neural
network maps/layers.

## Dense

```python
from studies.maps import complex_dense

y = complex_dense(x, weights, bias)
```

The dense map computes:

```text
y = x @ weights.T + bias
```

## Convolution

The convolution maps use channels-last inputs and deep-learning
cross-correlation semantics.

```python
from studies.maps import complex_conv1d, complex_conv2d
from studies.maps import complex_conv_transpose1d, complex_conv_transpose2d

y1 = complex_conv1d(x1, kernel1, bias1, stride=1, padding="same")
y2 = complex_conv2d(x2, kernel2, bias2, stride=(2, 2), padding="valid")
z1 = complex_conv_transpose1d(y1, transpose_kernel1, stride=2, padding=1)
z2 = complex_conv_transpose2d(y2, transpose_kernel2, stride=(2, 2), padding=1)
```

Shapes:

- `complex_conv1d`: input `(batch, length, channels)`, kernel
  `(out_channels, kernel_size, channels)`.
- `complex_conv2d`: input `(batch, height, width, channels)`, kernel
  `(out_channels, kernel_height, kernel_width, channels)`.
- `complex_conv_transpose1d`: input `(batch, length, channels)`, kernel
  `(in_channels, kernel_size, out_channels)`.
- `complex_conv_transpose2d`: input `(batch, height, width, channels)`, kernel
  `(in_channels, kernel_height, kernel_width, out_channels)`.

Transposed convolution uses PyTorch-style `stride`, `padding`,
`output_padding`, and `dilation` parameters, adapted to channels-last arrays.

## Transformer

`complex_transformer_encoder` implements a compact complex transformer encoder
block. It uses complex projections for `Q`, `K`, `V`, and output projection.
Attention scores are real:

```text
score = Re(Q @ K.H) / sqrt(head_dim)
```

The softmax probabilities are real and mix complex value vectors.

```python
from studies.maps import TransformerParams, complex_transformer_encoder

params = TransformerParams(
    query_weights=wq,
    key_weights=wk,
    value_weights=wv,
    output_weights=wo,
    feedforward_in_weights=w1,
    feedforward_out_weights=w2,
)

y, attention = complex_transformer_encoder(x, params, num_heads=2)
```
