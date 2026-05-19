# Complex-Valued Activations

This package contains NumPy implementations of common complex-valued activation
functions used in complex-valued neural network studies. Functions accept Python
complex scalars or NumPy arrays and follow NumPy broadcasting rules.

## Install

From this directory:

```powershell
pip install -r requirements.txt
```

The activation layer only depends on NumPy.

## Usage

```python
import numpy as np

from studies.activations import ACTIVATIONS, get_activation, modrelu, zrelu

z = np.array([1 + 2j, -1 + 2j, 1 - 2j, 0j])

print(zrelu(z))
print(modrelu(z, bias=-0.25))

activation = get_activation("split-relu")
print(activation(z))
```

`get_activation()` accepts names with hyphens or underscores, so
`split-relu` and `split_relu` resolve to the same function.

## Available Functions

The public registry is `ACTIVATIONS`.

| Name | Form |
| --- | --- |
| `identity` | Returns `z` unchanged. |
| `complex_tanh` | Holomorphic `tanh(z)`. |
| `complex_sigmoid` | Holomorphic `1 / (1 + exp(-z))`. |
| `split_relu`, `crelu` | Applies ReLU to real and imaginary parts independently. |
| `split_leaky_relu` | Applies leaky ReLU independently to real and imaginary parts. |
| `split_elu` | Applies ELU independently to real and imaginary parts. |
| `split_selu` | Applies SELU independently to real and imaginary parts. |
| `split_sigmoid` | Applies sigmoid independently to real and imaginary parts. |
| `split_tanh` | Applies tanh independently to real and imaginary parts. |
| `split_softplus` | Applies softplus independently to real and imaginary parts. |
| `split_swish` | Applies Swish independently to real and imaginary parts. |
| `split_gelu` | Applies GELU independently to real and imaginary parts. |
| `zrelu` | Keeps values in the first complex quadrant and zeros the rest. |
| `modrelu` | Applies ReLU to magnitude plus bias while preserving phase. |
| `cardioid` | Scales `z` by `0.5 * (1 + cos(angle(z)))`. |
| `amplitude_tanh` | Applies tanh to magnitude while preserving phase. |
| `amplitude_sigmoid` | Applies sigmoid to magnitude while preserving phase. |

## Structure

- `complex_valued.py`: function implementations and `ACTIVATIONS` registry
- `__init__.py`: public exports
- `app/`: interactive Panel/HoloViews demo for plotting the registry functions

## Notes

Phase-preserving activations use a small `EPSILON` guard internally so zero
inputs do not cause division by zero. Split activations are not holomorphic;
they are included because they are widely used as practical baselines in
complex-valued neural network experiments.
