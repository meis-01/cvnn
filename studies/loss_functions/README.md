# Loss Functions

This package contains NumPy losses that return both the scalar loss and the
upstream gradient needed by complex backpropagation.

## Reconstruction

Use reconstruction MSE for complex-valued reconstruction targets:

```python
from studies.loss_functions import reconstruction_mse_split_r2, reconstruction_mse_wirtinger

loss, grad_real, grad_imag = reconstruction_mse_split_r2(prediction, target)
loss, grad_conj = reconstruction_mse_wirtinger(prediction, target)
```

The loss is:

```text
mean_batch(sum_features(abs(prediction - target) ** 2))
```

## Binary Classification

For binary classification, use binary cross entropy. If your model already
outputs probabilities:

```python
from studies.loss_functions import binary_cross_entropy

loss, grad_probability = binary_cross_entropy(probability, target)
```

If your model outputs logits, prefer the logits version:

```python
from studies.loss_functions import (
    binary_cross_entropy_with_logits_split_r2,
    binary_cross_entropy_with_logits_wirtinger,
)

loss, grad_real, grad_imag = binary_cross_entropy_with_logits_split_r2(logit, target)
loss, grad_conj = binary_cross_entropy_with_logits_wirtinger(logit, target)
```

Complex logits are converted to a real classification score with `score_mode`:

- `real`: uses `Re(logit)` and leaves the imaginary upstream gradient at zero.
- `magnitude`: uses `abs(logit)` and backpropagates through the magnitude.
