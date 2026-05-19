# FFT MNIST 8-vs-5 Backprop Comparison

This experiment trains a binary complex-valued classifier on the 2D FFT of
MNIST digits `8` and `5`.

It compares two backpropagation conventions from the same initialization:

- `r2`: split real/imaginary gradients, treating complex values as points in
  `R2`
- `wirtinger`: Wirtinger gradients using `dL / dconj(z)`

By default, the `r2` path uses the literal packed split gradient
`dL/dRe(z) + i dL/dIm(z)`. For real-valued losses this is exactly `2x` the
Wirtinger gradient. Use `--r2-step-scale 0.5` as an equivalence sanity check;
with identical initialization and batches, that setting should produce matching
R2 and Wirtinger curves.

The model uses project-local complex tools:

- `studies.weight_init` for complex Glorot initialization and zero biases
- `studies.maps`/`studies.backprop` for complex dense layers and gradients
- `studies.activations` through cached activation forward/backward
- `studies.loss_functions` for binary cross-entropy upstream gradients

## Run

From the repository root:

```powershell
pip install -r experiments/requirements.txt
```

```powershell
python -m experiments.mnist.train_fft_mnist --download --epochs 10
```

Quick smoke run:

```powershell
python -m experiments.mnist.train_fft_mnist --download --epochs 1 --train-limit 2048 --test-limit 512
```

Equivalence sanity check:

```powershell
python -m experiments.mnist.train_fft_mnist --download --epochs 1 --train-limit 2048 --test-limit 512 --r2-step-scale 0.5
```

If MNIST IDX files already exist, place them under `experiments/mnist/data` and
omit `--download`.

Outputs are written under `experiments/mnist/runs/fft_8_vs_5_compare`:

- `best_model_r2.npz`
- `best_model_wirtinger.npz`
- `final_model_r2.npz`
- `final_model_wirtinger.npz`
- `metrics.json`
