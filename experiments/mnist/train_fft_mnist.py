"""Train/validate/test a complex classifier on FFT-transformed MNIST."""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.backprop import (  # noqa: E402
    activation_backward_split_r2,
    activation_backward_wirtinger,
    activation_forward,
    dense_backward_split_r2,
    dense_backward_wirtinger,
    dense_forward,
)
from studies.loss_functions import (  # noqa: E402
    binary_cross_entropy_with_logits_split_r2,
    binary_cross_entropy_with_logits_wirtinger,
)
from studies.weight_init import complex_glorot, zeros_complex  # noqa: E402


NEGATIVE_CLASS = 5
POSITIVE_CLASS = 8
IMAGE_SHAPE = (28, 28)
INPUT_DIM = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


@dataclass(frozen=True)
class Config:
    data_dir: str
    run_dir: str
    epochs: int
    batch_size: int
    hidden_dim: int
    learning_rate: float
    val_size: int
    activation: str
    seed: int
    train_limit: int | None
    test_limit: int | None
    fft_shift: bool
    download: bool
    r2_step_scale: float


@dataclass
class Model:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def _open_idx(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_idx(path) as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_idx(path) as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count)


def _find_idx(data_dir: Path, stem: str) -> Path | None:
    candidates = [
        data_dir / f"{stem}.gz",
        data_dir / stem,
        data_dir / "raw" / f"{stem}.gz",
        data_dir / "raw" / stem,
        data_dir / "MNIST" / "raw" / f"{stem}.gz",
        data_dir / "MNIST" / "raw" / stem,
    ]
    return next((path for path in candidates if path.exists()), None)


def _load_idx_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    train_images = _find_idx(data_dir, "train-images-idx3-ubyte")
    train_labels = _find_idx(data_dir, "train-labels-idx1-ubyte")
    test_images = _find_idx(data_dir, "t10k-images-idx3-ubyte")
    test_labels = _find_idx(data_dir, "t10k-labels-idx1-ubyte")
    if not all((train_images, train_labels, test_images, test_labels)):
        return None
    return (
        _read_idx_images(train_images),
        _read_idx_labels(train_labels),
        _read_idx_images(test_images),
        _read_idx_labels(test_labels),
    )


def _load_torchvision_mnist(data_dir: Path, download: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        from torchvision.datasets import MNIST
    except ImportError:
        return None

    train = MNIST(root=str(data_dir), train=True, download=download)
    test = MNIST(root=str(data_dir), train=False, download=download)
    return (
        train.data.numpy(),
        train.targets.numpy(),
        test.data.numpy(),
        test.targets.numpy(),
    )


def _load_openml_mnist(download: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not download:
        return None
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        return None

    try:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    except TypeError:
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    images = mnist.data.astype(np.uint8).reshape(-1, *IMAGE_SHAPE)
    labels = mnist.target.astype(np.uint8)
    return images[:60000], labels[:60000], images[60000:], labels[60000:]


def load_mnist(data_dir: Path, download: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from local IDX files, torchvision, or OpenML."""

    data = _load_idx_mnist(data_dir)
    if data is not None:
        return data

    data = _load_torchvision_mnist(data_dir, download=download)
    if data is not None:
        return data

    data = _load_openml_mnist(download=download)
    if data is not None:
        return data

    raise FileNotFoundError(
        "MNIST data was not found. Place IDX files under experiments/mnist/data "
        "or rerun with --download and torchvision or scikit-learn installed."
    )


def fft_features(images: np.ndarray, fft_shift: bool = False) -> np.ndarray:
    """Convert image batches to flattened complex FFT features."""

    normalized = images.astype(np.float64) / 255.0
    transformed = np.fft.fft2(normalized, axes=(-2, -1), norm="ortho")
    if fft_shift:
        transformed = np.fft.fftshift(transformed, axes=(-2, -1))
    return transformed.reshape(images.shape[0], -1).astype(np.complex128)


def train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    val_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < val_size < len(images):
        raise ValueError("val_size must be between 1 and number of training samples - 1")
    indices = rng.permutation(len(images))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return images[train_indices], labels[train_indices], images[val_indices], labels[val_indices]


def iter_batches(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(images))
    if rng is not None:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def initialize_model(hidden_dim: int, rng: np.random.Generator) -> Model:
    return Model(
        w1=complex_glorot((hidden_dim, INPUT_DIM), layout="dense", rng=rng),
        b1=zeros_complex((hidden_dim,)),
        w2=complex_glorot((1, hidden_dim), layout="dense", rng=rng),
        b2=zeros_complex((1,)),
    )


def clone_model(model: Model) -> Model:
    return Model(w1=model.w1.copy(), b1=model.b1.copy(), w2=model.w2.copy(), b2=model.b2.copy())


def forward(model: Model, x: np.ndarray, activation: str):
    z1, dense1_cache = dense_forward(x, model.w1, model.b1)
    h, activation_cache = activation_forward(activation, z1)
    z2, dense2_cache = dense_forward(h, model.w2, model.b2)
    return z2, (dense1_cache, activation_cache, dense2_cache)


def filter_binary_8_vs_5(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only classes 5 and 8, mapping 5 -> 0 and 8 -> 1."""

    mask = (labels == NEGATIVE_CLASS) | (labels == POSITIVE_CLASS)
    binary_labels = (labels[mask] == POSITIVE_CLASS).astype(np.float64).reshape(-1, 1)
    return images[mask], binary_labels


def binary_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    predictions = (logits.real >= 0.0).astype(np.float64)
    return float(np.mean(predictions == labels))


def r2_backward_and_step(
    model: Model,
    caches,
    logits: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    step_scale: float,
) -> float:
    dense1_cache, activation_cache, dense2_cache = caches

    loss, grad_z2_real, grad_z2_imag = binary_cross_entropy_with_logits_split_r2(logits, labels)
    grad_h, grad_w2, grad_b2 = dense_backward_split_r2(grad_z2_real, grad_z2_imag, dense2_cache)
    grad_z1 = activation_backward_split_r2(grad_h.real, grad_h.imag, activation_cache)
    _, grad_w1, grad_b1 = dense_backward_split_r2(grad_z1.real, grad_z1.imag, dense1_cache)

    model.w1 -= learning_rate * step_scale * grad_w1
    model.b1 -= learning_rate * step_scale * grad_b1
    model.w2 -= learning_rate * step_scale * grad_w2
    model.b2 -= learning_rate * step_scale * grad_b2
    return loss


def wirtinger_backward_and_step(
    model: Model,
    caches,
    logits: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> float:
    dense1_cache, activation_cache, dense2_cache = caches

    loss, grad_z2 = binary_cross_entropy_with_logits_wirtinger(logits, labels)
    grad_h, grad_w2, grad_b2 = dense_backward_wirtinger(grad_z2, dense2_cache)
    grad_z1 = activation_backward_wirtinger(grad_h, activation_cache)
    _, grad_w1, grad_b1 = dense_backward_wirtinger(grad_z1, dense1_cache)

    model.w1 -= learning_rate * grad_w1
    model.b1 -= learning_rate * grad_b1
    model.w2 -= learning_rate * grad_w2
    model.b2 -= learning_rate * grad_b2
    return loss


def evaluate(
    model: Model,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    activation: str,
    fft_shift: bool,
) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    total = 0
    for image_batch, label_batch in iter_batches(images, labels, batch_size):
        x = fft_features(image_batch, fft_shift=fft_shift)
        logits, _ = forward(model, x, activation)
        loss, _, _ = binary_cross_entropy_with_logits_split_r2(logits, label_batch)
        total_loss += loss * len(label_batch)
        correct += int(round(binary_accuracy(logits, label_batch) * len(label_batch)))
        total += len(label_batch)
    return total_loss / total, correct / total


def save_checkpoint(path: Path, model: Model, config: Config, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        config=json.dumps(asdict(config), indent=2),
        metrics=json.dumps(metrics, indent=2),
    )


def train(config: Config) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(config.seed)
    train_images, train_labels, test_images, test_labels = load_mnist(Path(config.data_dir), config.download)
    train_images, train_labels = filter_binary_8_vs_5(train_images, train_labels)
    test_images, test_labels = filter_binary_8_vs_5(test_images, test_labels)

    if config.train_limit is not None:
        train_images = train_images[: config.train_limit]
        train_labels = train_labels[: config.train_limit]
    if config.test_limit is not None:
        test_images = test_images[: config.test_limit]
        test_labels = test_labels[: config.test_limit]

    train_images, train_labels, val_images, val_labels = train_val_split(
        train_images,
        train_labels,
        min(config.val_size, len(train_images) - 1),
        rng,
    )

    initial_model = initialize_model(config.hidden_dim, rng)
    models = {
        "r2": clone_model(initial_model),
        "wirtinger": clone_model(initial_model),
    }
    best_val_accuracy = {"r2": -np.inf, "wirtinger": -np.inf}
    best_metrics: dict[str, dict[str, float]] = {}
    best_paths = {
        "r2": Path(config.run_dir) / "best_model_r2.npz",
        "wirtinger": Path(config.run_dir) / "best_model_wirtinger.npz",
    }

    for epoch in range(1, config.epochs + 1):
        train_loss_total = {"r2": 0.0, "wirtinger": 0.0}
        train_correct = {"r2": 0, "wirtinger": 0}
        train_total = 0
        for image_batch, label_batch in iter_batches(train_images, train_labels, config.batch_size, rng):
            x = fft_features(image_batch, fft_shift=config.fft_shift)
            r2_logits, r2_caches = forward(models["r2"], x, config.activation)
            r2_loss = r2_backward_and_step(
                models["r2"],
                r2_caches,
                r2_logits,
                label_batch,
                config.learning_rate,
                config.r2_step_scale,
            )
            w_logits, w_caches = forward(models["wirtinger"], x, config.activation)
            w_loss = wirtinger_backward_and_step(models["wirtinger"], w_caches, w_logits, label_batch, config.learning_rate)

            train_loss_total["r2"] += r2_loss * len(label_batch)
            train_loss_total["wirtinger"] += w_loss * len(label_batch)
            train_correct["r2"] += int(round(binary_accuracy(r2_logits, label_batch) * len(label_batch)))
            train_correct["wirtinger"] += int(round(binary_accuracy(w_logits, label_batch) * len(label_batch)))
            train_total += len(label_batch)

        metrics_by_method = {}
        for method, model in models.items():
            train_loss = train_loss_total[method] / train_total
            train_accuracy = train_correct[method] / train_total
            val_loss, val_accuracy = evaluate(
                model,
                val_images,
                val_labels,
                config.batch_size,
                config.activation,
                config.fft_shift,
            )
            metrics = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            metrics_by_method[method] = metrics
            if val_accuracy > best_val_accuracy[method]:
                best_val_accuracy[method] = val_accuracy
                best_metrics[method] = metrics
                save_checkpoint(best_paths[method], model, config, metrics)

        print(
            f"epoch {epoch:03d} "
            f"r2_loss={metrics_by_method['r2']['train_loss']:.4f} "
            f"r2_acc={metrics_by_method['r2']['train_accuracy']:.4f} "
            f"r2_val_acc={metrics_by_method['r2']['val_accuracy']:.4f} | "
            f"wirtinger_loss={metrics_by_method['wirtinger']['train_loss']:.4f} "
            f"wirtinger_acc={metrics_by_method['wirtinger']['train_accuracy']:.4f} "
            f"wirtinger_val_acc={metrics_by_method['wirtinger']['val_accuracy']:.4f}"
        )

    final_metrics = {}
    for method, model in models.items():
        test_loss, test_accuracy = evaluate(
            model,
            test_images,
            test_labels,
            config.batch_size,
            config.activation,
            config.fft_shift,
        )
        method_metrics = {
            **best_metrics.get(method, {}),
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        final_metrics[method] = method_metrics
        save_checkpoint(Path(config.run_dir) / f"final_model_{method}.npz", model, config, method_metrics)
        print(f"{method} test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")

    (Path(config.run_dir) / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(f"saved best checkpoints: {best_paths['r2']} and {best_paths['wirtinger']}")
    return final_metrics


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Compare R2 and Wirtinger backprop on FFT MNIST classes 8 vs 5.")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--run-dir", default=str(Path(__file__).resolve().parent / "runs" / "fft_8_vs_5_compare"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--activation", default="split_tanh")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--fft-shift", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--r2-step-scale",
        type=float,
        default=1.0,
        help="Scale applied to packed R2 gradients. Use 0.5 to exactly match Wirtinger update magnitude.",
    )
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
