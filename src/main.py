from __future__ import annotations

import argparse
from collections.abc import Sized
import json
import random
import time
from pathlib import Path
from typing import cast

import matplotlib  # type: ignore[import-not-found]
import matplotlib.pyplot as plt  # type: ignore[import-not-found]
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

matplotlib.use("Agg")


class DNNModel(nn.Module):
    """Fully connected baseline model for MNIST classification."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNNModel(nn.Module):
    """Convolutional model that preserves image spatial structure."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DNN and CNN on MNIST and compare performance."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for each model."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for train/test loaders."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Adam learning rate."
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use. auto picks CUDA if available.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to cache MNIST data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        help="Optional cap on the number of training samples (for quick experiments).",
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="Optional cap on the number of test samples (for quick experiments).",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=Path("charts"),
        help="Directory where training charts are saved.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("REPORT.md"),
        help="Path to write the markdown report.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(device_arg)


def maybe_subset(
    dataset: torch.utils.data.Dataset, max_items: int | None
) -> torch.utils.data.Dataset:
    sized_dataset = cast(Sized, dataset)
    if max_items is None or max_items >= len(sized_dataset):
        return dataset
    return Subset(dataset, range(max_items))


def build_mnist_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    train_subset: int | None,
    test_subset: int | None,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, transform=transform, download=True
    )

    train_dataset = maybe_subset(train_dataset, train_subset)
    test_dataset = maybe_subset(test_dataset, test_subset)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def run_experiment(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n===== {model_name} =====")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    history: list[dict] = []
    best_test_acc = 0.0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        best_test_acc = max(best_test_acc, test_acc)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(epoch_stats)

        print(
            f"Epoch {epoch:>2}/{epochs}: "
            f"train loss={train_loss:.4f}, train acc={train_acc:.2f}% | "
            f"test loss={test_loss:.4f}, test acc={test_acc:.2f}%"
        )

    elapsed_seconds = time.perf_counter() - start
    return {
        "model": model_name,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "trainable_parameters": count_trainable_parameters(model),
        "training_time_seconds": elapsed_seconds,
        "best_test_accuracy": best_test_acc,
        "final_test_accuracy": history[-1]["test_acc"],
        "history": history,
    }


def build_comparison(dnn_result: dict, cnn_result: dict) -> dict:
    accuracy_delta = cnn_result["best_test_accuracy"] - dnn_result["best_test_accuracy"]
    time_delta = (
        cnn_result["training_time_seconds"] - dnn_result["training_time_seconds"]
    )
    parameter_ratio = (
        cnn_result["trainable_parameters"] / dnn_result["trainable_parameters"]
    )

    return {
        "best_accuracy_delta_cnn_minus_dnn": accuracy_delta,
        "training_time_delta_seconds_cnn_minus_dnn": time_delta,
        "parameter_ratio_cnn_over_dnn": parameter_ratio,
    }


def print_summary(dnn_result: dict, cnn_result: dict, comparison: dict) -> None:
    print("\n===== Quantitative Comparison =====")
    print(
        "DNN: "
        f"best test acc={dnn_result['best_test_accuracy']:.2f}%, "
        f"params={dnn_result['trainable_parameters']:,}, "
        f"time={dnn_result['training_time_seconds']:.2f}s"
    )
    print(
        "CNN: "
        f"best test acc={cnn_result['best_test_accuracy']:.2f}%, "
        f"params={cnn_result['trainable_parameters']:,}, "
        f"time={cnn_result['training_time_seconds']:.2f}s"
    )
    print(
        "Delta (CNN - DNN): "
        f"acc={comparison['best_accuracy_delta_cnn_minus_dnn']:.2f}%, "
        f"time={comparison['training_time_delta_seconds_cnn_minus_dnn']:.2f}s, "
        f"param ratio={comparison['parameter_ratio_cnn_over_dnn']:.3f}"
    )


def _history_series(model_result: dict, key: str) -> tuple[list[int], list[float]]:
    epochs = [entry["epoch"] for entry in model_result["history"]]
    values = [entry[key] for entry in model_result["history"]]
    return epochs, values


def save_training_charts(
    dnn_result: dict, cnn_result: dict, charts_dir: Path
) -> dict[str, str]:
    charts_dir.mkdir(parents=True, exist_ok=True)

    dnn_epochs, dnn_train_loss = _history_series(dnn_result, "train_loss")
    _, dnn_test_loss = _history_series(dnn_result, "test_loss")
    cnn_epochs, cnn_train_loss = _history_series(cnn_result, "train_loss")
    _, cnn_test_loss = _history_series(cnn_result, "test_loss")

    plt.figure(figsize=(9, 6))
    plt.plot(dnn_epochs, dnn_train_loss, marker="o", label="DNN Train Loss")
    plt.plot(
        dnn_epochs, dnn_test_loss, marker="o", linestyle="--", label="DNN Test Loss"
    )
    plt.plot(cnn_epochs, cnn_train_loss, marker="s", label="CNN Train Loss")
    plt.plot(
        cnn_epochs, cnn_test_loss, marker="s", linestyle="--", label="CNN Test Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MNIST Loss Curves: DNN vs CNN")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = charts_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=150)
    plt.close()

    _, dnn_train_acc = _history_series(dnn_result, "train_acc")
    _, dnn_test_acc = _history_series(dnn_result, "test_acc")
    _, cnn_train_acc = _history_series(cnn_result, "train_acc")
    _, cnn_test_acc = _history_series(cnn_result, "test_acc")

    plt.figure(figsize=(9, 6))
    plt.plot(dnn_epochs, dnn_train_acc, marker="o", label="DNN Train Acc")
    plt.plot(dnn_epochs, dnn_test_acc, marker="o", linestyle="--", label="DNN Test Acc")
    plt.plot(cnn_epochs, cnn_train_acc, marker="s", label="CNN Train Acc")
    plt.plot(cnn_epochs, cnn_test_acc, marker="s", linestyle="--", label="CNN Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("MNIST Accuracy Curves: DNN vs CNN")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    accuracy_path = charts_dir / "accuracy_curves.png"
    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    print(f"Saved chart: {loss_path}")
    print(f"Saved chart: {accuracy_path}")
    return {
        "loss_curve": str(loss_path),
        "accuracy_curve": str(accuracy_path),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    train_loader, test_loader = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    train_size = len(cast(Sized, train_loader.dataset))
    test_size = len(cast(Sized, test_loader.dataset))
    print(f"Using device: {device}")
    print(f"Train samples: {train_size:,}, Test samples: {test_size:,}")

    dnn_result = run_experiment(
        model_name="DNN",
        model=DNNModel(),
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    cnn_result = run_experiment(
        model_name="CNN",
        model=CNNModel(),
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    comparison = build_comparison(dnn_result, cnn_result)
    print_summary(dnn_result, cnn_result, comparison)

    chart_paths = save_training_charts(dnn_result, cnn_result, args.charts_dir)

    report = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "device": str(device),
            "train_subset": args.train_subset,
            "test_subset": args.test_subset,
        },
        "results": {
            "DNN": dnn_result,
            "CNN": cnn_result,
        },
        "comparison": comparison,
        "artifacts": chart_paths,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
