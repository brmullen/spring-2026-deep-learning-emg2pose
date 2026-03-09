#!/usr/bin/env python3
"""
Train an LSTM model on EMG-to-pose data.

Usage:
  # Quick test with synthetic data
  python train.py --test --epochs 5

  # Train on real data
  python train.py --data_dir /path/to/hdf5s --epochs 100

  # Custom hyperparameters
  python train.py --data_dir /path/to/hdf5s --lr 5e-4 --hidden_size 512 --num_layers 3
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from load_data import get_dataloaders, BATCH_SIZE
from model import EMGPoseLSTM


def train_one_epoch(model, loader, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch. Returns average masked MAE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        emg = batch["emg"].to(device)                    # (B, 16, T)
        targets = batch["joint_angles"].to(device)       # (B, 20, T)
        mask = batch["no_ik_failure"].to(device)          # (B, T)

        preds = model(emg)                                # (B, 20, T)
        mask_exp = mask.unsqueeze(1).expand_as(preds)     # (B, 20, T)

        if mask_exp.any():
            loss = F.l1_loss(preds[mask_exp], targets[mask_exp])
        else:
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on val/test set. Returns average masked MAE loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        emg = batch["emg"].to(device)
        targets = batch["joint_angles"].to(device)
        mask = batch["no_ik_failure"].to(device)

        preds = model(emg)
        mask_exp = mask.unsqueeze(1).expand_as(preds)

        if mask_exp.any():
            loss = F.l1_loss(preds[mask_exp], targets[mask_exp])
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def save_history(history, save_path="checkpoints/loss_history.json"):
    """Save per-epoch loss history to a JSON file.

    Args:
        history: Dict with keys like 'train_loss', 'test_mae', and
                 optionally 'val_loss', each a list of per-epoch values.
        save_path: Where to write the JSON file.
    """
    path = Path(save_path)
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved loss history to {path}")


def save_model(model, optimizer, epoch, best_loss,
               save_path="checkpoints/best_model.pt"):
    """Save model checkpoint to disk.

    Args:
        model: The nn.Module to save.
        optimizer: The optimizer (state saved for resuming training).
        epoch: Current epoch number.
        best_loss: Best loss value at time of saving.
        save_path: Where to write the checkpoint file.
    """
    path = Path(save_path)
    path.parent.mkdir(exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }, path)
    print(f"Saved model checkpoint to {path} (epoch {epoch})")


def plot_losses(history, save_path="training_curves.png"):
    """Plot training curves vs epoch.

    Left y-axis: train and val loss.
    Right y-axis: test loss (if present).

    Args:
        history: Dict with keys 'train_loss', 'val_loss', and
                 optionally 'test_loss', each a list of per-epoch values.
        save_path: Where to save the figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: train and val loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MAE)")
    lines = ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    lines += ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.tick_params(axis="y")

    # Right y-axis: test loss (if present)
    if "test_loss" in history:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Test Loss (MAE)")
        lines += ax2.plot(epochs, history["test_loss"], "g--", label="Test Loss")
        ax2.tick_params(axis="y")

    # Combined legend
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title("Training Curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM on EMG-to-pose data")
    parser.add_argument("--data_dir", type=str, default=None,
        help="Path to directory containing HDF5 session files")
    parser.add_argument("--metadata", type=str, default=None,
        help="Path to emg2pose metadata CSV")
    parser.add_argument("--test", action="store_true",
        help="Run with synthetic test data (no download needed)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--use_test", action="store_true",
        help="Include validation set. Off by default (train+test only).")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
        help="Directory to save model checkpoints, loss history, and plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Data ---
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        metadata=args.metadata,
        test_mode=args.test,
        batch_size=args.batch_size,
        use_test=args.use_test,
    )

    print(f"\n{'Split':<8} {'Batches':>8} {'Samples':>10}")
    for name, loader in loaders.items():
        print(f"{name:<8} {len(loader):>8} {len(loader.dataset):>10}")

    # --- Model ---
    model = EMGPoseLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EMGPoseLSTM ({n_params:,} parameters)")
    print(f"  hidden_size={args.hidden_size}, num_layers={args.num_layers}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    best_loss = float("inf")
    patience = 20
    epochs_without_improvement = 0
    use_test = args.use_test and "test" in loaders

    # --- Training loop ---
    print(f"\nTraining for up to {args.epochs} epochs (patience={patience})")
    print(f"  Checkpoint metric: val_loss")
    if use_test:
        print(f"  Test set: enabled")
    print()

    history = {"train_loss": [], "val_loss": []}
    if use_test:
        history["test_loss"] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss = evaluate(model, loaders["val"], device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if use_test:
            test_loss = evaluate(model, loaders["test"], device)
            history["test_loss"].append(test_loss)

        # Checkpoint based on val loss
        improved = ""
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            save_model(model, optimizer, epoch, best_loss,
                       save_path=output_dir / "best_model.pt")
            improved = " *"
        else:
            epochs_without_improvement += 1

        log = f"Epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        if use_test:
            log += f"  test_loss={test_loss:.4f}"
        log += improved
        print(log)

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # --- Save history and plot ---
    save_history(history, save_path=output_dir / "loss_history.json")
    plot_losses(history, save_path=output_dir / "training_curves.png")

    # --- Final evaluation ---
    print(f"\nBest val_loss: {best_loss:.4f}")

    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    val_loss = evaluate(model, loaders["val"], device)
    print(f"Val loss (MAE): {val_loss:.4f}")

    if use_test:
        test_loss = evaluate(model, loaders["test"], device)
        print(f"Test loss (MAE): {test_loss:.4f}")


if __name__ == "__main__":
    main()
