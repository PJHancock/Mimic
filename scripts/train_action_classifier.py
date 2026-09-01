#!/usr/bin/env python3
"""Train temporal action classifier using embeddings + audio labels.

Combines V-JEPA embeddings with frame-level action labels to train an LSTM
classifier that predicts action sequences. Labels are extracted from audio narration.

Usage:
    uv run python scripts/train_action_classifier.py

    Or with custom paths:
    uv run python scripts/train_action_classifier.py \\
        --embeddings-dir data/embeddings/ \\
        --labels-dir data/labels/ \\
        --output-dir models/ \\
        --epochs 30
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mimic.integration import load_skill_system
from mimic.vision.action_classifier import ActionClassifier


def load_video_data(embeddings_dir: str, labels_dir: str):
    """Load embeddings and labels for all videos.

    Returns:
        List of (embeddings, labels) tuples
        Each embeddings: (num_frames, 1024)
        Each labels: (num_frames, 5) one-hot encoded
    """
    embeddings_dir = Path(embeddings_dir)
    labels_dir = Path(labels_dir)

    video_data = []

    # Find all embedding files
    emb_files = sorted(embeddings_dir.glob("*.npy"))
    emb_files = [f for f in emb_files if "_meta" not in f.name]

    if not emb_files:
        raise FileNotFoundError(f"No embeddings found in {embeddings_dir}")

    print(f"Loading {len(emb_files)} videos...\n")

    for emb_file in emb_files:
        video_stem = emb_file.stem

        # Load embeddings
        embeddings = np.load(emb_file)  # (num_frames, 1024)

        # Load labels
        label_file = labels_dir / f"{video_stem}.npy"
        if not label_file.exists():
            print(f"  ⚠ Skipping {video_stem}: labels not found")
            continue

        labels = np.load(label_file)  # (num_frames, 5) one-hot

        # Verify alignment
        if embeddings.shape[0] != labels.shape[0]:
            print(
                f"  ⚠ Skipping {video_stem}: shape mismatch "
                f"(embeddings: {embeddings.shape[0]}, labels: {labels.shape[0]})"
            )
            continue

        video_data.append((embeddings, labels))
        print(f"  ✓ {video_stem}: {embeddings.shape[0]} frames")

    if not video_data:
        raise ValueError("No valid video data loaded")

    total_frames = sum(v[0].shape[0] for v in video_data)
    print(f"\nTotal frames: {total_frames}")

    return video_data


def create_lstm_dataloader(videos, batch_size=2, shuffle=True):
    """Create dataloader for LSTM from video sequences.

    Args:
        videos: List of (embeddings, labels) tuples
        batch_size: Batch size for training
        shuffle: Whether to shuffle videos

    Returns:
        DataLoader yielding padded sequences
    """

    def pad_sequence_batch(sequences, pad_value=0):
        """Pad sequences to same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(np.pad(seq, ((0, pad_len), (0, 0)), constant_values=pad_value))
        return np.stack(padded)

    embeddings_list = [v[0] for v in videos]
    labels_list = [v[1] for v in videos]

    X_padded = pad_sequence_batch(embeddings_list)
    y_padded = pad_sequence_batch(labels_list)

    X_t = torch.from_numpy(X_padded).float()
    y_t = torch.from_numpy(y_padded).float()

    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_label_weights(all_labels):
    """Compute class weights to handle imbalanced actions.

    Args:
        all_labels: List of (num_frames, 5) label matrices

    Returns:
        torch.Tensor of shape (5,) with inverse class frequencies
    """
    # Stack all labels and sum across frames
    all_labels_stacked = np.concatenate([l for l in all_labels], axis=0)
    class_counts = all_labels_stacked.sum(axis=0)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Weights = total / (num_classes * count_per_class)
    total = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    weights = weights / weights.sum()  # Normalize

    return torch.from_numpy(weights).float()


def plot_training_curves(train_losses, val_losses, val_accs, output_dir):
    """Plot training and validation curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accs: List of validation accuracies per epoch
        output_dir: Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = np.arange(1, len(train_losses) + 1)

    # Plot 1: Loss curves
    ax1.plot(epochs, train_losses, "b-", label="Training loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation accuracy
    ax2.plot(epochs, val_accs, "g-", linewidth=2)
    best_epoch = np.argmax(val_accs) + 1
    best_acc = np.max(val_accs)
    ax2.scatter(
        [best_epoch],
        [best_acc],
        color="red",
        s=100,
        zorder=5,
        label=f"Best: {best_acc:.1%} (epoch {best_epoch})",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"  ✓ Saved plot to: {output_path}")

    plt.close()


def train_epoch(model, classifier, train_loader, device):
    """Train for one epoch.

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        logits = model(batch_X)  # (batch, seq_len, 5)

        # Reshape for loss computation
        batch_size, seq_len, num_actions = logits.shape
        logits_flat = logits.reshape(-1, num_actions)
        batch_y_flat = batch_y.reshape(-1, num_actions)

        # Loss: binary cross-entropy for multi-label (one-hot)
        loss = nn.BCEWithLogitsLoss()(logits_flat, batch_y_flat)

        # Backward
        classifier.optimizer.zero_grad()
        loss.backward()
        classifier.optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_epoch(model, val_loader, device):
    """Evaluate on validation set.

    Returns:
        (accuracy, loss)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)  # (batch, seq_len, 5)

            batch_size, seq_len, num_actions = logits.shape
            logits_flat = logits.reshape(-1, num_actions)
            batch_y_flat = batch_y.reshape(-1, num_actions)

            loss = nn.BCEWithLogitsLoss()(logits_flat, batch_y_flat)
            total_loss += loss.item()

            # Accuracy: one-hot predictions (argmax)
            preds = torch.argmax(logits_flat, dim=1)
            targets = torch.argmax(batch_y_flat, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    return accuracy, avg_loss


def main():
    """Train action classifier."""
    parser = argparse.ArgumentParser(
        description="Train temporal action classifier on embeddings + labels"
    )
    parser.add_argument(
        "--embeddings-dir",
        default="data/embeddings/",
        help="Directory containing embeddings (.npy files)",
    )
    parser.add_argument(
        "--labels-dir",
        default="data/labels/",
        help="Directory containing labels (.npy files)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--model-name",
        default="action_classifier_lstm.pt",
        help="Name of saved model file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of videos for training (rest for validation)",
    )
    parser.add_argument(
        "--skill-config",
        default="configs/skills/pick_place.yaml",
        help="Versioned classifier vocabulary and graph configuration",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ACTION CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"\nEmbeddings directory: {args.embeddings_dir}")
    print(f"Labels directory: {args.labels_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Train/Val split: {args.train_split:.0%}/{1-args.train_split:.0%}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skill_system = load_skill_system(args.skill_config)

    # Load data
    print("1. Loading embeddings and labels...")
    video_data = load_video_data(args.embeddings_dir, args.labels_dir)
    for _, labels in video_data:
        if labels.ndim != 2 or labels.shape[1] != skill_system.catalog.class_count:
            raise ValueError("Training labels must have one column for every active catalog label")

    # Split train/val by video
    print("\n2. Splitting data by video...")
    split_idx = int(len(video_data) * args.train_split)
    train_videos = video_data[:split_idx]
    val_videos = video_data[split_idx:]

    train_frames = sum(v[0].shape[0] for v in train_videos)
    val_frames = sum(v[0].shape[0] for v in val_videos)

    print(f"  Train: {len(train_videos)} videos, {train_frames} frames")
    print(f"  Val: {len(val_videos)} videos, {val_frames} frames")

    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader = create_lstm_dataloader(train_videos, batch_size=args.batch_size, shuffle=True)
    val_loader = create_lstm_dataloader(val_videos, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print("\n4. Initializing LSTM classifier...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    classifier = ActionClassifier(
        embedding_dim=1024,
        num_actions=skill_system.catalog.class_count,
        device=device,
        model_type="lstm",
    )

    # Update optimizer learning rate
    for param_group in classifier.optimizer.param_groups:
        param_group["lr"] = args.learning_rate

    # Training loop
    print("\n5. Training...")
    best_val_acc = 0.0
    best_epoch = 0

    # Track losses and accuracies for plotting
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(classifier.model, classifier, train_loader, device)
        val_acc, val_loss = evaluate_epoch(classifier.model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val acc: {val_acc:.3f} | "
                f"Val loss: {val_loss:.4f}", flush=True
            )

    print(f"\n  Best validation accuracy: {best_val_acc:.3f} (epoch {best_epoch})")

    # Save model
    print("\n6. Saving model...")
    model_path = output_dir / args.model_name
    classifier.save(str(model_path), catalog=skill_system.catalog)
    print(f"  Saved to: {model_path}")

    # Plot training curves
    print("\n7. Plotting training curves...")
    plot_training_curves(train_losses, val_losses, val_accs, output_dir)

    # Save training config
    config = {
        "embeddings_dir": str(args.embeddings_dir),
        "labels_dir": str(args.labels_dir),
        "num_videos": len(video_data),
        "num_train_videos": len(train_videos),
        "num_val_videos": len(val_videos),
        "num_train_frames": train_frames,
        "num_val_frames": val_frames,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_split": args.train_split,
        "best_val_accuracy": float(best_val_acc),
        "best_epoch": best_epoch,
        "model_path": str(model_path),
        "device": str(device),
        "catalog": {
            "schema_version": skill_system.catalog.schema_version,
            "fingerprint": skill_system.catalog.fingerprint,
            "labels": list(skill_system.catalog.labels),
        },
    }

    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved to: {config_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Best validation accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"Training frames: {train_frames}")
    print(f"Validation frames: {val_frames}")
    print("\nNext steps:")
    print("  1. Evaluate on test set (if available)")
    print("  2. Run inference on new videos:")
    print("     classifier = ActionClassifier(model_type='lstm')")
    print("     classifier.load(str(model_path))")
    print("     probabilities = classifier.predict_probabilities(embeddings)")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
