"""Action classifier: Maps embeddings to action labels.

Trains a simple MLP on V-JEPA embeddings + audio-labeled actions.
"""

from typing import Tuple, Optional
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ActionClassifierModel(nn.Module):
    """MLP for action classification from embeddings.

    Architecture:
        1024 (input embedding) → 256 → 4 (action classes)
    """

    def __init__(self, embedding_dim: int = 1024, num_actions: int = 4, dropout: float = 0.2):
        """Initialize classifier.

        Args:
            embedding_dim: Input embedding dimension (default 1024 for V-JEPA)
            num_actions: Number of action classes (default 4: APPROACH, GRASP, MOVE, RELEASE)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions),
        )
        self.num_actions = num_actions

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch_size, embedding_dim)

        Returns:
            (batch_size, num_actions) logits
        """
        return self.model(x)


class ActionClassifier:
    """Wrapper for training and inference."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_actions: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize classifier.

        Args:
            embedding_dim: Input embedding dimension
            num_actions: Number of action classes
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model = ActionClassifierModel(embedding_dim, num_actions).to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.num_actions = num_actions

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward
            logits = self.model(batch_X)
            loss = self.loss_fn(logits, batch_y)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            (accuracy, average_loss)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_X)
                loss = self.loss_fn(logits, batch_y)
                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item()
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        return accuracy, avg_loss

    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict actions for embeddings.

        Args:
            embeddings: (num_frames, embedding_dim) array

        Returns:
            (actions, confidences)
                actions: (num_frames,) predicted action indices
                confidences: (num_frames,) max softmax probability per frame
        """
        self.model.eval()

        with torch.no_grad():
            X = torch.from_numpy(embeddings).float().to(self.device)
            logits = self.model(X)  # (num_frames, num_actions)
            probs = torch.softmax(logits, dim=1)  # (num_frames, num_actions)

            actions = torch.argmax(probs, dim=1)  # (num_frames,)
            confidences = torch.max(probs, dim=1)[0]  # (num_frames,)

        return actions.cpu().numpy(), confidences.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model weights.

        Args:
            path: Path to save weights
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"✓ Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights.

        Args:
            path: Path to load weights from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"✓ Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    """
    Example of training action classifier on embeddings + audio labels.

    Before running this, you need:
    1. Embeddings: data/embeddings/*.npy (1891 frames × 1024 dims) ✅
    2. Labels: action labels extracted from audio narration
    """
    import json
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("ACTION CLASSIFIER: Example Training")
    print("=" * 70)

    # Step 1: Load embeddings
    print("\n1. Loading embeddings...")
    embeddings_dir = Path("data/embeddings")

    all_embeddings = []
    all_labels = []

    for npy_file in sorted(embeddings_dir.glob("*.npy")):
        emb = np.load(npy_file)
        print(f"   Loaded {npy_file.name}: {emb.shape}")

        all_embeddings.append(emb)

        # Placeholder: You need to create labels from audio
        # For now, create dummy labels for demo
        num_frames = emb.shape[0]
        dummy_labels = np.random.randint(0, 4, num_frames)
        all_labels.append(dummy_labels)

    X = np.concatenate(all_embeddings)  # (1891, 1024)
    y = np.concatenate(all_labels)      # (1891,)

    print(f"\n   Total data: {X.shape[0]} frames × {X.shape[1]} dims")
    print(f"   Action distribution: {np.bincount(y)}")

    # Step 2: Train/test split
    print("\n2. Splitting data (80/20 train/val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    # Step 3: Create data loaders
    print("\n3. Creating data loaders...")
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=32, shuffle=False
    )

    # Step 4: Train
    print("\n4. Training classifier...")
    classifier = ActionClassifier(embedding_dim=1024, num_actions=4)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = classifier.train_epoch(train_loader)
        val_acc, val_loss = classifier.evaluate(val_loader)

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val acc: {val_acc:.3f} | "
                  f"Val loss: {val_loss:.4f}")

    # Step 5: Save model
    print("\n5. Saving model...")
    classifier.save("action_classifier.pt")

    # Step 6: Test inference
    print("\n6. Testing inference...")
    actions, confidences = classifier.predict(X_val[:10])
    print(f"   Sample predictions:")
    for i in range(min(5, len(actions))):
        action_name = ["APPROACH", "GRASP", "MOVE", "RELEASE"][actions[i]]
        print(f"     Frame {i}: {action_name} (confidence: {confidences[i]:.3f})")

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
