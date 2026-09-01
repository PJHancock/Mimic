"""Temporal manipulation-phase classifier over cached frame features."""

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mimic.common.types import ActionPhase

if TYPE_CHECKING:
    from mimic.skills.catalog import SkillCatalog

logger = logging.getLogger(__name__)

DEFAULT_ACTION_NAMES = tuple(phase.value for phase in ActionPhase)


class ActionClassifierModel(nn.Module):
    """MLP for action classification from embeddings (baseline).

    Architecture:
        1024 (input embedding) → 256 → 5 (action classes)

    Action classes:
        0: IDLE (no action, waiting)
        1: HOVER (ungrasped hand motion)
        2: GRASP (grasping object)
        3: CARRY (moving with object)
        4: RELEASE (releasing object)

    Note: Treats each frame independently. Use ActionClassifierLSTM for sequential modeling.
    """

    def __init__(self, embedding_dim: int = 1024, num_actions: int = 5, dropout: float = 0.2):
        """Initialize classifier.

        Args:
            embedding_dim: Input embedding dimension (default 1024 for V-JEPA)
            num_actions: Number of classifier outputs (default 5)
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
            x: (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim)
               If 3D, reshapes to 2D, processes, then reshapes back.

        Returns:
            (batch_size, num_actions) or (batch_size, seq_len, num_actions) logits
        """
        input_shape = x.shape
        is_3d = len(input_shape) == 3

        if is_3d:
            batch_size, seq_len, embedding_dim = input_shape
            x = x.reshape(batch_size * seq_len, embedding_dim)

        logits = self.model(x)

        if is_3d:
            logits = logits.reshape(batch_size, seq_len, -1)

        return logits


class ActionClassifierLSTM(nn.Module):
    """LSTM for sequential action classification from embeddings.

    Architecture:
        Input (1024) → Bidirectional LSTM (2 layers, 256 hidden)
        → Per-frame classifier → Output (5 actions)

    Action classes:
        0: IDLE (no action, waiting)
        1: HOVER (ungrasped hand motion)
        2: GRASP (grasping object)
        3: CARRY (moving with object)
        4: RELEASE (releasing object)

    Learns temporal patterns and action transitions:
    - Models IDLE → HOVER → GRASP → CARRY → RELEASE sequence
    - Uses bidirectional context (past and future frames)
    - Better for continuous action sequences
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        hidden_dim: int = 256,
        num_actions: int = 5,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        """Initialize LSTM classifier.

        Args:
            embedding_dim: Input embedding dimension (1024 for V-JEPA)
            hidden_dim: LSTM hidden state dimension (256)
            num_actions: Number of classifier outputs (default 5)
            num_layers: Number of stacked LSTM layers (2)
            bidirectional: Use bidirectional LSTM (True)
            dropout: Dropout rate (0.3)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        """Forward pass through LSTM.

        Args:
            x: (batch_size, seq_len, embedding_dim)
               Sequences of embeddings

        Returns:
            (batch_size, seq_len, num_actions) logits
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        logits = self.classifier(lstm_out)
        return logits


class ActionClassifier:
    """Wrapper for training and inference with both MLP and LSTM models."""

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_actions: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        model_type: str = "lstm",
    ):
        """Initialize classifier.

        Args:
            embedding_dim: Input embedding dimension
            num_actions: Number of classifier outputs. The deployment default
                is IDLE, HOVER, GRASP, CARRY, RELEASE.
            device: "cuda" or "cpu"
            model_type: "lstm" (recommended) or "mlp" (baseline)
        """
        self.device = device
        self.model_type = model_type
        self.num_actions = num_actions

        if model_type == "lstm":
            self.model = ActionClassifierLSTM(embedding_dim, num_actions=num_actions).to(device)
        elif model_type == "mlp":
            self.model = ActionClassifierModel(embedding_dim, num_actions).to(device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
                For MLP: yields (batch_X, batch_y) with shapes (batch, 1024) and (batch,)
                For LSTM: yields (batch_X, batch_y) with shapes (batch, seq_len, 1024) and (batch, seq_len)

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            logits = self.model(batch_X)

            # Reshape for loss computation if needed
            if self.model_type == "lstm":
                batch_size, seq_len, num_actions = logits.shape
                logits = logits.reshape(-1, num_actions)
                batch_y = batch_y.reshape(-1)

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

                if self.model_type == "lstm":
                    batch_size, seq_len, num_actions = logits.shape
                    logits = logits.reshape(-1, num_actions)
                    batch_y_flat = batch_y.reshape(-1)
                else:
                    batch_y_flat = batch_y

                loss = self.loss_fn(logits, batch_y_flat)
                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item()
                correct += (preds == batch_y_flat).sum().item()
                total += batch_y_flat.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        return accuracy, avg_loss

    def predict_probabilities(
        self, embeddings: np.ndarray, context_window: Optional[int] = None
    ) -> np.ndarray:
        """Return the complete softmax distribution for every embedding.

        Args:
            embeddings: (num_frames, embedding_dim) array
            context_window: For LSTM only. If specified, uses a sliding window approach
                with context_window frames before and after each frame.
                E.g., context_window=32 means 32 frames before + current + 32 frames after.
                If None, uses full sequence (recommended for batch processing).

        Returns:
            (num_frames, num_actions) array. Columns retain training/catalog order.
        """
        if self.model_type == "lstm" and context_window is not None:
            return self._predict_probabilities_with_context_window(embeddings, context_window)

        self.model.eval()

        with torch.no_grad():
            X = torch.from_numpy(embeddings).float().to(self.device)

            if self.model_type == "lstm":
                X = X.unsqueeze(0)  # Add batch dimension: (1, num_frames, embedding_dim)
                logits = self.model(X)  # (1, num_frames, num_actions)
                logits = logits.squeeze(0)  # Remove batch dimension: (num_frames, num_actions)
            else:
                logits = self.model(X)  # (num_frames, num_actions)

            probs = torch.softmax(logits, dim=-1)  # (num_frames, num_actions)

        return probs.cpu().numpy()

    def _predict_probabilities_with_context_window(
        self, embeddings: np.ndarray, context_window: int
    ) -> np.ndarray:
        """Predict using sliding context window for LSTM.

        Args:
            embeddings: (num_frames, embedding_dim) array
            context_window: Number of frames before/after to include

        Returns:
            (num_frames, num_actions) probability array
        """
        num_frames = embeddings.shape[0]
        num_actions = self.num_actions
        probabilities = np.zeros((num_frames, num_actions), dtype=np.float32)

        self.model.eval()

        with torch.no_grad():
            for frame_idx in range(num_frames):
                # Define window boundaries
                start_idx = max(0, frame_idx - context_window)
                end_idx = min(num_frames, frame_idx + context_window + 1)

                # Extract window
                window = embeddings[start_idx:end_idx]  # (window_size, embedding_dim)

                # Pad if at boundaries
                if start_idx == 0 and end_idx == num_frames:
                    # Full window available (middle of sequence)
                    pass
                elif start_idx == 0:
                    # Pad at beginning
                    pad_size = context_window - frame_idx
                    pad = np.zeros((pad_size, embeddings.shape[1]))
                    window = np.vstack([pad, window])
                elif end_idx == num_frames:
                    # Pad at end
                    pad_size = context_window - (num_frames - frame_idx - 1)
                    pad = np.zeros((pad_size, embeddings.shape[1]))
                    window = np.vstack([window, pad])

                # Forward pass
                X = torch.from_numpy(window).float().to(self.device)
                X = X.unsqueeze(0)  # Add batch dimension: (1, window_size, embedding_dim)

                logits = self.model(X)  # (1, window_size, num_actions)
                logits = logits.squeeze(0)  # (window_size, num_actions)

                # Get prediction for center frame (current frame)
                center_idx = min(frame_idx - start_idx, context_window)
                frame_logits = logits[center_idx]  # (num_actions,)

                probs = torch.softmax(frame_logits, dim=-1)
                probabilities[frame_idx] = probs.cpu().numpy()

        return probabilities

    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return top-one predictions derived from the complete score matrix.

        This compatibility helper is appropriate for display and metrics only.
        Robot-facing inference must use :meth:`predict_probabilities` so graph
        post-processing receives every catalog score.
        """
        probabilities = self.predict_probabilities(embeddings)
        actions = np.argmax(probabilities, axis=-1)
        confidences = np.max(probabilities, axis=-1)
        return actions, confidences

    def save(self, path: str, catalog: Optional["SkillCatalog"] = None) -> None:
        """Save model weights.

        Args:
            path: Path to save weights
            catalog: Active classifier vocabulary. When supplied, its ordered
                labels and fingerprint are embedded in the checkpoint.
        """
        state_dict = self.model.state_dict()
        if catalog is None:
            torch.save(state_dict, path)
        else:
            if catalog.class_count != self.num_actions:
                raise ValueError(
                    "Classifier output count does not match the supplied skill catalog"
                )
            torch.save(
                {
                    "format": "mimic.action_classifier.v2",
                    "model_state_dict": state_dict,
                    "model_type": self.model_type,
                    "num_actions": self.num_actions,
                    "catalog": {
                        "schema_version": catalog.schema_version,
                        "fingerprint": catalog.fingerprint,
                        "labels": list(catalog.labels),
                    },
                },
                path,
            )
        logger.info(f"✓ Model saved to {path}")

    def load(self, path: str, catalog: Optional["SkillCatalog"] = None) -> None:
        """Load model weights.

        Args:
            path: Path to load weights from
            catalog: Expected active vocabulary. Supplying it requires a v2
                checkpoint with exactly matching catalog provenance.
        """
        checkpoint = torch.load(path, map_location=self.device)
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get("format") == "mimic.action_classifier.v2"
        ):
            if checkpoint.get("model_type") != self.model_type:
                raise ValueError("Checkpoint model type does not match the classifier")
            if checkpoint.get("num_actions") != self.num_actions:
                raise ValueError("Checkpoint output count does not match the classifier")
            if catalog is not None:
                metadata = checkpoint.get("catalog") or {}
                if (
                    metadata.get("fingerprint") != catalog.fingerprint
                    or tuple(metadata.get("labels", ())) != catalog.labels
                ):
                    raise ValueError("Checkpoint skill catalog does not match the active catalog")
            state_dict = checkpoint["model_state_dict"]
        else:
            if catalog is not None:
                raise ValueError(
                    "Legacy checkpoint has no skill-catalog provenance; retrain or migrate it "
                    "before robot-facing inference"
                )
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        logger.info(f"✓ Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    """
    Example of training action classifier on embeddings + audio labels.

    Before running this, you need:
    1. Embeddings: data/embeddings/*.npy (1891 frames × 1024 dims) ✅
    2. Labels: action labels extracted from audio narration

    Uses LSTM model (recommended) for sequential action classification.
    """
    import json
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    print("=" * 70)
    print("ACTION CLASSIFIER: LSTM Training Example")
    print("=" * 70)

    # Step 1: Load embeddings (per-video)
    print("\n1. Loading embeddings (per-video)...")
    embeddings_dir = Path("data/embeddings")

    video_data = []  # List of (embeddings, labels) tuples

    for npy_file in sorted(embeddings_dir.glob("*.npy")):
        if "_meta" in npy_file.name:
            continue

        emb = np.load(npy_file)
        print(f"   Loaded {npy_file.name}: {emb.shape}")

        num_frames = emb.shape[0]
        dummy_labels = np.random.randint(0, len(DEFAULT_ACTION_NAMES), num_frames)
        video_data.append((emb, dummy_labels))

    print(f"\n   Total videos: {len(video_data)}")
    print(f"   Total frames: {sum(v[0].shape[0] for v in video_data)}")

    # Step 2: Split videos (80/20 train/val, stratified by video)
    print("\n2. Splitting data (80/20 train/val by video)...")
    split_idx = int(len(video_data) * 0.8)
    train_videos = video_data[:split_idx]
    val_videos = video_data[split_idx:]
    print(f"   Train: {len(train_videos)} videos | Val: {len(val_videos)} videos")

    # Step 3: Create data loaders for LSTM
    print("\n3. Creating LSTM data loaders...")

    def pad_sequence_batch(sequences, pad_value=0):
        """Pad sequences to same length."""
        max_len = max(len(seq) for seq in sequences)
        padded = []
        masks = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(np.pad(seq, ((0, pad_len), (0, 0)), constant_values=pad_value))
            masks.append(np.concatenate([np.ones(len(seq)), np.zeros(pad_len)]))
        return np.stack(padded), np.stack(masks)

    def create_lstm_dataloader(videos, batch_size=2):
        """Create dataloader for LSTM from video sequences."""
        embeddings_list = [v[0] for v in videos]
        labels_list = [v[1] for v in videos]

        X_padded, _ = pad_sequence_batch(embeddings_list)
        y_padded, _ = pad_sequence_batch(
            [l.reshape(-1, 1) for l in labels_list],
            pad_value=-1,
        )
        y_padded = y_padded.squeeze(-1).astype(np.int64)

        X_t = torch.from_numpy(X_padded).float()
        y_t = torch.from_numpy(y_padded).long()

        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_lstm_dataloader(train_videos, batch_size=2)
    val_loader = create_lstm_dataloader(val_videos, batch_size=2)

    # Step 4: Train LSTM classifier
    print("\n4. Training LSTM classifier...")
    classifier = ActionClassifier(
        embedding_dim=1024,
        num_actions=len(DEFAULT_ACTION_NAMES),
        model_type="lstm",  # Use LSTM instead of MLP
    )

    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = classifier.train_epoch(train_loader)
        val_acc, val_loss = classifier.evaluate(val_loader)

        if (epoch + 1) % 5 == 0:
            print(
                f"   Epoch {epoch+1:2d}/{num_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Val acc: {val_acc:.3f} | "
                f"Val loss: {val_loss:.4f}"
            )

    # Step 5: Save model
    print("\n5. Saving LSTM model...")
    classifier.save("action_classifier_lstm.pt")

    # Step 6: Test inference on a full video
    print("\n6. Testing inference on a video...")
    test_video_embeddings = val_videos[0][0]  # (num_frames, 1024)
    actions, confidences = classifier.predict(test_video_embeddings)
    print(f"   Predicted actions for {len(actions)} frames:")
    action_names = DEFAULT_ACTION_NAMES
    for i in range(min(5, len(actions))):
        action_name = action_names[actions[i]]
        print(f"     Frame {i}: {action_name} (confidence: {confidences[i]:.3f})")
    print(f"   ... ({len(actions) - 5} more frames)")

    print("\n" + "=" * 70)
    print("✓ LSTM Training complete!")
    print("=" * 70)
    print("\nKey differences from MLP:")
    print("  - Processes entire video sequences at once")
    print("  - Learns temporal patterns (IDLE → HOVER → GRASP → CARRY → RELEASE)")
    print("  - Uses bidirectional context (past and future frames)")
    print("  - Better handling of action transitions")
    print("=" * 70)
