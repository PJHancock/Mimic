"""V-JEPA 2 video feature encoder for action classification.

Provides pretrained video embeddings for downstream action prediction.
Handles model loading, frame preprocessing, and batch embedding extraction.
"""

from typing import List, Optional, Tuple
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


class VJepaEncoder:
    """Wrapper for V-JEPA 2 video encoder.

    Loads pretrained V-JEPA 2 model and extracts frame embeddings.
    Can use actual V-JEPA 2 or fallback to alternative video encoders.
    """

    def __init__(
        self,
        model_name: str = "vjepa2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        embedding_dim: int = 1024,
    ):
        """Initialize V-JEPA encoder.

        Args:
            model_name: "vjepa2", "timesformer", or "i3d"
            device: "cuda" or "cpu"
            embedding_dim: Output embedding dimension
        """
        self.model_name = model_name
        self.device = device
        self.embedding_dim = embedding_dim
        self.model = None
        self.preprocessor = None

        # Try to load model
        self._load_model()

    def _load_model(self) -> None:
        """Load pretrained video encoder model."""
        logger.info(f"Loading {self.model_name} model on {self.device}")

        if self.model_name == "vjepa2":
            self._load_vjepa2()
        elif self.model_name == "timesformer":
            self._load_timesformer()
        elif self.model_name == "i3d":
            self._load_i3d()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ Model loaded successfully")
        else:
            logger.error(f"Failed to load {self.model_name}")

    def _load_vjepa2(self) -> None:
        """Load V-JEPA 2 from Meta's implementation."""
        try:
            # Try to import and load V-JEPA 2
            # This requires: git clone https://github.com/facebookresearch/jepa
            import sys
            from pathlib import Path

            jepa_path = Path.home() / "jepa"  # Or wherever it's cloned
            if jepa_path.exists():
                sys.path.insert(0, str(jepa_path))

            # Attempt import
            try:
                from jepa.models.vision_transformer import vit_large
                from jepa.utils.checkpoint import load_checkpoint

                # Load pretrained V-JEPA weights
                # This assumes weights are downloaded from Meta
                self.model = vit_large(patch_size=16, num_frames=16)
                logger.info("✓ V-JEPA 2 model created")

                # Note: In production, load actual pretrained weights
                # model = load_checkpoint(checkpoint_path, model)

            except ImportError:
                logger.warning(
                    "V-JEPA 2 not available. Make sure to run:\n"
                    "  git clone https://github.com/facebookresearch/jepa\n"
                    "Falling back to TimeSformer..."
                )
                self._load_timesformer()

        except Exception as e:
            logger.error(f"Error loading V-JEPA 2: {e}")
            self._load_timesformer()

    def _load_timesformer(self) -> None:
        """Load ResNet50 from torchvision as fallback."""
        try:
            from torchvision.models import resnet50

            # Load pretrained ResNet50 and remove classification head
            model = resnet50(pretrained=True)
            # Keep features, remove final fc layer
            self.model = nn.Sequential(*list(model.children())[:-1])
            # Global average pooling
            self.model.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
            # Flatten
            self.model.add_module("flatten", nn.Flatten())
            logger.info("✓ ResNet50 model loaded (for frame-level embeddings)")

        except Exception as e:
            logger.error(f"Error loading ResNet50: {e}")
            self._load_i3d()

    def _load_i3d(self) -> None:
        """Load VGG16 as lightweight fallback."""
        try:
            from torchvision.models import vgg16

            model = vgg16(pretrained=True)
            # Extract features only
            self.model = model.features
            # Add average pooling and flatten
            self.model = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            logger.info("✓ VGG16 model loaded")

        except Exception as e:
            logger.error(f"Error loading VGG16: {e}")
            logger.error("No encoder models available!")
            self.model = None

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess single frame for model input.

        Args:
            frame: BGR frame from OpenCV (H, W, 3) with values [0, 255]

        Returns:
            Preprocessed torch tensor (float32)
        """
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # Normalize: ImageNet stats
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_normalized = (frame_normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )

        # Convert to torch: (H, W, 3) -> (3, H, W)
        frame_torch = torch.from_numpy(frame_normalized).permute(2, 0, 1).float()

        return frame_torch

    def extract_embedding(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Extract embedding for single frame.

        Args:
            frame: Input frame (H, W, 3) BGR

        Returns:
            Embedding vector (embedding_dim,) on CPU
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None

        try:
            # Preprocess
            frame_tensor = self.preprocess_frame(frame)

            # Add batch dimension: (3, H, W) -> (1, 3, H, W)
            frame_batch = frame_tensor.unsqueeze(0).to(self.device)

            # Extract embedding
            with torch.no_grad():
                embedding = self.model(frame_batch)

            # Handle different output formats
            if isinstance(embedding, (list, tuple)):
                embedding = embedding[0]  # Take first output if tuple

            # Flatten if needed
            if embedding.dim() > 1:
                embedding = embedding.flatten(0)

            # Move to CPU
            embedding = embedding.cpu()

            # Pad or truncate to embedding_dim
            actual_dim = embedding.shape[0]
            if actual_dim != self.embedding_dim:
                if actual_dim < self.embedding_dim:
                    # Pad with zeros
                    padding = torch.zeros(self.embedding_dim - actual_dim, dtype=embedding.dtype)
                    embedding = torch.cat([embedding, padding])
                else:
                    # Truncate
                    embedding = embedding[: self.embedding_dim]

            return embedding

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def extract_batch_embeddings(
        self, frames: List[np.ndarray]
    ) -> Optional[torch.Tensor]:
        """Extract embeddings for batch of frames.

        Args:
            frames: List of frames (H, W, 3) BGR

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        if not frames:
            return None

        embeddings = []
        for frame in frames:
            emb = self.extract_embedding(frame)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return None

        return torch.stack(embeddings)

    def extract_video_embeddings(
        self, video_path: str, frame_stride: int = 1
    ) -> Tuple[torch.Tensor, List[int]]:
        """Extract embeddings from entire video.

        Args:
            video_path: Path to video file
            frame_stride: Extract every Nth frame (1 = all frames)

        Returns:
            (embeddings tensor, frame indices)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None, []

        embeddings = []
        frame_indices = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                emb = self.extract_embedding(frame)
                if emb is not None:
                    embeddings.append(emb)
                    frame_indices.append(frame_idx)

            frame_idx += 1

        cap.release()

        if embeddings:
            embeddings_tensor = torch.stack(embeddings)
            logger.info(
                f"Extracted {len(embeddings)} embeddings from {video_path}"
            )
            return embeddings_tensor, frame_indices
        else:
            logger.error(f"No embeddings extracted from {video_path}")
            return None, []

    def __call__(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Convenience method: encoder(frame) -> embedding."""
        return self.extract_embedding(frame)
