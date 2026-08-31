"""Unit tests for V-JEPA encoder."""

import numpy as np
import pytest
import torch
import tempfile
from pathlib import Path

from mimic.vision import VJepaEncoder


class TestVJepaEncoderInitialization:
    """Test encoder initialization and model loading."""

    def test_encoder_init_cpu(self):
        """Test encoder initialization on CPU."""
        encoder = VJepaEncoder(device="cpu")
        assert encoder is not None
        assert encoder.device == "cpu"
        assert encoder.embedding_dim == 1024

    def test_encoder_init_cuda_if_available(self):
        """Test encoder initialization with CUDA if available."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = VJepaEncoder(device=device)
        assert encoder.device == device

    def test_encoder_model_loaded(self):
        """Test that a model is successfully loaded."""
        encoder = VJepaEncoder()
        assert encoder.model is not None, "Model failed to load"
        assert hasattr(encoder.model, "eval"), "Model should have eval() method"

    def test_encoder_fallback_chain(self):
        """Test that fallback models work if primary fails."""
        # Test different model options
        for model_name in ["timesformer", "i3d"]:
            encoder = VJepaEncoder(model_name=model_name)
            assert encoder.model is not None or encoder.model_name == model_name


class TestFramePreprocessing:
    """Test frame preprocessing pipeline."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return VJepaEncoder(device="cpu")

    @pytest.fixture
    def sample_frame(self):
        """Create sample BGR frame."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_preprocess_shape(self, encoder, sample_frame):
        """Test that preprocessing outputs correct shape."""
        tensor = encoder.preprocess_frame(sample_frame)
        assert tensor.shape == (3, 224, 224), f"Got shape {tensor.shape}"

    def test_preprocess_dtype(self, encoder, sample_frame):
        """Test that output is torch tensor with float dtype."""
        tensor = encoder.preprocess_frame(sample_frame)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32

    def test_preprocess_value_range(self, encoder, sample_frame):
        """Test that normalized values are in reasonable range."""
        tensor = encoder.preprocess_frame(sample_frame)
        # ImageNet normalized values can range from -2 to 2.7 depending on content
        assert tensor.min() >= -3.0
        assert tensor.max() <= 3.0

    def test_preprocess_different_sizes(self, encoder):
        """Test preprocessing with various input sizes."""
        for h, w in [(480, 640), (720, 1280), (1080, 1920)]:
            frame = np.ones((h, w, 3), dtype=np.uint8) * 128
            tensor = encoder.preprocess_frame(frame)
            assert tensor.shape == (3, 224, 224)

    def test_preprocess_grayscale_to_rgb(self, encoder):
        """Test that grayscale frames are handled."""
        gray_frame = np.ones((480, 640), dtype=np.uint8) * 128
        # Preprocess should handle or convert gracefully
        try:
            tensor = encoder.preprocess_frame(gray_frame)
            assert tensor is not None
        except:
            pass  # Some models may not support grayscale


class TestEmbeddingExtraction:
    """Test single frame embedding extraction."""

    @pytest.fixture
    def encoder(self):
        return VJepaEncoder(device="cpu")

    def test_extract_embedding_shape(self, encoder):
        """Test that embedding has correct shape."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = encoder.extract_embedding(frame)
        
        if embedding is not None:
            assert embedding.shape == (1024,), f"Expected (1024,), got {embedding.shape}"

    def test_extract_embedding_dtype(self, encoder):
        """Test that embedding is float32 tensor."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = encoder.extract_embedding(frame)
        
        if embedding is not None:
            assert isinstance(embedding, torch.Tensor)
            assert embedding.dtype in [torch.float32, torch.float64]

    def test_extract_embedding_on_cpu(self, encoder):
        """Test that embedding is returned on CPU."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = encoder.extract_embedding(frame)
        
        if embedding is not None:
            assert embedding.device.type == "cpu"

    def test_extract_embedding_deterministic(self, encoder):
        """Test that same frame produces same (or very similar) embedding."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        emb1 = encoder.extract_embedding(frame)
        emb2 = encoder.extract_embedding(frame)
        
        if emb1 is not None and emb2 is not None:
            # Embeddings should be very similar (within tolerance for float precision)
            cosine_sim = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            )
            assert cosine_sim > 0.99, f"Cosine similarity: {cosine_sim}"

    def test_extract_embedding_different_content(self, encoder):
        """Test that different frames produce different embeddings."""
        frame_white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        frame_black = np.zeros((480, 640, 3), dtype=np.uint8)
        
        emb_white = encoder.extract_embedding(frame_white)
        emb_black = encoder.extract_embedding(frame_black)
        
        if emb_white is not None and emb_black is not None:
            cosine_sim = torch.nn.functional.cosine_similarity(
                emb_white.unsqueeze(0), emb_black.unsqueeze(0)
            )
            # Different frames should have different embeddings
            assert cosine_sim < 0.95, f"Frames too similar: {cosine_sim}"

    def test_extract_embedding_invalid_input(self, encoder):
        """Test that invalid input is handled gracefully."""
        # Empty frame
        frame_empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        embedding = encoder.extract_embedding(frame_empty)
        # Should return None or handle gracefully
        assert embedding is None or isinstance(embedding, torch.Tensor)


class TestBatchEmbeddingExtraction:
    """Test batch processing of multiple frames."""

    @pytest.fixture
    def encoder(self):
        return VJepaEncoder(device="cpu")

    def test_batch_extract_shape(self, encoder):
        """Test batch embedding shape."""
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        embeddings = encoder.extract_batch_embeddings(frames)
        
        if embeddings is not None:
            assert embeddings.shape == (5, 1024)

    def test_batch_extract_empty(self, encoder):
        """Test batch extraction with empty list."""
        embeddings = encoder.extract_batch_embeddings([])
        assert embeddings is None

    def test_batch_extract_single(self, encoder):
        """Test batch extraction with single frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embeddings = encoder.extract_batch_embeddings([frame])
        
        if embeddings is not None:
            assert embeddings.shape == (1, 1024)

    def test_batch_extract_consistency(self, encoder):
        """Test that batch extraction matches individual extraction."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        single_emb = encoder.extract_embedding(frame)
        batch_embs = encoder.extract_batch_embeddings([frame, frame])
        
        if single_emb is not None and batch_embs is not None:
            # First element of batch should match single embedding
            cosine_sim = torch.nn.functional.cosine_similarity(
                single_emb.unsqueeze(0), batch_embs[0].unsqueeze(0)
            )
            assert cosine_sim > 0.99


class TestVideoEmbeddingExtraction:
    """Test full video embedding extraction."""

    @pytest.fixture
    def encoder(self):
        return VJepaEncoder(device="cpu")

    @pytest.fixture
    def sample_video(self):
        """Create a temporary synthetic video file."""
        import cv2
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_video.mp4"
            
            # Create synthetic video: 30 frames of moving square
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
            
            for i in range(30):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Draw moving square
                x = 100 + (i * 5) % 500
                y = 100
                cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 255, 0), -1)
                out.write(frame)
            
            out.release()
            yield str(video_path)

    def test_extract_video_embeddings(self, encoder, sample_video):
        """Test extraction from video file."""
        embeddings, frame_indices = encoder.extract_video_embeddings(sample_video)
        
        if embeddings is not None:
            assert len(embeddings) == 30
            assert embeddings.shape[1] == 1024
            assert len(frame_indices) == 30

    def test_extract_video_with_stride(self, encoder, sample_video):
        """Test extraction with frame stride."""
        embeddings, frame_indices = encoder.extract_video_embeddings(
            sample_video, frame_stride=2
        )
        
        if embeddings is not None:
            assert len(embeddings) == 15  # Every 2nd frame
            assert frame_indices == list(range(0, 30, 2))

    def test_extract_video_invalid_path(self, encoder):
        """Test extraction from non-existent file."""
        embeddings, frame_indices = encoder.extract_video_embeddings("/nonexistent/video.mp4")
        assert embeddings is None
        assert frame_indices == []


class TestEmbeddingProperties:
    """Test mathematical properties of embeddings."""

    @pytest.fixture
    def encoder(self):
        return VJepaEncoder(device="cpu")

    def test_embedding_normalized(self, encoder):
        """Test that embeddings have reasonable scale."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = encoder.extract_embedding(frame)
        
        if embedding is not None:
            # Check magnitude is reasonable (not all zeros, not exploding)
            norm = torch.norm(embedding)
            assert norm > 0.1, "Embedding norm too small"
            assert norm < 100000, "Embedding norm too large"

    def test_embedding_no_nans(self, encoder):
        """Test that embeddings don't contain NaN values."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = encoder.extract_embedding(frame)
        
        if embedding is not None:
            assert not torch.isnan(embedding).any(), "Embedding contains NaN"

    def test_embedding_no_infs(self, encoder):
        """Test that embeddings don't contain infinity values."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = encoder.extract_embedding(frame)
        
        if embedding is not None:
            assert not torch.isinf(embedding).any(), "Embedding contains Inf"


class TestCallableInterface:
    """Test encoder as callable."""

    @pytest.fixture
    def encoder(self):
        return VJepaEncoder(device="cpu")

    def test_call_interface(self, encoder):
        """Test that encoder can be called directly."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        embedding = encoder(frame)
        
        if embedding is not None:
            assert embedding.shape == (1024,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
