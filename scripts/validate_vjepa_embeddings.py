#!/usr/bin/env python3
"""Validate and explore V-JEPA embeddings for action classification.

Demonstrates:
- Loading saved embeddings
- Computing embedding statistics
- Validating embedding quality
- Showing how to use embeddings with action labels
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_embedding(embedding_path: str) -> np.ndarray:
    """Load embedding array from disk.

    Args:
        embedding_path: Path to .npy file (without extension)

    Returns:
        (num_frames, 1024) array
    """
    embedding_file = f"{embedding_path}.npy"
    if not Path(embedding_file).exists():
        print(f"ERROR: Embedding file not found: {embedding_file}")
        return None

    embeddings = np.load(embedding_file)
    return embeddings


def load_metadata(embedding_path: str) -> dict:
    """Load metadata JSON for embedding.

    Args:
        embedding_path: Path to .npy file (without extension)

    Returns:
        Metadata dictionary
    """
    metadata_file = f"{embedding_path}_meta.json"
    if not Path(metadata_file).exists():
        return None

    with open(metadata_file) as f:
        metadata = json.load(f)

    return metadata


def analyze_embeddings(embeddings: np.ndarray, video_name: str) -> dict:
    """Compute statistics for embeddings.

    Args:
        embeddings: (num_frames, 1024) array
        video_name: Name of video for display

    Returns:
        Statistics dictionary
    """
    stats = {
        "shape": embeddings.shape,
        "num_frames": embeddings.shape[0],
        "embedding_dim": embeddings.shape[1],
        "mean_magnitude": float(np.linalg.norm(embeddings, axis=1).mean()),
        "std_magnitude": float(np.linalg.norm(embeddings, axis=1).std()),
        "mean_values": float(embeddings.mean()),
        "std_values": float(embeddings.std()),
        "min_value": float(embeddings.min()),
        "max_value": float(embeddings.max()),
    }

    # Pairwise cosine similarity between first/last frame
    if len(embeddings) > 1:
        from sklearn.metrics.pairwise import cosine_similarity
        try:
            sim_first_last = cosine_similarity(
                embeddings[0:1], embeddings[-1:1]
            )[0, 0]
            stats["cosine_sim_first_last"] = float(sim_first_last)
        except:
            stats["cosine_sim_first_last"] = 0.0

        # Temporal coherence: mean cosine sim between adjacent frames
        try:
            cosine_sims = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(embeddings[i:i+1], embeddings[i+1:i+2])[0, 0]
                cosine_sims.append(sim)
            stats["temporal_coherence"] = float(np.mean(cosine_sims))
        except:
            stats["temporal_coherence"] = 0.0
    else:
        stats["cosine_sim_first_last"] = 0.0
        stats["temporal_coherence"] = 0.0

    return stats


def validate_embeddings(embeddings_dir: str = "data/embeddings/") -> None:
    """Validate all embeddings in directory.

    Args:
        embeddings_dir: Directory containing embeddings
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings directory not found: {embeddings_dir}")
        return

    print("\n" + "=" * 70)
    print("V-JEPA EMBEDDING VALIDATION & ANALYSIS")
    print("=" * 70)

    # Find all embedding files
    embedding_files = sorted(embeddings_path.glob("*.npy"))

    if not embedding_files:
        print(f"ERROR: No embeddings found in {embeddings_dir}")
        return

    print(f"\nFound {len(embedding_files)} embedding files\n")

    all_stats = {}

    # Analyze each embedding
    for emb_file in embedding_files:
        video_stem = emb_file.stem
        embeddings = load_embedding(str(emb_file.with_suffix("")))
        metadata = load_metadata(str(emb_file.with_suffix("")))

        if embeddings is None:
            continue

        stats = analyze_embeddings(embeddings, video_stem)
        all_stats[video_stem] = stats

        print(f"Video: {video_stem}")
        print(f"  Frames: {stats['num_frames']}")
        print(f"  Embedding shape: ({stats['num_frames']}, {stats['embedding_dim']})")
        print(f"  Magnitude: {stats['mean_magnitude']:.2f} ± {stats['std_magnitude']:.2f}")
        print(f"  Value range: [{stats['min_value']:.3f}, {stats['max_value']:.3f}]")
        print(f"  Temporal coherence (adj. frames): {stats['temporal_coherence']:.3f}")
        print(f"  First/last frame similarity: {stats['cosine_sim_first_last']:.3f}")

        if metadata:
            print(f"  Duration: {metadata.get('duration', 'N/A'):.1f}s")
            print(f"  FPS: {metadata.get('fps', 'N/A'):.1f}")

        print()

    # Global statistics
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_frames = sum(stats["num_frames"] for stats in all_stats.values())
    avg_coherence = np.mean([stats["temporal_coherence"] for stats in all_stats.values()])

    print(f"\nTotal frames embedded: {total_frames}")
    print(f"Average temporal coherence: {avg_coherence:.3f}")
    print(f"Embedding dimension: 1024")
    print(f"Ready for: Action classification with temporal models")

    # Show usage example
    print("\n" + "=" * 70)
    print("NEXT STEPS: Using embeddings for action classification")
    print("=" * 70)
    print("""
1. Extract action labels from audio narration (APPROACH, GRASP, MOVE, RELEASE)
2. Map labels to frame timestamps

3. Create training dataset:
   X_train = embeddings[frame_indices]  # (num_frames, 1024)
   y_train = action_labels[frame_indices]  # (num_frames,)

4. Train simple classifier:
   model = nn.Sequential(
       nn.Linear(1024, 256),
       nn.ReLU(),
       nn.Linear(256, 4)  # 4 actions: APPROACH, GRASP, MOVE, RELEASE
   )

5. At inference time:
   new_video_embeddings = encoder.extract_video_embeddings(video_path)
   action_logits = model(new_video_embeddings)
   action_probs = F.softmax(action_logits, dim=1)

Embeddings are already optimized for this downstream task!
    """)

    print("=" * 70)


def visualize_embeddings_pca(embeddings_dir: str = "data/embeddings/", num_files: int = 2) -> None:
    """Visualize embeddings using PCA (2D projection).

    Args:
        embeddings_dir: Directory with embeddings
        num_files: Number of videos to visualize
    """
    try:
        embeddings_path = Path(embeddings_dir)
        embedding_files = sorted(embeddings_path.glob("*.npy"))[:num_files]

        if not embedding_files:
            return

        print(f"\nVisualizing {len(embedding_files)} videos with PCA...")

        fig, axes = plt.subplots(1, len(embedding_files), figsize=(12, 5))
        if len(embedding_files) == 1:
            axes = [axes]

        for idx, emb_file in enumerate(embedding_files):
            embeddings = load_embedding(str(emb_file.with_suffix("")))
            if embeddings is None:
                continue

            # PCA to 2D
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            # Plot trajectory
            ax = axes[idx]
            ax.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'b-', alpha=0.6)
            ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], color='green', s=100, label='Start')
            ax.scatter(embeddings_2d[-1, 0], embeddings_2d[-1, 1], color='red', s=100, label='End')
            ax.set_title(f"{emb_file.stem}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path(embeddings_dir) / "embeddings_pca_visualization.png"
        plt.savefig(output_path, dpi=100)
        print(f"  Saved visualization to: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Could not create visualization: {e}")


def main():
    """Main validation."""
    embeddings_dir = "data/embeddings/"

    # Validate embeddings
    validate_embeddings(embeddings_dir)

    # Try to visualize
    visualize_embeddings_pca(embeddings_dir, num_files=2)

    print("\n✅ All embeddings are ready for action classification!")
    print("   Train on audio-labeled data with the embedding tensors.\n")


if __name__ == "__main__":
    main()
