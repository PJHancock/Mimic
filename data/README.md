# Data Directory

Organized storage for all project data.

## Structure

- **raw/** — Raw video files from human demonstrations (`.gitignore`d)
- **processed/** — Preprocessed video frames and cleaned data
- **embeddings/** — V-JEPA cached embeddings (`.gitignore`d)
- **annotations/** — Manual labels, timestamps, speech transcriptions (JSON/CSV)
- **splits/** — Train/val/test split definitions (ensuring no demonstration leakage)

## Guidelines

1. **Raw videos** should not be committed to git (too large). Use git-lfs or external storage.
2. **Embeddings** are cached to `data/embeddings/` with naming: `demo_NNN_embeddings.pt`
3. **Annotations** use JSON with structure:
   ```json
   {
     "demo_id": "demo_001",
     "video_path": "raw/demo_001.mp4",
     "speech_transcript": [
       {"start": 1.8, "end": 3.2, "text": "grab the ball", "phase": "APPROACH"}
     ],
     "manual_labels": [
       {"frame": 50, "phase": "APPROACH", "confidence": 0.95}
     ]
   }
   ```
4. **Splits** prevent data leakage by keeping complete demonstrations in one split:
   ```yaml
   train: [demo_001, demo_002, ..., demo_040]
   val: [demo_041, demo_042, demo_043, demo_044, demo_045]
   test: [demo_046, demo_047, demo_048, demo_049, demo_050]
   ```

## Caching Strategy

- V-JEPA embeddings are computed once and cached for fast iteration on temporal model
- Tracking results (hand/object) can be cached if expensive
- Preprocessed frames can be cached if storage permits

## Access from Code

```python
from mimic.config import get_data_dir

data_dir = get_data_dir()
raw_videos = data_dir / "raw"
embeddings = data_dir / "embeddings"
```
