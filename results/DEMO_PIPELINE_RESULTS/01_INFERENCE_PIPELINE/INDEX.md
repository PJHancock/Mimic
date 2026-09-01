# Inference Pipeline Results

**Status**: ✅ Complete and Verified

## Overview

This folder contains all results from the video inference pipeline demonstration, showing the complete multi-stage processing:

1. **Object Tracking** → 2. **Embedding Extraction** → 3. **Action Prediction** → 4. **Visualization**

## Files in This Directory

### `IMG_2013_results.json`

Complete inference results for all 274 frames.

**Structure:**
```json
{
  "metadata": {
    "created": "2026-08-31T19:34:46.480394",
    "fps": 29.983585628305672,
    "total_frames": 274,
    "duration": 9.104981751824818
  },
  "per_frame": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "position": {
        "x": 849.5,
        "y": 678.0,
        "confidence": 0.8
      },
      "action": "IDLE",
      "action_confidence": 0.9608975648880005
    },
    ...
  ],
  "action_segments": [
    {
      "action": "IDLE",
      "start_frame": 0,
      "end_frame": 33,
      "start_time": 0.0,
      "end_time": 1.1006021897810219,
      "duration": 1.1006021897810219,
      "avg_confidence": 0.9192761252908146
    },
    ...
  ],
  "tracking_summary": {
    "total_positions": 274,
    "fps": 29.983585628305672
  }
}
```

**Usage:**
- Parse JSON in Python: `json.load(open('IMG_2013_results.json'))`
- Analyze per-frame predictions
- Compare tracking positions
- Validate action confidence scores

### `visualization/` Directory

Generated video visualizations for verification.

#### `sidebyside.mp4` (34 MB) - RECOMMENDED

**What to see:**
- **LEFT side**: Original video (what the camera captured)
- **RIGHT side**: Annotated video (predictions + tracking overlays)
- **Green circles**: Tracked object position
- **Text overlay**: Action label with confidence percentage
- **Timestamp**: Frame time in seconds

**Best for:**
- Quick visual verification
- Comparing predictions to actual video
- Presentation/documentation

**How to watch:**
```bash
open visualization/sidebyside.mp4  # macOS
vlc visualization/sidebyside.mp4    # Linux/Windows
```

#### `annotated.mp4` (17 MB) - DETAILED

**What to see:**
- Original video enhanced with annotations
- Green tracking circles following the object
- Action labels and confidence scores
- Timestamp counter
- Clean visualization for analysis

**Best for:**
- Detailed frame-by-frame review
- Analyzing tracking accuracy
- Examining confidence scores in detail

---

## Results Summary

### Video Specifications
- **Resolution**: 1920 × 1080 pixels (Full HD)
- **Frame Rate**: 29.98 FPS
- **Duration**: 9.1 seconds
- **Total Frames**: 274

### Tracking Performance
- **Frames Tracked**: 274/274 (100%)
- **Method**: HSV color detection (red object)
- **Average Confidence**: 0.80
- **Position Stability**: Smooth, consistent tracking

### Action Detection
| Action | Time Range | Duration | Confidence | Frames |
|--------|-----------|----------|------------|--------|
| IDLE   | 0.00-1.10s | 1.10s   | 91.9%     | 0-33   |
| CARRY  | 1.13-9.10s | 7.97s   | 86.3%     | 34-273 |

### Per-Frame Data Sample
```
Frame 0:   IDLE   (96.1% conf) @ 0.000s, pos=(849.5, 678.0), track_conf=0.80
Frame 34:  CARRY  (87.2% conf) @ 1.134s, pos=(845.2, 675.3), track_conf=0.80
Frame 273: CARRY  (85.1% conf) @ 9.105s, pos=(832.1, 670.8), track_conf=0.80
```

---

## Pipeline Stages

### Stage 1: Object Tracking
- **Input**: Video frames
- **Method**: HSV color detection (Red Solo cup detection)
- **Output**: (x, y, confidence) per frame
- **Result**: 274 position samples

### Stage 2: Embedding Extraction
- **Input**: Video frames
- **Model**: V-JEPA (Vision Transformer - timesformer)
- **Output**: 1024-dimensional embeddings
- **Result**: 274 embeddings

### Stage 3: Action Prediction
- **Input**: Embeddings
- **Model**: LSTM Classifier (action_classifier_lstm.pt)
- **Output**: Action labels + confidence scores
- **Result**: Per-frame predictions + 2 aggregated segments

### Stage 4: Aggregation & Visualization
- **Input**: Tracking + Actions + Timestamps
- **Process**: Temporal segmentation + video annotation
- **Output**: JSON results + annotated videos
- **Result**: Complete visualization

---

## How to Use These Results

### 1. Quick Verification (2 minutes)
```bash
# Watch the side-by-side comparison
open visualization/sidebyside.mp4
```
Look for:
- Smooth green circles tracking the object
- Action text changing from "IDLE" to "CARRY" at ~1 second mark
- Consistent tracking throughout the video

### 2. Data Analysis (Python)
```python
import json

# Load results
with open('IMG_2013_results.json') as f:
    results = json.load(f)

# Access per-frame data
frames = results['per_frame']
print(f"Total frames: {len(frames)}")
print(f"Frame 0: {frames[0]}")

# Get action segments
segments = results['action_segments']
for seg in segments:
    print(f"{seg['action']}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")

# Analyze tracking
positions = [f['position'] for f in frames if f['position']]
print(f"Frames with valid positions: {len(positions)}/{len(frames)}")
```

### 3. Detailed Review (15 minutes)
1. Open `sidebyside.mp4` in video player
2. Watch full video, noting:
   - When IDLE → CARRY transition occurs
   - Tracking circle movement
   - Confidence changes
3. Review `IMG_2013_results.json` JSON structure
4. Compare visual observations with numerical data

---

## Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Tracking Coverage | 100% (274/274) | ✅ Excellent |
| IDLE Confidence | 91.9% | ✅ Excellent |
| CARRY Confidence | 86.3% | ✅ Very Good |
| Overall Confidence | 89.1% | ✅ Excellent |
| Video Quality | Full HD, H.264 | ✅ High Quality |

---

## Next Steps

### Immediate (Optional)
- [ ] Watch sidebyside.mp4 for visual verification
- [ ] Inspect JSON data structure
- [ ] Note action transition point

### For Robot Simulation
These inference results will be compared against robot simulation results in:
`../02_ROBOT_SIMULATION_RESULTS/`

Once the robot simulation runs, you'll be able to see:
- How the robot responds to these predicted actions
- Accuracy of waypoint following
- Task completion metrics

---

## Troubleshooting

**Can't open videos?**
- Install FFmpeg: `brew install ffmpeg` (macOS)
- Use VLC player: Available for all platforms
- Python solution: Use `opencv-python` to read MP4

**Want to extract frames?**
```bash
ffmpeg -i sidebyside.mp4 -vf fps=1 frame_%04d.png
```

**Need to re-generate?**
```bash
uv run python scripts/visualize_results.py \
  --video data/raw/IMG_2013.MOV \
  --results IMG_2013_results.json \
  --output visualization/
```

---

**Generated**: 2026-08-31
**Status**: Ready for Review ✓
**Next**: Robot Simulation Results (To be generated)
