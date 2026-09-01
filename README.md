# MIMIC: Learning Manipulation Skills from Demonstration

A system that learns robot manipulation skills from human video demonstrations and executes them with a simulated Franka Panda robot.

---

## 🌐 Interactive Dashboard

**[View the Live Dashboard](https://pjhancock.github.io/Mimic/)** ← Start here to explore the project!

The interactive dashboard provides:
- 📊 **Visual Pipeline** — Click through each stage to see what the system is doing
- 🎬 **Live Video Processing** — Watch videos flow through the complete pipeline in real-time
- 📈 **Expandable Visualizations** — Inspect loss curves and detailed analysis
- 🤖 **Execution Results** — See the robot simulation playback and movement tracking
- ⚙️ **Interactive Controls** — Run your own videos through the pipeline

**Run locally:**
```bash
cd web && npm install && npm run dev
```

---

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Run pipeline on a video
uv sync --group robot
uv run --group robot mimic --video data/raw/demo.mp4 --robot panda
```

Results are saved to `results/demo/` including simulation video and task artifacts.

## 🏗️ Project Structure

```
mimic/
├── src/mimic/
│   ├── vision/          # Frame features & temporal classifier
│   ├── tracking/        # Object tracking & coordinate mapping
│   ├── skills/          # Action labels & state machine
│   ├── robot/           # Task representation & IK control
│   └── integration/     # End-to-end pipeline
├── scripts/             # Training and simulation
├── configs/             # Configuration files
├── results/             # Pipeline artifacts
└── web/                 # Interactive dashboard
```

## 📋 System Pipeline

1. **Video Input** → Extract raw frames
2. **Visual Embeddings** → V-JEPA self-supervised features
3. **Temporal Classifier** → LSTM predicts manipulation phases
4. **Object Tracking** → HSV detection of target object
5. **Task Extraction** → Combine predictions and coordinates
6. **Robot Control** → Generate waypoints and execute in MuJoCo

## 📝 Configuration

- **Model**: `models/action_classifier_lstm.pt`
- **Robot**: Franka Panda (configurable profiles in `configs/robots/panda/`)
- **Inference**: Supports CPU, MPS (Apple Silicon), and CUDA

## 🔗 Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Task Extraction & Retargeting](docs/TASK_EXTRACTION_AND_RETARGETING.md)
- [Robot Execution](docs/ROBOT_EXECUTION.md)
- [Contributing](CONTRIBUTING.md)

## 👥 Contributors

Preston Hancock, Josh McConkie, Peter Bickel

---

For detailed setup and advanced usage, see the [docs](./docs/).
