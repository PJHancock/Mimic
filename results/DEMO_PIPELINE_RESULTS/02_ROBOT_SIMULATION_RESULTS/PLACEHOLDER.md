# Robot Simulation Results - PLACEHOLDER

**Status**: 🔄 Ready for Execution (Not Yet Generated)

## Overview

This folder will contain the results of running the robot simulation once the command is executed. Currently, it serves as a placeholder showing what outputs will be generated.

---

## Expected Outputs

When you run the robot simulation with the following command:

```bash
uv run python scripts/process_demo_video.py \
  --video data/raw/IMG_2013.MOV \
  --model models/action_classifier_lstm.pt \
  --config configs/robots/panda_complete.yaml \
  --output results/DEMO_PIPELINE_RESULTS/02_ROBOT_SIMULATION_RESULTS/ \
  --simulate-robot
```

The following files will be generated in this directory:

### 1. `execution_log.jsonl` (NEW LINE JSON FORMAT)

**Size**: ~1-5 MB
**Format**: Line-delimited JSON (one event per line)

**Contains:**
- MuJoCo simulation metadata
- Joint trajectory over time
- Gripper state transitions
- End-effector positions
- Force/torque data
- Collision events (if any)

**Example Events:**
```json
{"event": "metadata", "config": "...", "waypoints_sha256": "...", "versions": {...}}
{"event": "step", "time": 0.0, "joint_positions": [0, 0.785, 0, -2.356, 0, 2.356, 0.785], "gripper_width": 0.08}
{"event": "step", "time": 0.01, "joint_positions": [...], "gripper_width": 0.08}
...
{"event": "task_complete", "success": true, "duration": 15.3, "path_error": 0.002}
```

### 2. `simulation_video.mp4` (NEW VIDEO)

**Size**: ~20-50 MB
**Resolution**: Typically 1280×720 or 1920×1080
**Frame Rate**: 30 FPS
**Codec**: H.264

**Shows:**
- 3D rendering of Panda robot arm
- Table and workspace environment
- Red object being picked and placed
- Joint angle annotations
- Gripper state visualization
- Trajectory path visualization

### 3. `sidebyside_comparison.mp4` (NEW VIDEO - COMPARISON)

**Size**: ~40-100 MB
**Format**: Side-by-side composite video

**Left Side**: Original demo video with inference tracking
- Green tracking circles (inferred positions)
- Action labels (IDLE/CARRY)
- Timestamps

**Right Side**: Robot simulation rendering
- 3D robot arm animation
- Object position in simulation
- Gripper state
- Joint angle display

**Purpose**: Direct visual comparison between:
- What the inference pipeline predicted
- How the robot simulation executed

### 4. `trajectory_data.json` (NEW DATA FILE)

**Size**: ~200 KB - 1 MB
**Format**: Structured JSON

**Contains:**
```json
{
  "metadata": {
    "task": "pick_place",
    "robot": "panda",
    "success": true,
    "duration_s": 15.3,
    "num_steps": 1530
  },
  "waypoints": {
    "approach": {"position": [...], "quaternion_wxyz": [...]},
    "grasp": {...},
    "lift": {...},
    "lower": {...},
    "retreat": {...},
    "path": [...]
  },
  "trajectory": {
    "times": [0.0, 0.01, 0.02, ...],
    "joint_positions": [[...], [...], ...],
    "joint_velocities": [...],
    "gripper_states": [...]
  },
  "object_trajectory": {
    "positions": [[x, y, z], ...],
    "orientations": [...]
  },
  "errors": {
    "path_tracking_error_mean": 0.002,
    "path_tracking_error_max": 0.015,
    "goal_position_error": 0.003
  }
}
```

### 5. `simulation_summary.txt` (NEW TEXT SUMMARY)

**Size**: ~5-20 KB
**Format**: Human-readable text

**Contents:**
```
ROBOT SIMULATION SUMMARY
=======================
Task: Pick and Place
Robot: Franka Panda
Success: YES/NO

METRICS:
  Duration: 15.3 seconds
  Total Steps: 1530
  Path Tracking Error (mean): 0.002 m
  Path Tracking Error (max): 0.015 m
  Goal Position Error: 0.003 m
  Gripper Success: YES/NO

WAYPOINTS EXECUTED:
  1. Approach: [0.5, 0.0, 0.4] ✓
  2. Grasp: [0.5, 0.0, 0.1] ✓
  3. Lift: [0.5, 0.0, 0.3] ✓
  4. Transport Path: 3 waypoints ✓
  5. Lower: [0.6, 0.1, 0.1] ✓
  6. Retreat: [0.6, 0.1, 0.4] ✓

COMPARISON TO INFERENCE:
  Inference Prediction: CARRY (86.3% confidence)
  Simulation Result: CARRY (waypoint 3-5) ✓
  Alignment: Excellent

NOTES:
  - Smooth trajectory execution
  - No collisions detected
  - Gripper engaged correctly
  - Object placed at goal position
```

---

## Directory Structure (After Generation)

```
02_ROBOT_SIMULATION_RESULTS/
├── execution_log.jsonl              (MuJoCo event log)
├── simulation_video.mp4             (Robot arm animation)
├── sidebyside_comparison.mp4        (Inference vs Simulation)
├── trajectory_data.json             (Detailed motion data)
├── simulation_summary.txt           (Human-readable results)
└── PLACEHOLDER.md                   (This file - will be replaced)
```

---

## How to Generate

### Prerequisites
1. Ensure `mink` is installed (for inverse kinematics)
2. Ensure `ruckig` is installed (for trajectory planning)
3. Ensure `mujoco` is installed (for physics simulation)

Check installation:
```bash
uv pip list | grep -E "mink|ruckig|mujoco"
```

### Run Simulation

From the project root:

```bash
uv run python scripts/process_demo_video.py \
  --video data/raw/IMG_2013.MOV \
  --model models/action_classifier_lstm.pt \
  --config configs/robots/panda_complete.yaml \
  --output results/DEMO_PIPELINE_RESULTS/02_ROBOT_SIMULATION_RESULTS/ \
  --simulate-robot
```

### Expected Output

You should see:
```
6. Running robot simulation...
   ✓ Executor built from config: configs/robots/panda_complete.yaml
   ✓ Simulation executed (success=True)
   ✓ Execution log saved to: results/DEMO_PIPELINE_RESULTS/02_ROBOT_SIMULATION_RESULTS/execution_log.jsonl
```

### Execution Time

- **First run**: 30-60 seconds (includes model loading)
- **Typical run**: 20-40 seconds
- **With visualization**: +10-20 seconds

---

## Comparison With Inference Pipeline

Once generated, you'll be able to compare:

### Timing
- Inference prediction of "CARRY" action
- Robot simulator executing carry waypoints
- Frame-by-frame alignment

### Accuracy
- Tracking positions from demo video
- Simulated object positions
- Error metrics

### Execution
- Inferred action durations
- Simulated trajectory timing
- Overall coordination

---

## Next Steps

### To Generate Results

1. **Run the command** (see above):
   ```bash
   uv run python scripts/process_demo_video.py \
     --video data/raw/IMG_2013.MOV \
     --model models/action_classifier_lstm.pt \
     --config configs/robots/panda_complete.yaml \
     --output results/DEMO_PIPELINE_RESULTS/02_ROBOT_SIMULATION_RESULTS/ \
     --simulate-robot
   ```

2. **Monitor output**:
   - Watch console for progress messages
   - Files will appear in this directory as generation completes

3. **Review results**:
   - Open `sidebyside_comparison.mp4` first
   - Review `trajectory_data.json` for detailed metrics
   - Read `simulation_summary.txt` for high-level overview

### To Compare With Inference

1. **Open side-by-side comparison**:
   ```bash
   open sidebyside_comparison.mp4
   ```

2. **Compare with inference**:
   - Open `../01_INFERENCE_PIPELINE/visualization/sidebyside.mp4` alongside
   - Note tracking vs simulated positions
   - Verify action predictions match robot execution

3. **Analyze data**:
   ```bash
   # View trajectory data
   python -m json.tool trajectory_data.json | less
   
   # Extract error metrics
   grep -E "error|success" simulation_summary.txt
   ```

---

## Troubleshooting

### Simulation Fails to Run

**Error**: `No module named 'mink'`
```bash
uv pip install mink
```

**Error**: `No module named 'ruckig'`
```bash
uv pip install ruckig
```

**Error**: `XML Schema violation`
- Check `configs/robots/panda_complete.yaml` for valid syntax
- Verify model paths are correct

### Video Won't Render

**Issue**: `ffmpeg not found`
```bash
brew install ffmpeg  # macOS
apt-get install ffmpeg  # Linux
```

### JSON Parsing Issues

```bash
# Validate JSON
python -m json.tool trajectory_data.json > /dev/null

# Pretty print
python -c "import json; json.dump(json.load(open('trajectory_data.json')), open('pretty.json', 'w'), indent=2)"
```

---

## When This File Will Change

This `PLACEHOLDER.md` file will be replaced or archived when:

1. Robot simulation is successfully executed
2. All output files are generated
3. Results are validated

At that point, this directory will contain actual simulation results instead of this placeholder.

---

## Reference

- **Inference Results**: See `../01_INFERENCE_PIPELINE/`
- **Pipeline Code**: `scripts/process_demo_video.py`
- **Robot Config**: `configs/robots/panda_complete.yaml`
- **Documentation**: `../README.md`

---

**Status**: Ready for Execution
**Last Updated**: 2026-08-31
**Next Action**: Run simulation with --simulate-robot flag
