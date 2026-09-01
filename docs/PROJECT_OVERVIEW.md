# Project Overview: Learning Manipulation Skills from Human Demonstration Video

## Goal

Build a system that watches a human perform a simple tabletop manipulation task, understands **what action is happening**, extracts **where the object is moving**, and then reproduces the task with a simulated Franka Panda robot arm.

The current default composite-skill episode is intentionally simple:

**IDLE → HOVER → GRASP → CARRY → RELEASE → HOVER → IDLE**

For example, a person picks up a ball, moves it across a table, and places it somewhere else. The system should infer the manipulation phases from the demonstration and retarget the motion to the Panda arm in MuJoCo.

The main focus is **learning the manipulation skill/action state**, not memorizing exact trajectories or positions.

---

# Core Idea

Separate the problem into two parallel questions:

### 1. What is happening?

Use a frozen visual feature encoder followed by a small temporal classifier that we train. V-JEPA 2 is the intended representation experiment; the current integrated backend is documented separately.

The classifier predicts:

- `IDLE`
- `HOVER`
- `GRASP`
- `CARRY`
- `RELEASE`

### 2. Where is it happening?

Use existing tracking tools rather than learning positioning from scratch.

Possible tools:

- MediaPipe for hand tracking
- SAM2 or another tracker for the manipulated object
- Fixed-camera calibration for converting image coordinates into tabletop coordinates

These two outputs are combined into a symbolic robot task.

```text
Human Video
    │
    ├─────────────── Visual understanding ──────────────┐
    │                                                   │
    │     Frame Feature Encoder → Temporal Classifier   │
    │                                                   │
    │   IDLE / HOVER / GRASP / CARRY / RELEASE        │
    │                                                   │
    ├──────────────── Geometry ─────────────────────────┤
    │                                                   │
    │     Hand + Object Tracking → positions/path       │
    │                                                   │
    └───────────────────────┬───────────────────────────┘
                            │
                            ▼
                  Task Representation
                            │
                            ▼
                    Franka Panda
                            │
                            ▼
                         MuJoCo
```

---

# Human Demonstration

The human performs a manipulation task in front of a fixed camera.

The initial environment is constrained to a tabletop.

We intentionally narrate the demonstration with phrases such as:

- "grab the ball"
- "move it over here"
- "put it here"
- "release"

Approximately **50 demonstrations** should be enough for an initial experiment because we are not training the video encoder from scratch.

We should vary:

- starting object position
- destination position
- movement direction
- path shape
- movement speed
- duration of each phase

while keeping relatively constant:

- camera position
- tabletop
- lighting
- object type initially

This helps test whether the model learns the **action phase** rather than memorizing location.

---

# Audio Pipeline

Audio is transcribed using an existing speech model such as Whisper.

```text
Audio
  ↓
Whisper
  ↓
Timestamped Transcript
```

Example:

```text
1.8 s  "grab the ball"
3.2 s  "move it over here"
5.6 s  "release"
```

For the MVP, narration primarily serves as **cheap supervision for labeling demonstrations**.

Instead of manually labeling every video frame, speech timestamps can provide approximate boundaries for:

```text
IDLE
HOVER
GRASP
CARRY
RELEASE
```

These can then be manually corrected if necessary.

An important goal is for the final visual model to recognize these actions **without needing narration**.

Language can still later be used separately to capture higher-level intent such as:

> "Put the ball over here."

---

# Visual Representation

The experimental design calls for a frozen pretrained video representation such
as V-JEPA 2. Only the smaller temporal classifier is trained.

The current integrated `VJepaEncoder` path does not yet establish that
experiment: it selects a framewise ResNet50 backend, and the optional V-JEPA 2
loader has no committed pretrained weights. See
[Frame Features and Action Classification](VJEPA_CLASSIFIER_PIPELINE.md). Encoder
identity and checkpoint provenance must be recorded before comparing feature
families.

---

# Temporal Action Model

The main ML contribution is a lightweight temporal model that maps visual feature sequences to manipulation phases.

Conceptually:

```text
Visual features over time
          ↓
     Temporal Head
          ↓
IDLE
HOVER
GRASP
CARRY
RELEASE
```

Possible models include:

- linear classifier as a baseline
- small MLP
- GRU
- small temporal transformer

A GRU or similarly lightweight temporal model is likely a good first learned approach.

The model learns something approximately equivalent to:

```text
recent visual history
        ↓
"What phase of manipulation are we in?"
```

The objective is specifically **not** to learn object coordinates.

---

# Training Dataset

Each demonstration produces a sequence of visual features and corresponding phase labels.

Conceptually:

| Time | Video Representation | Label |
|---|---|---|
| t0 | z0 | IDLE |
| t1 | z1 | HOVER |
| t2 | z2 | HOVER |
| t3 | z3 | GRASP |
| t4 | z4 | CARRY |
| t5 | z5 | CARRY |
| t6 | z6 | RELEASE |
| t7 | z7 | HOVER |
| t8 | z8 | IDLE |

Features should be precomputed once and cached.

```text
50 Videos
    ↓
Frozen encoder
    ↓
Cached Embeddings
    ↓
Train Temporal Model
```

This makes experimentation fast because changing the temporal model does not require rerunning the video encoder.

---

# Dataset Split

The split must happen at the **demonstration level**, not the video-window level.

For example:

```text
40 complete demonstrations → training
5 complete demonstrations  → validation
5 complete demonstrations  → testing
```

We should never place clips from the same demonstration in both training and testing.

Otherwise the evaluation would substantially overestimate generalization.

---

# Geometric Tracking

Positioning is handled separately from action recognition.

The system tracks:

```text
hand_position(t)
object_position(t)
object_trajectory(t)
```

Possible tools:

### MediaPipe

Used for hand landmarks and potentially:

- hand center
- wrist position
- fingertip locations
- finger closure
- hand velocity

### SAM2 / Object Tracker

Used to track the manipulated object across the video.

Once the object has been grasped, its trajectory is generally more useful than the hand trajectory itself.

---

# Coordinate Mapping

The first version avoids full monocular 3D reconstruction.

Because everything happens on a tabletop, image pixels are calibrated into a
2D table frame in meters before task extraction.

```text
Camera pixels
        ↓ explicit homography
Table XY meters (top-left origin, +X right, +Y down)
        ↓ explicit CoordinateRetargeter mapping
MuJoCo world XY meters
```

Retargeting names both frames and supplies the target origin plus perpendicular
unit source-axis directions. It does not resize, normalize, shear, clamp, or
silently flip axes. This keeps coordinate semantics inspectable across
embodiments.

---

# Z Position

For the MVP, vertical position does not need to be estimated from video.

Instead, height can depend on the inferred action state.

```text
HOVER
→ safe height above object

GRASP
→ lower to object height

CARRY
→ transport height

RELEASE
→ lower to placement height
```

This converts a difficult monocular 3D reconstruction problem into a much simpler tabletop manipulation problem.

Later versions could estimate full XYZ motion.

---

# Task Representation

The learned state and tracked geometry are combined into a simple symbolic representation.

For example:

```text
GRASP(ball, start_position)

CARRY(
    object = ball,
    trajectory = [...]
)

RELEASE(ball, destination)
```

This representation is intentionally independent of the robot itself.

The human demonstration describes **what happened**, while the Panda controller determines **how the Panda should accomplish it**.

---

# Human-to-Robot Retargeting

The robot should not blindly reproduce every human movement.

Human demonstrations can contain:

- jitter
- unnecessary wrist motion
- overshoot
- irregular paths
- movements that do not make sense for a robot arm

Instead:

```text
Calibrated demonstrated path
        ↓
Retarget every retained sample
        ↓
Explicit path-processing policy
        ↓
Configured robot waypoints
```

For example:

```text
Human:
wiggle → reach → grab → curved motion → adjust → release

Robot:
approach → grasp → smooth path → release
```

This provides some flexibility while preserving the demonstrated task.

---

# Post-Temporal-Model Robot Pipeline

After the action classifier predicts the current manipulation phase:

```text
Temporal Model
IDLE / HOVER / GRASP / CARRY / RELEASE
               │
               │
Object Tracking ───────┐
Hand Tracking           │
Trajectory              │
Destination             │
                       ▼
               Task State Machine
                       │
                       ▼
               Robot Task Command
                       │
                       ▼
              Coordinate Retargeting
                       │
                       ▼
               Cartesian Waypoints
                       │
                       ▼
               Trajectory Smoothing
                       │
                       ▼
                Inverse Kinematics
                       │
                       ▼
                  Joint Targets
                       │
              ┌────────┴────────┐
              ▼                 ▼
         Arm Control      Gripper Control
              │                 │
              └────────┬────────┘
                       ▼
                 Franka Panda
                       │
                       ▼
                     MuJoCo
```

---

# Robot State Machine

A simple deterministic state machine translates learned actions into robot behavior.

### IDLE

```text
Issue no arm or gripper change; preserve the preceding state.
```

### HOVER

```text
From IDLE, move above the configured grasp point.
From RELEASE, return to the saved home joint configuration.
```

### GRASP

```text
Move down to object.
Close gripper.
```

### CARRY

```text
Keep gripper closed.
Follow retargeted object trajectory.
```

### RELEASE

```text
Move to destination.
Lower object.
Open gripper.
```

The ML model determines the **semantic state**.

Traditional robotics determines how the robot executes it.

---

# Trajectory Generation

Tracked human motion should be cleaned before controlling the Panda.

```text
Raw tracked trajectory
        ↓
Resampling
        ↓
Smoothing
        ↓
Robot workspace mapping
        ↓
Cartesian trajectory
```

The resulting desired end-effector trajectory is:

```text
x_d(t)
y_d(t)
z_d(t)
```

For the first version, end-effector orientation remains fixed with the gripper pointing downward.

---

# Inverse Kinematics

The Panda has seven arm joints.

The desired Cartesian position must therefore be converted into Panda joint configurations.

```text
Desired End-Effector Pose
          ↓
Inverse Kinematics
          ↓
[q1, q2, q3, q4, q5, q6, q7]
```

Existing IK tools should be used rather than implementing an IK solver from scratch.

---

# Panda Control

The Panda simulation receives joint targets from the IK system.

The arm controller runs much faster than the visual understanding model.

Conceptually:

```text
Feature Encoder / Action Recognition
~a few updates per second

Trajectory Generator
~tens of updates per second

Robot Controller
~hundreds of updates per second

MuJoCo Physics
high-frequency simulation
```

These systems therefore do not need to operate at the same rate.

---

# Gripper Control

The gripper is controlled separately from the arm joints.

```text
HOVER → OPEN
GRASP    → CLOSE
CARRY     → CLOSED
RELEASE  → OPEN
```

This keeps manipulation behavior simple and interpretable.

---

# Initial Simulation

Use an existing Franka Panda MuJoCo model rather than constructing the robot manually.

The initial scene requires only:

- Panda arm
- gripper
- table
- ball or simple graspable object

The first success condition is basic pick-and-place.

---

# MVP Success Criterion

The minimum successful demonstration is:

1. Record an unseen human demonstration.
2. Track the manipulated object.
3. Pass the video through the declared frozen feature encoder.
4. Classify the sequence into:

```text
HOVER
GRASP
CARRY
RELEASE
```

5. Determine the object's start position, trajectory, and destination.
6. Retarget the geometry into the Panda workspace.
7. Generate Panda end-effector waypoints.
8. Solve IK.
9. Execute the action in MuJoCo.
10. Successfully move the simulated object to approximately the demonstrated destination.

Critically, the test demonstration should use a different start/end location than the training examples.

---

# What We Are Actually Testing

The core research question is:

> Can a pretrained video representation be adapted with a small amount of human demonstration data to recognize manipulation phases independently of the exact trajectory or object location?

A useful baseline would use simple geometric signals such as:

```text
hand-object distance
finger closure
object velocity
```

and compare:

```text
Geometric Features
        ↓
Classifier
```

against:

```text
Declared pretrained features
       ↓
Temporal Classifier
```

This gives us an actual experiment rather than simply plugging a large model into a robotics pipeline.

---

# What We Are Not Training

For the initial hackathon version, we deliberately avoid training:

- the video encoder
- the object tracker
- the hand tracker
- the robot controller
- inverse kinematics
- the robot motion planner
- the position estimator

The primary learned component is the **temporal manipulation-phase model**.

This keeps the scope realistic while preserving a meaningful ML contribution.

---

# Hardware

Available compute includes:

- RTX 3090 with 24 GB VRAM
- M4 Pro MacBook
- potential BYU supercomputing access

Suggested split:

```text
RTX 3090
├── feature extraction
├── embedding extraction
└── temporal model training

M4 Pro
├── data recording
├── tracking
├── transcription
├── MuJoCo
├── visualization
└── integration
```

Features should be cached so temporal-model experiments can run without repeatedly processing all videos.

---

# Stretch Goals

Once the MVP works, possible extensions include:

### Better language grounding

Use speech not only for training labels but also for commands such as:

> "Put the ball on the left."

or:

> "Move this one over there."

### Orientation

Infer wrist/object orientation and reproduce rotations with the Panda.

### Multiple objects

Track multiple objects and infer which object is being manipulated.

### More skills

Expand the action vocabulary:

```text
HOVER
GRASP
CARRY
RELEASE
PUSH
PULL
ROTATE
WAIT
```

### Variable camera viewpoints

Test whether the action model generalizes beyond a fixed camera.

### Different operators

Train on one person and evaluate on another.

### V-JEPA 2-AC / Learned Planning

Explore action-conditioned world models for predicting robot outcomes and planning actions.

### Real Robot Transfer

Replace the simulated Panda with a physical robot while preserving the same symbolic task representation.

---

# One-Sentence Project Pitch

**We learn manipulation phases from narrated human demonstration video using pretrained video representations, separately recover the task geometry, and retarget the resulting skill sequence to a simulated Franka Panda robot.**

# Short Pitch

A person demonstrates a manipulation task such as picking up and moving an object. A pretrained video encoder processes the demonstration, while a lightweight temporal model learns to recognize manipulation phases like approach, grasp, move, and release. Existing tracking tools independently recover the object's motion. The learned skill sequence and tracked geometry are then converted into robot-compatible trajectories and executed by a Franka Panda in MuJoCo.

The key idea is separating **what action is being performed** from **where the action occurs**, allowing the robot to reproduce the demonstrated skill without simply memorizing the human's exact trajectory.
