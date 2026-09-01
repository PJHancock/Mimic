# Project Overview: Learning Manipulation Skills from Human Demonstration Video

## Goal

Watch a human perform a tabletop manipulation task, recover the manipulation phase and demonstrated object motion, retarget that task to a simulated Franka Panda, and execute it in MuJoCo.

The default composite-skill episode is:

**IDLE → HOVER → GRASP → CARRY → RELEASE → HOVER → IDLE**

A complete episode may omit the terminal HOVER. After one complete episode, playback also supports `RELEASE → IDLE → GRASP → CARRY → RELEASE`; the robot still generates an approach/hover before each grasp.

The learned model classifies phase. It does not estimate object coordinates or command the arm.

## Core split

### What is happening?

A frozen visual encoder plus a trained temporal classifier predicts `IDLE`, `HOVER`, `GRASP`, `CARRY`, and `RELEASE`. Graph-aware post-processing turns noisy scores into a stable accepted-state stream.

### Where is it happening?

An independent object tracker supplies calibrated table XY in meters (top-left origin, +X right, +Y down). Coordinate retargeting maps those meters into the Panda/MuJoCo workspace. Path processing selects robot-compatible XY geometry; waypoint construction adds configured height and a fixed downward tool orientation.

```text
video
  ├─ frame features -> temporal classifier -> accepted phases
  └─ object tracking -> calibrated table XY meters
                       │
                       v
            mimic.demo_task_input.v1
                       │
  TaskExtractor -> CoordinateRetargeter -> PathProcessor
                       │
                       v
         waypoints -> Mink IK -> Ruckig -> MuJoCo Panda
```

## Authoritative details

- [Architecture](ARCHITECTURE.md) — components and ownership boundaries
- [Skill graph](SKILL_GRAPH.md) — labels, transitions, and post-state policy
- [Task extraction and retargeting](TASK_EXTRACTION_AND_RETARGETING.md) — frame-based meter-space contract
- [Robot execution](ROBOT_EXECUTION.md) — Panda scene, IK, control, and safety gates
- `AGENTS.md` — engineering and safety contract
