# Architecture

Mimic separates learned phase recognition, observed object geometry, and deterministic robot execution.

```text
video
  ├─ frame features -> temporal classifier -> complete skill scores
  │                                      -> graph state post-processing
  └─ object tracking -> calibrated table XY in meters
                                           │
                                           v
                              mimic.demo_task_input.v1
                                           │
          TaskExtractor -> CoordinateRetargeter -> PathProcessor
                                           │
                                           v
             WaypointBuilder -> skill execution -> Mink IK
                                           │
                                           v
                  Ruckig joint reference -> MuJoCo Panda
                                           │
                                           v
                              evaluation and JSONL trace
```

## Boundaries

- The learned model classifies `IDLE`, `HOVER`, `GRASP`, `CARRY`, and `RELEASE`. It does not estimate object motion or control the robot.
- Tracking and classifier outputs share one-based source-video frame IDs but may have different sampling rates.
- The robot handoff stores accepted actions and the independent tracker stream. Raw scores remain a diagnostic artifact.
- Tracking enters task extraction as calibrated table coordinates in meters: top-left origin, +X right, +Y down.
- Retargeting alone maps table coordinates into a named target frame. It does not scale, clamp, or infer axes.
- Path processing selects XY geometry. Waypoint construction adds configured Z values and one fixed tool orientation.
- Robot execution is simulation-only and uses measured MuJoCo state for arrival, grasp, transport, release, and placement gates.

## Package layout

- `src/mimic/common/` — shared records and constants.
- `src/mimic/vision/` — frame feature encoder and temporal classifier.
- `src/mimic/tracking/` — hand/object tracking and camera calibration.
- `src/mimic/skills/` — versioned catalog, transition graph, postprocessor, and handler registry.
- `src/mimic/robot/` — extraction, retargeting, paths, waypoints, IK, control, and MuJoCo I/O.
- `src/mimic/integration/` — persisted schemas and pipeline entry points.

## Configuration

- `configs/default.yaml` — general project defaults and authoritative path-processing default.
- `configs/skills/pick_place.yaml` — active skill catalog, graph, and post-state settings.
- `configs/retargeting.yaml` — explicit table-to-MuJoCo mapping and tabletop clone.
- `configs/robot_pipeline.yaml` — path and waypoint policy for the checked-in cube fixture.
- `configs/robots/panda.yaml` — fail-fast contract template.
- `configs/robots/panda_complete.yaml` — runnable simulation fixture.

## Authoritative details

- [Task extraction and retargeting](TASK_EXTRACTION_AND_RETARGETING.md)
- [Skill graph](SKILL_GRAPH.md)
- [Robot execution](ROBOT_EXECUTION.md)
- [Classifier pipeline](VJEPA_CLASSIFIER_PIPELINE.md)

`AGENTS.md` contains the durable engineering and safety contract.
