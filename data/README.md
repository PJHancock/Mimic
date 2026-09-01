# Data

Generated or private datasets are ignored by default.

- `raw/` — source demonstration videos.
- `processed/` — derived frames or cleaned media.
- `embeddings/` — cached frame features.
- `labels/` — frame-aligned classifier labels.
- `tracks/` — cached tracker output.
- `annotations/` — committed calibration or annotation metadata when needed for reproducibility.

The checked-in `annotations/calibrations.json` maps image pixels into table coordinates in meters. Robot task extraction consumes those calibrated coordinates, not raw pixels or normalized workspace values.

Current offline robot handoff:

```text
mimic.demo_task_input.v1
├── catalog/checkpoint/postprocessor provenance
├── resolved_actions[]  # one accepted phase per classifier timestep
└── object_tracks[]     # independently sampled source-video frames
```

Both streams use positive one-based source-video `frame_idx` values. Seconds are optional metadata and are never substituted with frame counts.

Do not commit large videos, cached features, or generated visualizations. Preserve the configuration and provenance required to reproduce any committed model or result.
