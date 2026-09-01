# Standard Menagerie Panda

Run `uv run python scripts/fetch_panda_model.py` once to populate `upstream/`.
The downloader pins Menagerie revision `da76818e269b82289eba39808e2fb91d679d6994`,
retains the Apache-2.0 license, records file hashes, and never overwrites files.
Runtime execution does not access the network. Downloaded meshes (~34 MB) are
excluded from Git. Keep `manifest.json` with the assets for provenance.

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/franka_emika_panda

Use standard `panda.xml`, not `mjx_panda.xml`: their gripper controls differ.
The model is unmodified. A tool offset is supplied explicitly by the caller;
no default grasp center or tabletop coordinates are inferred here.

The checked-in `../panda_pick_place_scene.xml` wraps this pinned model with the
approved 0.508 m x 0.762 m left-edge tabletop clone and the simulation-default
4 cm, 30 g cube. Its `pick_place_home` keyframe includes the cube free-joint pose;
the upstream `home` keyframe contains only Panda coordinates and must not reset
this augmented scene.
The table's near edge is 0.15 m in front of the Panda base. At reset-only
simulation initialization, the pipeline replaces the cube keyframe translation
with the retargeted grasp position and clears its free-joint velocity.
