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
