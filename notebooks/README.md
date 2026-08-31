# Notebooks

Exploratory analysis, debugging, and visualization notebooks.

## Purpose

Notebooks are for:
- **Exploratory analysis** — understanding data patterns
- **Debugging** — visualizing intermediate outputs
- **Prototyping** — trying new ideas before moving to production code
- **Visualization** — creating plots and reports

Notebooks are **not** for:
- Core pipeline logic (keep that in `src/mimic/`)
- Production inference (use scripts or modules)
- Anything that should be version-controlled (use `.py` files)

## Naming Convention

Use numbered prefixes to indicate order of exploration:

```
01_data_exploration.ipynb
02_embedding_analysis.ipynb
03_tracking_debug.ipynb
04_robot_simulation.ipynb
```

## Example Notebooks

### Data Exploration
Load a few demonstrations and visualize:
- Frame distribution
- Object motion patterns
- Action phase durations
- Speech-action alignment

### Embedding Analysis
Visualize V-JEPA embeddings:
- t-SNE or UMAP projection
- Cluster by action phase
- Compare geometric features

### Tracking Debug
Visualize hand/object tracking:
- Overlay tracks on video
- Check for failures/discontinuities
- Compare tracker confidence

### Robot Simulation
Test robot control:
- Visualize trajectories
- Check IK solutions
- Preview MuJoCo execution

## Workflow

1. Create a notebook for your investigation
2. Keep it lightweight (import from `src/mimic/` modules)
3. Add markdown cells explaining findings
4. When insights are useful, move to production code
5. Don't commit notebooks with cell outputs (use `nbstripout`)

## Setup

Install notebook tools:
```bash
pip install jupyter jupyterlab nbstripout
jupyter notebook
```

Strip outputs before committing:
```bash
git add notebooks/*.ipynb
nbstripout notebooks/*.ipynb
git add notebooks/*.ipynb
git commit
```

---

**Key principle**: Notebooks are exploration tools. Core logic belongs in `src/`.
