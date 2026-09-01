# Mimic Run Inspector

A read-only local dashboard for completed Mimic runs. It discovers artifact sets
under `../results/`, summarizes their JSON/JSONL records, and streams retained
videos without changing the Python pipeline.

```bash
cd web
npm install
npm run dev
```

Open the local URL printed by Vite. The dashboard prioritizes successful runs
and selects the newest successful artifact set by default.

The **Process raw demonstration** control lists videos already present under
`data/raw/`. Choose the panda execution config (`slow` or `fast`), the
inference device, and run or rerun the existing simulation-only `mimic`
pipeline. The dashboard shows live tracking,
classification, artifact-writing, retargeting, and MuJoCo progress, then loads
the completed result automatically. Reprocessing replaces that run's
`.mimic.mp4` simulation recording. Only one local pipeline job runs at a time.

For completed runs, **Phase-aligned playback** uses the MuJoCo recording as the
master clock and slows the raw demonstration independently within each episode's
`HOVER`, `GRASP`, `CARRY`, and `RELEASE`. Repeated phases are paired in source
order, so a second carry is aligned with the second carry rather than the first.
A robot-only continuation hover maps onto the intervening raw `IDLE` footage.
This does not change source files, inference timestamps, or robot execution timing.

Source demonstration videos are discovered by matching the artifact basename
under `data/raw/` or `data/videos/`. When a source video was not retained, the
main player clearly labels and uses the run's `.mimic.mp4` simulation recording
as a presentation fallback. Robot outcome metrics come from the execution log,
not from visual appearance.

## GitHub Pages

The production build is a static, read-only showcase of the tracked `IMG_2067`
run. It packages that run's compact artifact data, source demonstration, and
MuJoCo recording; the local development server remains the only mode that can
start new pipeline jobs.

Push changes to `main` or run the **Deploy website to GitHub Pages** workflow
manually. In the repository's **Settings → Pages**, set the source to **GitHub
Actions**. The workflow builds `web/` with the repository subpath and deploys
`web/dist/`.
