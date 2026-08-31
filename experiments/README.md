# Experiments

Results and logs from different experimental runs.

## Structure

Each experiment gets its own directory:

```
experiments/
├── exp_01_baseline/
│   ├── config.yaml              # Configuration used
│   ├── model.pth                # Trained model weights
│   ├── metrics.json             # Evaluation metrics
│   ├── logs/                    # TensorBoard logs
│   ├── notes.md                 # Experiment notes
│   └── results/
│       ├── predictions.json     # Model predictions
│       └── visualizations/
├── exp_02_vjepa_temporal/
│   └── ...
└── README.md
```

## Running an Experiment

1. Create a config file: `configs/experiment_name.yaml`
2. Run training: `python scripts/train_temporal_model.py --config configs/experiment_name.yaml`
3. Create results directory: `mkdir -p experiments/experiment_name`
4. Copy artifacts:
   ```bash
   cp configs/experiment_name.yaml experiments/experiment_name/
   cp outputs/model.pth experiments/experiment_name/
   cp outputs/metrics.json experiments/experiment_name/
   ```
5. Write notes: `vim experiments/experiment_name/notes.md`

## Experiment Notes Template

```markdown
# Experiment: [Name]

## Objective
What research question are we answering?

## Setup
- Model architecture: [GRU, Transformer, MLP]
- Training data: [# demonstrations, split]
- Hardware: [GPU/CPU, memory]
- Training duration: [hours]

## Results
- Training loss: [value]
- Validation accuracy: [value]
- Test accuracy: [value]

## Analysis
What worked well? What didn't?

## Insights
Key takeaways for future experiments.

## Next Steps
Ideas for improvement.
```

## Comparison

To compare experiments:
```bash
python scripts/compare_experiments.py \
  experiments/exp_01_baseline/metrics.json \
  experiments/exp_02_vjepa_temporal/metrics.json
```

---

**Goal**: Keep experiments reproducible and well-documented so we can build on successful approaches.
