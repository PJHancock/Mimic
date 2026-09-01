# Frame Features and Action Classification

The temporal model owns manipulation-phase classification only. Object position, task geometry, IK, and robot control remain separate.

## Current implementation

1. `scripts/process_demo_video.py` tracks the object and extracts one feature vector per accepted video frame.
2. `ActionClassifier` evaluates a sliding temporal context and returns one probability for every label in the active `SkillCatalog` order.
3. `GraphStatePostProcessor` validates transitions and emits one accepted state per classifier timestep.
4. `mimic.demo_task_input.v1` stores accepted actions with the independent tracking stream. `mimic.skill_scores.v2` stores full score distributions for diagnostics.

The committed checkpoint is `models/action_classifier_lstm.pt`. Its catalog fingerprint and training summary are recorded in `models/training_config.json`.

## Encoder status

`VJepaEncoder` is an abstraction name, not proof that the current pipeline is executing pretrained V-JEPA 2. The integrated pipeline currently requests the `timesformer` backend, which the implementation maps to a pretrained torchvision ResNet50 frame encoder. The `vjepa2` loader is incomplete and falls back when the external implementation is unavailable; it does not load committed V-JEPA 2 weights.

Treat any claim comparing V-JEPA 2 quality with the current checkpoint as unverified until the loaded encoder, checkpoint hash, and training data provenance are recorded together. Changing the encoder or fallback policy is a model/design change.

## Commands

Extract cached features:

```bash
uv run python scripts/extract_vjepa_embeddings.py \
  --video-dir data/raw \
  --output-dir data/embeddings \
  --device cpu
```

Train the classifier from cached embeddings and labels:

```bash
uv run python scripts/train_action_classifier.py \
  --embeddings-dir data/embeddings \
  --labels-dir data/labels \
  --output-dir models
```

Run the supported video pipeline:

```bash
uv run mimic-video-pipeline path/to/demo.mov
```

The active skill configuration must match the checkpoint catalog. Robot-facing inference rejects missing catalog metadata and superseded result schemas.
