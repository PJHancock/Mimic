# create automatic labels
uv run python scripts/extract_labels.py

# calibrate motion tracking
uv run python scripts/extract_calibration_frame.py --video data/raw/IMG_2013.MOV --output data/raw/calibration_frame.png
uv run python scripts/calibrate_camera.py --image data/raw/calibration_frame.png --width 0.6 --height 0.4 --output data/annotations/calibration.json

# create embeddings
uv run python scripts/extract_vjepa_embeddings.py --video-dir data/raw/ --output-dir data/embeddings/ --device cuda

# train
uv run python scripts/train_action_classifier.py --embeddings-dir data/embeddings/ --labels-dir data/labels/ --output-dir models/ --epochs 150

# inference
uv run python scripts/process_demo_video.py \
    --video data/demo/IMG_2067.MOV \
    --model models/action_classifier_lstm.pt \
    --skill-config configs/skills/pick_place.yaml \
    --output results/short_demo/

uv run python scripts/process_demo_video.py \
    --video data/demo/IMG_2068.MOV \
    --model models/action_classifier_lstm.pt \
    --skill-config configs/skills/pick_place.yaml \
    --output results/long_demo/