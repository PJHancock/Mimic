import argparse
import difflib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import torchaudio

from mimic.common.types import ActionPhase

STATE_ALIASES: Dict[str, str] = {
    "aproach": "hover",
    "approach": "hover",
    "approch": "hover",
    "reach": "hover",
    "grab": "grasp",
    "grap": "grasp",
    "clasp": "grasp",
    "hold": "grasp",
    "shift": "carry",
    "move": "carry",
    "drop": "release",
    "letgo": "release",
}

# Default vocabulary mapping for hand-object interaction states
DEFAULT_LABELS = [phase.value.lower() for phase in ActionPhase]


def get_video_metadata(video_path: str) -> Tuple[float, int]:
    """Extracts FPS and total frame count directly from MP4 header using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or total_frames <= 0:
        raise ValueError(
            f"Failed to read valid metadata from {video_path} (fps={fps}, total_frames={total_frames})"
        )

    return fps, total_frames


def extract_emissions(
    video_path: str,
    device: torch.device,
) -> Tuple[torch.Tensor, int, List[str]]:
    """Extracts audio directly from MP4 via FFmpeg pipe and computes Wav2Vec2 emissions."""
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    model.eval()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Pipe 16kHz mono 32-bit float PCM audio directly from MP4 using FFmpeg
    cmd = [
        ffmpeg_exe,
        "-i", video_path,
        "-vn",                   # Disable video decoding
        "-ac", "1",               # Convert to mono
        "-ar", str(bundle.sample_rate), # Resample to 16000 Hz
        "-f", "f32le",           # Raw 32-bit little-endian float output
        "pipe:1"
    ]

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    raw_audio, _ = process.communicate()

    if len(raw_audio) == 0:
        raise ValueError(f"No audio stream found or failed to decode audio from: {video_path}")

    # Convert binary buffer into NumPy float32 array
    audio_np = np.frombuffer(raw_audio, dtype=np.float32)

    # Convert to PyTorch Tensor -> Shape: [1, num_samples]
    waveform = torch.from_numpy(audio_np).unsqueeze(0).to(device)
    sample_rate = bundle.sample_rate

    with torch.inference_mode():
        emissions, _ = model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)

    labels = bundle.get_labels()
    labels = [label.upper() if isinstance(label, str) else label for label in labels]

    return emissions[0].cpu(), sample_rate, labels


def decode_constrained_timestamps(
    emissions: torch.Tensor,
    labels: List[str],
    target_words: List[str],
) -> List[Tuple[str, float]]:
    """Greedy CTC decoder using pure PyTorch tensors."""
    frame_duration = 0.02  # Wav2Vec2 frame stride (20ms)

    best_tokens = torch.argmax(emissions, dim=-1).cpu().numpy()
    blank_idx = labels.index("-") if "-" in labels else 0

    dedup_tokens = []
    for i, token_idx in enumerate(best_tokens):
        if token_idx != blank_idx:
            if i == 0 or token_idx != best_tokens[i - 1]:
                dedup_tokens.append((token_idx, i))

    word_timestamps: List[Tuple[str, float]] = []
    current_word = ""
    start_frame = None

    for token_idx, frame_idx in dedup_tokens:
        char = labels[token_idx].lower()
        if char in ["|", " "]:
            if matched(current_word) in target_words and start_frame is not None:
                word_timestamps.append((matched(current_word), start_frame * frame_duration))
            else:
                print(current_word)
                print(start_frame * frame_duration)
            current_word = ""
            start_frame = None
        else:
            if not current_word:
                start_frame = frame_idx
            current_word += char

    if current_word in target_words and start_frame is not None:
        word_timestamps.append((current_word, start_frame * frame_duration))

    return word_timestamps

def matched(
    word: str,
    target_words: List[str] = DEFAULT_LABELS,
    alias_map: Dict[str, str] = STATE_ALIASES,
    cutoff: float = 0.65,
) -> Optional[str]:
    """Resolves raw CTC words to canonical state labels using aliases or fuzzy matching.
    
    Returns:
        str: The canonical target word that was matched (e.g., 'approach').
        None: If no valid match is found.
    """
    clean_word = word.lower().strip()

    if not clean_word:
        return None

    # 1. Direct match in target vocabulary
    if clean_word in target_words:
        return clean_word

    # 2. Direct match in alias mapping
    if clean_word in alias_map:
        canonical = alias_map[clean_word]
        if canonical in target_words:
            print(f"[DEBUG] Direct Alias Resolved: '{clean_word}' -> '{canonical}'")
            return canonical

    # 3. Fuzzy match across BOTH target_words and alias keys
    candidates = list(set(target_words + list(alias_map.keys())))
    matches = difflib.get_close_matches(clean_word, candidates, n=1, cutoff=cutoff)

    if matches:
        matched_candidate = matches[0]

        # Resolve to canonical word if matched_candidate is an alias
        if matched_candidate in alias_map:
            canonical = alias_map[matched_candidate]
            if canonical in target_words:
                print(
                    f"[DEBUG] Fuzzy Alias Matched: '{clean_word}' -> '{matched_candidate}' -> '{canonical}'"
                )
                return canonical
        elif matched_candidate in target_words:
            print(
                f"[DEBUG] Fuzzy Direct Matched: '{clean_word}' -> '{matched_candidate}'"
            )
            return matched_candidate

    return None

def generate_frame_labels(
    word_timestamps: List[Tuple[str, float]],
    total_frames: int,
    fps: float,
    reaction_offset_sec: float = -0.25,
    idle_label: str = "idle",
) -> np.ndarray:
    """Maps speech timestamps to video frames using forward-latching logic."""
    frame_array = np.full(total_frames, idle_label, dtype=object)

    if not word_timestamps:
        return frame_array

    events = []
    for label, t_speech in word_timestamps:
        adjusted_time = max(0.0, t_speech + reaction_offset_sec)
        frame_idx = int(np.floor(adjusted_time * fps))
        events.append((frame_idx, label))

    events.sort(key=lambda x: x[0])

    for i in range(len(events)):
        curr_frame, curr_label = events[i]
        next_frame = events[i + 1][0] if i + 1 < len(events) else total_frames

        start_idx = min(curr_frame, total_frames)
        end_idx = min(next_frame, total_frames)

        if start_idx < end_idx:
            frame_array[start_idx:end_idx] = curr_label

    return frame_array


def process_pipeline(
    video_path: str,
    output_npy_path: str,
    fps: Optional[float] = None,
    total_frames: Optional[int] = None,
    target_words: List[str] = DEFAULT_LABELS,
    reaction_offset_sec: float = -0.25,
) -> np.ndarray:
    """Executes full audio extraction, CTC alignment, and frame-latching workflow."""
    auto_fps, auto_frames = get_video_metadata(video_path)
    
    fps = fps if fps is not None else auto_fps
    total_frames = total_frames if total_frames is not None else auto_frames

    print(f"Processing {video_path} | FPS: {fps:.2f} | Total Frames: {total_frames}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emissions, sample_rate, labels = extract_emissions(video_path, device)

    target_words_clean = [w.lower() for w in target_words]
    word_timestamps = decode_constrained_timestamps(
        emissions, labels, target_words_clean
    )

    print('word_timestamps')
    print(word_timestamps)

    frame_labels = generate_frame_labels(
        word_timestamps=word_timestamps,
        total_frames=total_frames,
        fps=fps,
        reaction_offset_sec=reaction_offset_sec,
    )

    # 1. Define class order (K classes)
    k = len(DEFAULT_LABELS)
    label_to_idx = {label: i for i, label in enumerate(DEFAULT_LABELS)}

    # 2. Map string frame labels to integer indices (0 to K-1)
    # (If frame_labels is already an integer array, skip this map step)
    numeric_indices = np.array([label_to_idx[label] for label in frame_labels])

    # 3. Create the N x K one-hot matrix
    N = len(frame_labels)
    frame_matrix = np.zeros((N, k), dtype=np.float32)
    frame_matrix[np.arange(N), numeric_indices] = 1.0


    np.save(output_npy_path, frame_matrix)
    print(f"Saved frame state matrix to {output_npy_path}")
    print(frame_labels)
    print(frame_matrix)
    return frame_matrix


if __name__ == "__main__":
    print('start')
    parser = argparse.ArgumentParser(
        description="Extract frame-accurate action state labels directly from MP4 videos using Wav2Vec2."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to video file (.mp4)")
    parser.add_argument("--output", type=str, required=True, help="Output path for .npy file")
    parser.add_argument("--fps", type=float, default=None, help="Override video framerate (optional)")
    parser.add_argument("--frames", type=int, default=None, help="Override total video frame count (optional)")
    parser.add_argument(
        "--offset",
        type=float,
        default=-0.25,
        help="Motor reaction offset in seconds (default: -0.25)",
    )

    args = parser.parse_args()

    process_pipeline(
        video_path=args.video,
        output_npy_path=args.output,
        fps=args.fps,
        total_frames=args.frames,
        reaction_offset_sec=args.offset,
    )
