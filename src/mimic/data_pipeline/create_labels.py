import math
import cv2
import numpy as np
import torch
import torchaudio
from pyctcdecode import build_ctcdecoder

# ==========================================
# 1. CONFIGURATION & STATE DEFINITIONS
# ==========================================
AUDIO_PATH = "input_video_audio.wav"
VIDEO_PATH = "input_video.mp4"
OUTPUT_NPY_PATH = "frame_states.npy"

# Allowed dynamic vocabulary (can be modified at runtime)
TARGET_STATES = ["approach", "grasp", "move", "release"]
DEFAULT_INITIAL_STATE = "idle"

# Human motor latency offset (e.g., 250ms = -0.25s)
REACTION_OFFSET_SEC = -0.25

# ==========================================
# 2. LOAD MODEL & BUILD CONSTRAINED DECODER
# ==========================================
def initialize_pipeline(state_vocabulary):
    """Loads Wav2Vec2 bundle and configures pyctcdecode with vocabulary constraints."""
    # Load light off-the-shelf bundle
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model()
    model.eval()
    
    # Clean vocabulary for CTC decoder (convert to uppercase for Wav2Vec2 mapping)
    labels = bundle.get_labels()
    
    # Build decoder constrained strictly to unigrams in our target list
    decoder = build_ctcdecoder(
        labels=labels,
        unigrams=[w.upper() for w in state_vocabulary]
    )
    
    return model, bundle, decoder


# ==========================================
# 3. ACOUSTIC EMISSION & DECODING
# ==========================================
def extract_state_timestamps(audio_path, model, bundle, decoder):
    """Runs Wav2Vec2 emissions pass and returns state words with time boundaries."""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz if necessary
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)
        waveform = resampler(waveform)

    # 1. Forward pass for emission logits
    with torch.inference_mode():
        emissions, _ = model(waveform)
        
    # Wav2Vec2 base frame stride is ~20ms (sample_rate / emission length ratio)
    time_stride = waveform.shape[1] / emissions.shape[1] / bundle.sample_rate

    # 2. Constrained CTC Decoding
    emission_logits = emissions[0].numpy()
    beams = decoder.decode_beams(emission_logits)
    
    # Extract recognized words along with timestamp boundaries
    time_events = []
    # Best beam output contains token/word timing tuples
    for word, (char_start, char_end) in beams[0][2]: 
        word_str = word.lower()
        start_time = char_start * time_stride
        time_events.append((word_str, start_time))
        
    # Sort events chronologically
    time_events.sort(key=lambda x: x[1])
    return time_events

# ==========================================
# 4. FRAME ALIGNMENT & STATE LATCHING
# ==========================================
def build_frame_state_vector(video_path, time_events, initial_state=DEFAULT_INITIAL_STATE):
    """Maps timestamped audio events to video frame indices using forward-latching."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Initialize state array
    state_vector = np.full(total_frames, initial_state, dtype=object)

    if not time_events:
        return state_vector

    current_state = initial_state
    
    # Convert timestamp events to frame indices
    event_frames = []
    for state_label, start_time in time_events:
        # Apply reaction offset and convert to integer frame index
        adjusted_time = max(0.0, start_time + REACTION_OFFSET_SEC)
        frame_idx = int(math.floor(adjusted_time * fps))
        if frame_idx < total_frames:
            event_frames.append((frame_idx, state_label))

    # Fill frame ranges (forward-latching state boundaries)
    last_frame = 0
    for frame_idx, new_state in event_frames:
        state_vector[last_frame:frame_idx] = current_state
        current_state = new_state
        last_frame = frame_idx
        
    # Fill remaining frames to end of video
    state_vector[last_frame:] = current_state

    return state_vector


# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    print("[1/3] Initializing Wav2Vec2 & pyctcdecode...")
    model, bundle, decoder = initialize_pipeline(TARGET_STATES)
    

    # print("[2/3] Extracting state speech events from audio...")
    # time_events = extract_state_timestamps(AUDIO_PATH, model, bundle, decoder)
    # print(f"Detected events: {time_events}")

    # print("[3/3] Aligning timestamps to video frames...")
    # state_vector = build_frame_state_vector(VIDEO_PATH, time_events)

    # # Save array
    # np.save(OUTPUT_NPY_PATH, state_vector)
    # print(f"Saved state alignment vector ({len(state_vector)} frames) to '{OUTPUT_NPY_PATH}'.")+