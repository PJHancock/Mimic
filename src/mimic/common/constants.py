"""Shared constants across all modules."""

# Action phases
ACTION_PHASES = ["IDLE", "HOVER", "GRASP", "CARRY", "RELEASE"]

# Default heights for Panda (in meters)
DEFAULT_APPROACH_HEIGHT = 0.15
DEFAULT_GRASP_HEIGHT = 0.0
DEFAULT_TRANSPORT_HEIGHT = 0.15
DEFAULT_RELEASE_HEIGHT = 0.0

# Video processing
DEFAULT_FPS = 30
DEFAULT_VIDEO_EXTENSION = ".mp4"

# Panda robot specifications
PANDA_WORKSPACE_X_MIN = 0.2
PANDA_WORKSPACE_X_MAX = 0.8
PANDA_WORKSPACE_Y_MIN = -0.4
PANDA_WORKSPACE_Y_MAX = 0.4
PANDA_WORKSPACE_Z_MIN = 0.0
PANDA_WORKSPACE_Z_MAX = 0.5

# Embedding dimensions
VJEPA_EMBEDDING_DIM = 1024  # Adjust based on actual V-JEPA 2 output

# Tracking confidence thresholds
HAND_TRACK_CONFIDENCE_THRESHOLD = 0.5
OBJECT_TRACK_CONFIDENCE_THRESHOLD = 0.5

# Cache settings
CACHE_EMBEDDINGS = True
CACHE_TRACKS = True
EMBEDDING_CACHE_DIR = "data/embeddings"
TRACK_CACHE_DIR = "data/tracks"

# Calibration: table dimensions (in meters)
# Measured from actual poster board setup
TABLE_WIDTH_M = 0.508   # 20 inches in meters
TABLE_HEIGHT_M = 0.762  # 30 inches in meters
