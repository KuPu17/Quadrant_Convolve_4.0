"""
Disaster Response Intelligence System Configuration
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
SRC_DIR = BASE_DIR / "src"
OUTPUT_DIR = BASE_DIR / "output"
CONFIG_DIR = BASE_DIR / "config"
TESTS_DIR = BASE_DIR / "tests"

QDRANT_CONFIG = {
    "path": str(BASE_DIR / "qdrant_storage"),
    "collection_name": "disaster_response_memory",
    "vector_size": 512,
    "distance_metric": "cosine",
}

EMBEDDING_CONFIG = {
    "embedding_dim": 512,
    "model_type": "hybrid",
    "text_model": "clip",
    "image_model": "clip",
    "audio_model": "librosa",
}

MEMORY_CONFIG = {
    "decay_factor": 0.95,
    "decay_period_hours": 24,
    "max_session_interactions": 100,
    "memory_pruning_threshold": 0.1,
}

RETRIEVAL_CONFIG = {
    "default_limit": 10,
    "score_threshold": 0.3,
    "enable_reranking": True,
    "temporal_window_days": 7,
}

DISASTER_TYPES = [
    "earthquake",
    "flood",
    "wildfire",
    "hurricane",
    "landslide",
    "tsunami",
]

SEVERITY_LEVELS = {
    1: "Minor",
    2: "Moderate",
    3: "Severe",
    4: "Critical",
    5: "Catastrophic",
}

DATA_MODALITIES = ["text", "image", "audio"]

def ensure_directories():
    for directory in [OUTPUT_DIR, CONFIG_DIR, TESTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

ensure_directories()