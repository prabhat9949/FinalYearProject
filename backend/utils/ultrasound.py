"""
Ultrasound pipeline integration.

Connects the FastAPI backend with the Byra/IJCARS
and BEHSOF ultrasound models.
"""

import sys
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
IMAGE_MODEL_DIR = PROJECT_ROOT / "image-model"


# Add image-model to Python import path
if str(IMAGE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGE_MODEL_DIR))


# Import ultrasound pipeline
from image_model.pipeline.ultrasound_pipeline import UltrasoundPipeline



# ---------------------------------------------------------
# Singleton pipeline
# ---------------------------------------------------------

_ultrasound_pipeline = None


def get_ultrasound_pipeline():
    """
    Load the ultrasound models once and reuse them.
    """

    global _ultrasound_pipeline

    if _ultrasound_pipeline is None:
        _ultrasound_pipeline = UltrasoundPipeline()

    return _ultrasound_pipeline


def analyze_ultrasound(image_path: str):
    """
    Analyze an ultrasound image using both models.
    """

    pipeline = get_ultrasound_pipeline()

    return pipeline.analyze(image_path)