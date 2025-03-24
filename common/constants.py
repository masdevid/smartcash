"""
File: smartcash/common/constants.py
Deskripsi: Konstanta global yang digunakan di seluruh project
"""

from enum import Enum, auto
from pathlib import Path
import os

# Versi aplikasi
VERSION = "0.1.0"
APP_NAME = "SmartCash"

# Paths
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODEL_DIR = "models"
DEFAULT_LOGS_DIR = "logs"


# Layer detection
class DetectionLayer(Enum):
    """Layer untuk deteksi objek."""
    BANKNOTE = "banknote"  # Deteksi uang kertas utuh
    NOMINAL = "nominal"    # Deteksi area nominal
    SECURITY = "security"  # Deteksi fitur keamanan

# Format Input/Output
class ModelFormat(Enum):
    """Format model yang didukung."""
    PYTORCH = auto()
    ONNX = auto()
    TORCHSCRIPT = auto()
    TENSORRT = auto()
    TFLITE = auto()

# File extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
MODEL_EXTENSIONS = {
    ModelFormat.PYTORCH: '.pt',
    ModelFormat.ONNX: '.onnx',
    ModelFormat.TORCHSCRIPT: '.pt',
    ModelFormat.TENSORRT: '.engine',
    ModelFormat.TFLITE: '.tflite'
}

# Environment variables
ENV_CONFIG_PATH = os.environ.get("SMARTCASH_CONFIG_PATH", "")
ENV_MODEL_PATH = os.environ.get("SMARTCASH_MODEL_PATH", "")
ENV_DATA_PATH = os.environ.get("SMARTCASH_DATA_PATH", "")

# Default values
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

# Limits
MAX_BATCH_SIZE = 64
MAX_IMAGE_SIZE = 1280

# API settings (if applicable)
API_PORT = 8000
API_HOST = "0.0.0.0"