"""
File: smartcash/common/constants/enums.py
Deskripsi: Enum dan tipe data terdefinisi untuk aplikasi
"""

from enum import Enum, auto

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