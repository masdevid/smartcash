"""
File: smartcash/common/constants/file_types.py
Deskripsi: Konstanta untuk tipe file, ekstensi, dan format yang didukung
"""

from smartcash.common.constants.enums import ModelFormat

# File extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

# Model file extensions
MODEL_EXTENSIONS = {
    ModelFormat.PYTORCH: '.pt',
    ModelFormat.ONNX: '.onnx',
    ModelFormat.TORCHSCRIPT: '.pt',
    ModelFormat.TENSORRT: '.engine',
    ModelFormat.TFLITE: '.tflite'
}