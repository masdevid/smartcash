"""
File: smartcash/detection/adapters/__init__.py
Deskripsi: Export adapter untuk berbagai format model.
"""

from smartcash.detection.adapters.onnx_adapter import ONNXModelAdapter
from smartcash.detection.adapters.torchscript_adapter import TorchScriptAdapter

__all__ = [
    'ONNXModelAdapter',
    'TorchScriptAdapter'
]