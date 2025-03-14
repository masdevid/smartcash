# smartcash/common/types.py
"""
File: smartcash/common/types.py
Deskripsi: Type definitions untuk SmartCash
"""

from typing import Dict, List, Tuple, Union, Optional, Any, TypedDict, NewType, Callable
import numpy as np
import torch
from enum import Enum, auto

# Type aliases
ImageType = Union[np.ndarray, str, bytes]
PathType = Union[str, 'Path']
TensorType = Union[torch.Tensor, np.ndarray]
ConfigType = Dict[str, Any]

# Callback types
ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str, str], None]

# Typed dictionaries
class BoundingBox(TypedDict):
    """Bounding box dengan format [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    
class Detection(TypedDict):
    """Hasil deteksi objek."""
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    layer: str

class ModelInfo(TypedDict):
    """Informasi model."""
    name: str
    version: str
    format: str
    input_size: Tuple[int, int]
    layers: List[str]
    classes: Dict[str, List[str]]
    
class DatasetStats(TypedDict):
    """Statistik dataset."""
    total_images: int
    total_annotations: int
    class_distribution: Dict[str, int]
    split_info: Dict[str, Dict[str, int]]