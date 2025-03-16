# smartcash/common/interfaces/visualization_interface.py
"""
Interface untuk visualisasi yang digunakan oleh domain Detection dan Model
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional

class IDetectionVisualizer(ABC):
    @abstractmethod
    def visualize_detection(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                           filename: Optional[str] = None, conf_threshold: float = 0.25) -> np.ndarray:
        """Visualisasi hasil deteksi pada gambar"""
        pass

class IMetricsVisualizer(ABC):
    @abstractmethod
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                             title: str, filename: Optional[str] = None):
        """Plot confusion matrix"""
        pass