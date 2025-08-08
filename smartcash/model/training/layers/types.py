"""
Type hints and data classes for layer management.

This module provides type hints and data classes used throughout the layer
management system to ensure type safety and improve code clarity.
"""

from typing import Dict, List, Tuple, Union, Any, Optional
from dataclasses import dataclass

# Type aliases
LayerName = str
LayerMode = str  # 'single' or 'multilayer'
LayerOffset = Tuple[int, int]  # (start_idx, end_idx)

@dataclass
class LayerConfig:
    """Configuration for a single detection layer."""
    name: str
    num_classes: int
    description: str = ""
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            'name': self.name,
            'num_classes': self.num_classes,
            'description': self.description,
            'active': self.active
        }

@dataclass
class LayerMetrics:
    """Metrics for a single detection layer."""
    layer_name: str
    total_detections: int
    avg_confidence: float
    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision
    class_distribution: Dict[int, int]
    confidence_distribution: Dict[float, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            'layer_name': self.layer_name,
            'total_detections': self.total_detections,
            'avg_confidence': self.avg_confidence,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'ap': self.ap,
            'class_distribution': self.class_distribution,
            'confidence_distribution': self.confidence_distribution
        }

# Layer validation result types
LayerValidationResult = Dict[str, Union[bool, str, Dict[str, Any]]]
