"""
File: smartcash/model/__init__.py
Deskripsi: Komponen utama model deteksi objek
"""

from smartcash.model.manager import ModelManager, YOLOv5Model, DETECTION_LAYERS
from smartcash.common.exceptions import (
    ModelError,
    ModelConfigurationError,
    ModelTrainingError,
    ModelInferenceError,
    ModelCheckpointError,
    ModelExportError,
    ModelServiceError,
    BackboneError,
    NeckError,
    HeadError
)

# Re-export dari architectures submodule
from smartcash.model.architectures.backbones import (
    BaseBackbone,
    EfficientNetBackbone,
    CSPDarknet
)
from smartcash.model.architectures.necks import (
    FeatureProcessingNeck,
    FeaturePyramidNetwork,
    PathAggregationNetwork
)
from smartcash.model.architectures.heads import DetectionHead

# Re-export dari components submodule
from smartcash.model.components import YOLOLoss

# Re-export dari visualization submodule
from smartcash.model.visualization import (
    MetricsVisualizer,
    DetectionVisualizer,
    ExperimentVisualizer,
    ScenarioVisualizer,
    ResearchVisualizer
)

__all__ = [
    # Core classes
    'ModelManager',
    'YOLOv5Model',
    'DETECTION_LAYERS',
    
    # Exceptions
    'ModelError',
    'ModelConfigurationError',
    'ModelTrainingError',
    'ModelInferenceError',
    'ModelCheckpointError',
    'ModelExportError',
    'ModelServiceError',
    'BackboneError',
    'NeckError',
    'HeadError',
    
    # Architectures - Backbones
    'BaseBackbone',
    'EfficientNetBackbone',
    'CSPDarknet',
    
    # Architectures - Necks
    'FeatureProcessingNeck',
    'FeaturePyramidNetwork',
    'PathAggregationNetwork',
    
    # Architectures - Heads
    'DetectionHead',
    
    # Components
    'YOLOLoss',
    
    # Visualization
    'MetricsVisualizer',
    'DetectionVisualizer',
    'ExperimentVisualizer',
    'ScenarioVisualizer',
    'ResearchVisualizer'
]