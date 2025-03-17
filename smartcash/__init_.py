"""
File: smartcash/__init__.py
Deskripsi: File inisialisasi untuk package SmartCash dengan impor exceptions untuk akses global
"""

from smartcash.common.exceptions import (
    # Exception dasar
    SmartCashError,
    
    # Exception Config
    ConfigError,
    
    # Exception Dataset
    DatasetError, DatasetFileError, DatasetValidationError, 
    DatasetProcessingError, DatasetCompatibilityError,
    
    # Exception Model
    ModelError, ModelConfigurationError, ModelTrainingError,
    ModelInferenceError, ModelCheckpointError, ModelExportError,
    ModelEvaluationError, ModelServiceError,
    
    # Exception Model Components
    ModelComponentError, BackboneError, UnsupportedBackboneError,
    NeckError, HeadError,
    
    # Exception Detection
    DetectionError, DetectionInferenceError, DetectionPostprocessingError,
    
    # Exception I/O
    FileError,
    
    # Exception API & Validation
    APIError, ValidationError,
    
    # Exception Lainnya
    NotSupportedError, ExperimentError
)

__version__ = '2.0.0'