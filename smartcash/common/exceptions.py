"""
File: smartcash/common/exceptions.py
Deskripsi: Definisi hierarki exception terpadu untuk seluruh komponen SmartCash
"""
from typing import Any, Dict, Optional, Type, Union
from dataclasses import dataclass

@dataclass
class ErrorContext:
    """Context information for better error handling and reporting."""
    component: str = ""
    operation: str = ""
    details: Optional[Dict[str, Any]] = None
    ui_components: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        return {
            'component': self.component,
            'operation': self.operation,
            'details': self.details or {}
        }

# Exception Classes Dasar
class SmartCashError(Exception):
    """Base exception for all SmartCash errors with context support."""
    def __init__(
        self, 
        message: str = "Terjadi error pada sistem SmartCash",
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext()
        super().__init__(self.message)
        
    def with_context(self, **kwargs) -> 'SmartCashError':
        """Add context to the error and return self for chaining."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
        return self

# Exception Config
class ConfigError(SmartCashError):
    """Error pada konfigurasi"""
    def __init__(self, message="Error pada konfigurasi SmartCash"):
        super().__init__(message)

# Exception Dataset
class DatasetError(SmartCashError):
    """Base exception for dataset-related errors."""
    def __init__(
        self, 
        message: str = "Error pada dataset",
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        dataset_path: Optional[str] = None
    ):
        context = context or ErrorContext()
        if dataset_path:
            context.details = {**(context.details or {}), 'dataset_path': dataset_path}
        super().__init__(message, error_code, context)

class DatasetFileError(DatasetError):
    """Error related to dataset file operations."""
    def __init__(
        self,
        message: str = "Error pada file dataset",
        file_path: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        if file_path:
            context.details = {**(context.details or {}), 'file_path': file_path}
        super().__init__(message, **kwargs, context=context)

class DatasetValidationError(DatasetError):
    """Error during dataset validation."""
    def __init__(
        self,
        message: str = "Error validasi dataset",
        validation_errors: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        if validation_errors:
            context.details = {**(context.details or {}), 'validation_errors': validation_errors}
        super().__init__(message, **kwargs, context=context)
        
class DatasetProcessingError(DatasetError):
    """Error during dataset processing."""
    def __init__(
        self,
        message: str = "Error saat memproses dataset",
        processing_stage: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        if processing_stage:
            context.details = {**(context.details or {}), 'processing_stage': processing_stage}
        super().__init__(message, **kwargs, context=context)

# UI Exceptions
class UIError(SmartCashError):
    """Base exception for UI-related errors."""
    def __init__(
        self,
        message: str = "Terjadi kesalahan pada antarmuka pengguna",
        component: str = "unknown",
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        context.component = component or context.component or "unknown"
        super().__init__(message, **kwargs, context=context)

class UIComponentError(UIError):
    """Error related to UI component initialization or operation."""
    def __init__(
        self,
        message: str = "Kesalahan komponen antarmuka",
        component_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        if component_type:
            context.details = {**(context.details or {}), 'component_type': component_type}
        super().__init__(message, **kwargs, context=context)

class UIActionError(UIError):
    """Error during UI action execution."""
    def __init__(
        self,
        message: str = "Gagal menjalankan aksi",
        action: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', ErrorContext())
        if action:
            context.operation = action
        super().__init__(message, **kwargs, context=context)
        super().__init__(message)

class DatasetCompatibilityError(DatasetError):
    """Masalah kompatibilitas dataset dengan model"""
    def __init__(self, message="Dataset tidak kompatibel dengan model"):
        super().__init__(message)

# Exception Model
class ModelError(SmartCashError):
    """Error dasar terkait model"""
    def __init__(self, message="Error pada model"):
        super().__init__(message)

class ModelConfigurationError(ModelError):
    """Error konfigurasi model"""
    def __init__(self, message="Error konfigurasi model"):
        super().__init__(message)

class ModelTrainingError(ModelError):
    """Error proses training model"""
    def __init__(self, message="Error saat training model"):
        super().__init__(message)

class ModelInferenceError(ModelError):
    """Error inferensi model"""
    def __init__(self, message="Error saat inferensi model"):
        super().__init__(message)

class ModelCheckpointError(ModelError):
    """Error checkpoint model"""
    def __init__(self, message="Error pada checkpoint model"):
        super().__init__(message)

class ModelExportError(ModelError):
    """Error ekspor model"""
    def __init__(self, message="Error saat ekspor model"):
        super().__init__(message)

class ModelEvaluationError(ModelError):
    """Error evaluasi model"""
    def __init__(self, message="Error saat evaluasi model"):
        super().__init__(message)

class ModelServiceError(ModelError):
    """Error model service"""
    def __init__(self, message="Error pada model service"):
        super().__init__(message)

# Exception Model Components
class ModelComponentError(ModelError):
    """Error dasar komponen model"""
    def __init__(self, message="Error pada komponen model"):
        super().__init__(message)

class BackboneError(ModelComponentError):
    """Error backbone model"""
    def __init__(self, message="Error pada backbone model"):
        super().__init__(message)

class UnsupportedBackboneError(BackboneError):
    """Error backbone tidak didukung"""
    def __init__(self, message="Backbone model tidak didukung"):
        super().__init__(message)

class NeckError(ModelComponentError):
    """Error neck model"""
    def __init__(self, message="Error pada neck model"):
        super().__init__(message)

class HeadError(ModelComponentError):
    """Error detection head model"""
    def __init__(self, message="Error pada detection head model"):
        super().__init__(message)

# Exception Detection
class DetectionError(SmartCashError):
    """Error dasar proses deteksi"""
    def __init__(self, message="Error pada proses deteksi"):
        super().__init__(message)

class DetectionInferenceError(DetectionError):
    """Error inferensi deteksi"""
    def __init__(self, message="Error pada inferensi deteksi"):
        super().__init__(message)

class DetectionPostprocessingError(DetectionError):
    """Error post-processing deteksi"""
    def __init__(self, message="Error saat post-processing deteksi"):
        super().__init__(message)

# Exception I/O
class FileError(SmartCashError):
    """Error file I/O"""
    def __init__(self, message="Error pada operasi file"):
        super().__init__(message)

# Exception API & Validation
class APIError(SmartCashError):
    """Error API"""
    def __init__(self, message="Error pada API"):
        super().__init__(message)

class ValidationError(SmartCashError):
    """Error validasi input"""
    def __init__(self, message="Error validasi input"):
        super().__init__(message)

# Exception Lainnya
class NotSupportedError(SmartCashError):
    """Fitur tidak didukung"""
    def __init__(self, message="Fitur ini tidak didukung"):
        super().__init__(message)
        
class ExperimentError(SmartCashError):
    """Error manajemen eksperimen"""
    def __init__(self, message="Error pada manajemen eksperimen"):
        super().__init__(message)