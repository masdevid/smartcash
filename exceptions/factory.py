# File: smartcash/exceptions/factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory untuk pembuatan dan pengelolaan error secara terpusat

from typing import Optional, Dict, Type, Any
from .base import (
    SmartCashError, ConfigError, DataError, ModelError,
    TrainingError, EvaluationError, PreprocessingError,
    ValidationError, ResourceError
)

class ErrorFactory:
    """Factory untuk membuat error SmartCash dengan cara yang konsisten."""
    
    # Mapping dari kode error ke tipe exception
    ERROR_TYPES: Dict[str, Type[SmartCashError]] = {
        'config': ConfigError,
        'data': DataError,
        'model': ModelError,
        'training': TrainingError,
        'evaluation': EvaluationError,
        'preprocessing': PreprocessingError,
        'validation': ValidationError,
        'resource': ResourceError,
        'general': SmartCashError
    }
    
    @classmethod
    def create(
        cls, 
        error_type: str, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> SmartCashError:
        """
        Buat error dengan tipe yang ditentukan.
        
        Args:
            error_type: Tipe error ('config', 'data', dll)
            message: Pesan error utama
            details: Detail tambahan terkait error (opsional)
            
        Returns:
            Instance SmartCashError yang sesuai
        """
        # Tambahkan detail ke pesan jika disediakan
        if details:
            detail_str = ", ".join(f"{k}: {v}" for k, v in details.items())
            full_message = f"{message} ({detail_str})"
        else:
            full_message = message
            
        # Tentukan tipe exception
        exception_class = cls.ERROR_TYPES.get(error_type.lower(), SmartCashError)
        
        # Buat exception
        return exception_class(full_message)
    
    @classmethod
    def from_exception(
        cls, 
        exception: Exception, 
        error_type: Optional[str] = None,
        additional_message: Optional[str] = None
    ) -> SmartCashError:
        """
        Konversi exception standar/lainnya ke SmartCashError.
        
        Args:
            exception: Exception asli
            error_type: Tipe error SmartCash (opsional)
            additional_message: Pesan tambahan (opsional)
            
        Returns:
            Instance SmartCashError yang sesuai
        """
        # Tentukan tipe error berdasarkan exception
        if error_type is None:
            if isinstance(exception, FileNotFoundError):
                error_type = 'data'
            elif isinstance(exception, PermissionError):
                error_type = 'resource'
            elif isinstance(exception, KeyError) or isinstance(exception, ValueError):
                error_type = 'validation'
            else:
                error_type = 'general'
        
        # Buat pesan
        if additional_message:
            message = f"{additional_message}: {str(exception)}"
        else:
            message = str(exception)
            
        # Buat error
        return cls.create(error_type, message)