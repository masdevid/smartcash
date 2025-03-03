# File: smartcash/exceptions/base.py
# Author: Alfrida Sabar
# Deskripsi: Base exceptions untuk SmartCash

class SmartCashError(Exception):
    """Base exception untuk semua error dalam SmartCash."""
    
    def __init__(self, message: str, *args):
        self.message = message
        super().__init__(message, *args)

class ConfigError(SmartCashError):
    """Error terkait konfigurasi aplikasi."""
    pass

class DataError(SmartCashError):
    """Error terkait pemrosesan data."""
    pass

class ModelError(SmartCashError):
    """Error terkait model dan inferensi."""
    pass

class TrainingError(SmartCashError):
    """Error terkait proses training."""
    pass

class EvaluationError(SmartCashError):
    """Error terkait proses evaluasi."""
    pass

class PreprocessingError(SmartCashError):
    """Error terkait preprocessing data."""
    pass

class ValidationError(SmartCashError):
    """Error terkait validasi input atau konfigurasi."""
    pass

class ResourceError(SmartCashError):
    """Error terkait resource sistem (memori, GPU, dll)."""
    pass