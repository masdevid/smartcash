# smartcash/common/exceptions.py
"""
File: smartcash/common/exceptions.py
Deskripsi: Custom exceptions untuk SmartCash
"""

class SmartCashError(Exception):
    """Exception dasar untuk semua error SmartCash."""
    pass

class ConfigError(SmartCashError):
    """Exception untuk error konfigurasi."""
    pass

class DatasetError(SmartCashError):
    """Exception untuk error terkait dataset."""
    pass

class ModelError(SmartCashError):
    """Exception untuk error terkait model."""
    pass

class DetectionError(SmartCashError):
    """Exception untuk error terkait proses deteksi."""
    pass

class FileError(SmartCashError):
    """Exception untuk error file I/O."""
    pass

class APIError(SmartCashError):
    """Exception untuk error API."""
    pass

class ValidationError(SmartCashError):
    """Exception untuk error validasi input."""
    pass

class NotSupportedError(SmartCashError):
    """Exception untuk fitur yang tidak didukung."""
    pass
