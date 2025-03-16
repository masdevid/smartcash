"""
File: smartcash/common/exceptions.py
Deskripsi: Hierarki exception terpadu untuk seluruh komponen SmartCash
"""

class SmartCashError(Exception):
    """Exception dasar untuk semua error SmartCash."""
    pass


# =============== Exception Config ===============
class ConfigError(SmartCashError):
    """Exception untuk error konfigurasi."""
    pass


# =============== Exception Dataset ===============
class DatasetError(SmartCashError):
    """Exception untuk error terkait dataset."""
    pass


class DatasetFileError(DatasetError):
    """Exception untuk error file dataset."""
    pass


class DatasetValidationError(DatasetError):
    """Exception untuk error validasi dataset."""
    pass


class DatasetProcessingError(DatasetError):
    """Exception untuk error pemrosesan dataset."""
    pass


class DatasetCompatibilityError(DatasetError):
    """Exception untuk masalah kompatibilitas dataset dengan model."""
    pass


# =============== Exception Model ===============
class ModelError(SmartCashError):
    """Exception untuk error terkait model."""
    pass


class ModelConfigurationError(ModelError):
    """Error konfigurasi model."""
    pass


class ModelTrainingError(ModelError):
    """Error saat proses training model."""
    pass


class ModelInferenceError(ModelError):
    """Error saat melakukan inferensi dengan model."""
    pass


class ModelCheckpointError(ModelError):
    """Error terkait checkpoint model."""
    pass


class ModelExportError(ModelError):
    """Error saat mengekspor model."""
    pass


class ModelEvaluationError(ModelError):
    """Error selama proses evaluasi model."""
    pass


class ModelServiceError(ModelError):
    """Error saat menggunakan model service."""
    pass


# =============== Exception Model Components ===============
class ModelComponentError(ModelError):
    """Exception dasar untuk error komponen model."""
    pass


class BackboneError(ModelComponentError):
    """Error terkait backbone model."""
    pass


class UnsupportedBackboneError(BackboneError):
    """Error untuk jenis backbone yang tidak didukung."""
    pass


class NeckError(ModelComponentError):
    """Error terkait neck model."""
    pass


class HeadError(ModelComponentError):
    """Error terkait detection head model."""
    pass


# =============== Exception Detection ===============
class DetectionError(SmartCashError):
    """Exception untuk error terkait proses deteksi."""
    pass


class DetectionInferenceError(DetectionError):
    """Error saat inferensi pada proses deteksi."""
    pass


class DetectionPostprocessingError(DetectionError):
    """Error saat post-processing hasil deteksi."""
    pass


# =============== Exception I/O ===============
class FileError(SmartCashError):
    """Exception untuk error file I/O."""
    pass


# =============== Exception API & Validation ===============
class APIError(SmartCashError):
    """Exception untuk error API."""
    pass


class ValidationError(SmartCashError):
    """Exception untuk error validasi input."""
    pass


# =============== Exception Lainnya ===============
class NotSupportedError(SmartCashError):
    """Exception untuk fitur yang tidak didukung."""
    pass


class ExperimentError(SmartCashError):
    """Exception untuk kesalahan dalam manajemen eksperimen."""
    pass