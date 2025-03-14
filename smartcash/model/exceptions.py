"""
File: smartcash/model/exceptions.py
Deskripsi: Komponen untuk exceptions pada model deteksi objek
"""

class ModelError(Exception):
    """Exception dasar untuk semua error terkait model"""
    pass
class ModelConfigurationError(ModelError):
    """Error konfigurasi model"""
    pass
class ModelTrainingError(ModelError):
    """Error saat proses training model"""
    pass
class ModelInferenceError(ModelError):
    """Error saat melakukan inferensi dengan model"""
    pass
class ModelCheckpointError(ModelError):
    """Error terkait checkpoint model"""
    pass
class ModelExportError(ModelError):
    """Error saat mengekspor model"""
    pass
class ModelEvaluationError(ModelError):
    """Eksepsi untuk kesalahan selama proses evaluasi model."""
    pass
class ModelServiceError(ModelError):
    """Error saat menggunakan model service"""
    pass
class BackboneError(ModelError):
    """Error terkait backbone model"""
    pass
class UnsupportedBackboneError(BackboneError):
    """Eksepsi untuk jenis backbone yang tidak didukung."""
    pass
class DatasetCompatibilityError(ModelError):
    """Eksepsi untuk masalah kompatibilitas dataset dengan model."""
    pass
class ExperimentError(ModelError):
    """Eksepsi untuk kesalahan dalam manajemen eksperimen."""
    pass
class NeckError(ModelError):
    """Error terkait neck model"""
    pass
class HeadError(ModelError):
    """Error terkait detection head model"""
    pass