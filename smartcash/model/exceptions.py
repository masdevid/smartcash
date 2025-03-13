"""
File: smartcash/model/exceptions.py
Deskripsi: Definisi kelas exception khusus untuk modul model
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

class ModelServiceError(ModelError):
    """Error saat menggunakan model service"""
    pass

class BackboneError(ModelError):
    """Error terkait backbone model"""
    pass

class NeckError(ModelError):
    """Error terkait neck model"""
    pass

class HeadError(ModelError):
    """Error terkait detection head model"""
    pass