"""
File: smartcash/common/exceptions.py
Deskripsi: Definisi hierarki exception terpadu untuk seluruh komponen SmartCash
"""

# Exception Classes Dasar
class SmartCashError(Exception):
    """Exception dasar untuk semua error SmartCash"""
    def __init__(self, message="Terjadi error pada sistem SmartCash"):
        self.message = message
        super().__init__(self.message)

# Exception Config
class ConfigError(SmartCashError):
    """Error pada konfigurasi"""
    def __init__(self, message="Error pada konfigurasi SmartCash"):
        super().__init__(message)

# Exception Dataset
class DatasetError(SmartCashError):
    """Error dasar terkait dataset"""
    def __init__(self, message="Error pada dataset"):
        super().__init__(message)

class DatasetFileError(DatasetError):
    """Error file dataset"""
    def __init__(self, message="Error pada file dataset"):
        super().__init__(message)

class DatasetValidationError(DatasetError):
    """Error validasi dataset"""
    def __init__(self, message="Error validasi dataset"):
        super().__init__(message)
        
class DatasetProcessingError(DatasetError):
    """Error pemrosesan dataset"""
    def __init__(self, message="Error saat memproses dataset"):
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