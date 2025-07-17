"""
File: smartcash/common/__init__.py
Deskripsi: Inisialisasi dan ekspor modul-modul umum untuk SmartCash dengan pengelompokan yang lebih terstruktur
"""

# ===== Konfigurasi =====
from smartcash.common.config import (
    get_config_manager,
    SimpleConfigManager,
    get_module_config
)

# Alias untuk kompatibilitas
ConfigManager = SimpleConfigManager

# ===== Kompatibilitas OpenCV =====
try:
    from smartcash.common.opencv_compat import CV_8U, threshold as cv2_threshold, convert_to_uint8
except ImportError:
    # Fallback if OpenCV is not available
    CV_8U = None
    cv2_threshold = None
    convert_to_uint8 = None

# ===== Lingkungan =====
from smartcash.common.environment import (
    get_environment_manager,
    EnvironmentManager
)

# ===== Konfigurasi Layer =====
from smartcash.common.layer_config import (
    get_layer_config,
    LayerConfigManager
)

# ===== Logging =====
from smartcash.common.logger import (
    get_logger,
    SmartCashLogger,
    LogLevel
)

# ===== Interfaces =====
from smartcash.common.interfaces import (
    ILayerConfigManager,
)

# ===== Utilitas =====
from smartcash.common.utils import (
    is_colab, 
    is_notebook, 
    get_system_info, 
    generate_unique_id, 
    format_time, 
    get_timestamp,
    get_project_root,
    load_json,
    save_json,
    load_yaml,
    save_yaml
)

# ===== Tipe Data =====
from smartcash.common.types import (
    ImageType, 
    PathType, 
    TensorType, 
    ConfigType, 
    ProgressCallback, 
    LogCallback,
    BoundingBox, 
    Detection, 
    ModelInfo, 
    DatasetStats
)

# ===== Exceptions =====
from smartcash.common.exceptions import (
    SmartCashError, 
    ConfigError, 
    DatasetError, 
    ModelError,
    DetectionError, 
    FileError, 
    APIError, 
    ValidationError,
    DatasetFileError,
    DatasetValidationError,
    DatasetProcessingError,
    DatasetCompatibilityError,
    ModelConfigurationError,
    ModelTrainingError,
    ModelInferenceError,
    ModelCheckpointError,
    ModelExportError,
    ModelEvaluationError,
    ModelServiceError,
    ModelComponentError,
    BackboneError,
    UnsupportedBackboneError,
    NeckError,
    HeadError,
    DetectionInferenceError,
    DetectionPostprocessingError,
    NotSupportedError,
    ExperimentError
)

# ===== Threading dan Multiprocessing =====
from smartcash.common.threadpools import (
    get_optimal_thread_count, 
    process_in_parallel, 
    process_with_stats
)

# ===== Konstanta =====
from smartcash.common.constants.core import (
    VERSION,
    APP_NAME,
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_LOGS_DIR,
    DEFAULT_OUTPUT_DIR
)

from smartcash.common.constants.dataset import (
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_VAL_SPLIT,
    DEFAULT_TEST_SPLIT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_AUGMENTATION_MULTIPLIER,
    DEFAULT_AUGMENTATIONS,
    DATA_VERSION_FORMAT,
    DEFAULT_DATA_VERSION,
    PATH_TEMPLATES
)

from smartcash.common.constants.enums import (
    DetectionLayer,
    ModelFormat
)

from smartcash.common.constants.file_types import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    MODEL_EXTENSIONS
)

from smartcash.common.constants.model import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    MAX_BATCH_SIZE,
    MAX_IMAGE_SIZE
)

from smartcash.common.constants.paths import (
    BASE_PATH,
    COLAB_PATH,
    PREPROCESSED_DIR,
    AUGMENTED_DIR,
    BACKUP_DIR,
    INVALID_DIR,
    VISUALIZATION_DIR,
    CHECKPOINT_DIR,
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    DATA_ROOT,
    DOWNLOADS_DIR,
    COLAB_DATA_ROOT,
    COLAB_DOWNLOADS_PATH,
    COLAB_PREPROCESSED_PATH,
    COLAB_AUGMENTED_PATH,
    COLAB_BACKUP_PATH,
    COLAB_INVALID_PATH,
    COLAB_VISUALIZATION_PATH,
    COLAB_TRAIN_PATH,
    COLAB_VALID_PATH,
    COLAB_TEST_PATH
)

# Daftar ekspor lengkap
__all__ = [
    # Kompatibilitas OpenCV
    'CV_8U',
    'cv2_threshold',
    'convert_to_uint8',
    
    # Konfigurasi
    'get_config_manager',
    'SimpleConfigManager',
    'ConfigManager',  # Alias untuk kompatibilitas
    'get_module_config',
    
    # Lingkungan
    'get_environment_manager',
    'EnvironmentManager',
    
    # Konfigurasi Layer
    'get_layer_config',
    'LayerConfigManager',
    
    # Logging
    'get_logger',
    'SmartCashLogger',
    'LogLevel',
    
    # Interfaces
    'IDetectionVisualizer',
    'IMetricsVisualizer',
    'ILayerConfigManager',
    'ICheckpointService',
    
    # Utilitas
    'is_colab',
    'is_notebook',
    'get_system_info',
    'generate_unique_id',
    'format_time',
    'get_timestamp',
    'get_project_root',
    'load_json',
    'save_json',
    'load_yaml',
    'save_yaml',
    
    # Tipe Data
    'ImageType',
    'PathType',
    'TensorType',
    'ConfigType',
    'ProgressCallback',
    'LogCallback',
    'BoundingBox',
    'Detection',
    'ModelInfo',
    'DatasetStats',
    
    # Exceptions
    'SmartCashError',
    'ConfigError',
    'DatasetError',
    'ModelError',
    'DetectionError',
    'FileError',
    'APIError',
    'ValidationError',
    'DatasetFileError',
    'DatasetValidationError',
    'DatasetProcessingError',
    'DatasetCompatibilityError',
    'ModelConfigurationError',
    'ModelTrainingError',
    'ModelInferenceError',
    'ModelCheckpointError',
    'ModelExportError',
    'ModelEvaluationError',
    'ModelServiceError',
    'ModelComponentError',
    'BackboneError',
    'UnsupportedBackboneError',
    'NeckError',
    'HeadError',
    'DetectionInferenceError',
    'DetectionPostprocessingError',
    'NotSupportedError',
    'ExperimentError',
    
    # Threading dan Multiprocessing
    'get_optimal_thread_count',
    'process_in_parallel',
    'process_with_stats',
    
    # Konstanta Core
    'VERSION',
    'APP_NAME',
    'DEFAULT_CONFIG_DIR',
    'DEFAULT_DATA_DIR',
    'DEFAULT_MODEL_DIR',
    'DEFAULT_LOGS_DIR',
    'DEFAULT_OUTPUT_DIR',
    
    # Konstanta Dataset
    'DEFAULT_TRAIN_SPLIT',
    'DEFAULT_VAL_SPLIT',
    'DEFAULT_TEST_SPLIT',
    'DEFAULT_IMAGE_SIZE',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_AUGMENTATION_MULTIPLIER',
    'DEFAULT_AUGMENTATIONS',
    'DATA_VERSION_FORMAT',
    'DEFAULT_DATA_VERSION',
    'PATH_TEMPLATES',
    
    # Konstanta Enums
    'DetectionLayer',
    'ModelFormat',
    
    # Konstanta File Types
    'IMAGE_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'MODEL_EXTENSIONS',
    
    # Konstanta Model
    'DEFAULT_CONF_THRESHOLD',
    'DEFAULT_IOU_THRESHOLD',
    'MAX_BATCH_SIZE',
    'MAX_IMAGE_SIZE',
    
    # Konstanta Paths
    'BASE_PATH',
    'COLAB_PATH',
    'PREPROCESSED_DIR',
    'AUGMENTED_DIR',
    'BACKUP_DIR',
    'INVALID_DIR',
    'VISUALIZATION_DIR',
    'CHECKPOINT_DIR',
    'TRAIN_DIR',
    'VALID_DIR',
    'TEST_DIR',
    'DATA_ROOT',
    'DOWNLOADS_DIR',
    'COLAB_DATA_ROOT',
    'COLAB_DOWNLOADS_PATH',
    'COLAB_PREPROCESSED_PATH',
    'COLAB_AUGMENTED_PATH',
    'COLAB_BACKUP_PATH',
    'COLAB_INVALID_PATH',
    'COLAB_VISUALIZATION_PATH',
    'COLAB_TRAIN_PATH',
    'COLAB_VALID_PATH',
    'COLAB_TEST_PATH'
]