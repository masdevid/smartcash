"""
File: smartcash/common/__init__.py
Deskripsi: Inisialisasi dan ekspor modul-modul umum untuk SmartCash
"""

# Konfigurasi
from smartcash.common.config import get_config_manager, ConfigManager
from smartcash.common.default_config import (
    generate_default_config, 
    ensure_base_config_exists, 
    ensure_colab_config_exists
)
from smartcash.common.config_sync import sync_config_with_drive, sync_all_configs

# Lingkungan
from smartcash.common.environment import get_environment_manager, EnvironmentManager

# Konfigurasi Layer
from smartcash.common.layer_config import get_layer_config, LayerConfigManager

# Logging
from smartcash.common.logger import get_logger, SmartCashLogger, LogLevel

# Utilitas
from smartcash.common.utils import (
    is_colab, is_notebook, 
    get_system_info, generate_unique_id, 
    format_time, get_timestamp
)

# Tipe Data dan Antarmuka
from smartcash.common.types import (
    ImageType, PathType, TensorType, ConfigType, 
    ProgressCallback, LogCallback,
    BoundingBox, Detection, ModelInfo, DatasetStats
)

# Exceptions
from smartcash.common.exceptions import (
    SmartCashError, ConfigError, DatasetError, ModelError,
    DetectionError, FileError, APIError, ValidationError
)

# Konstanta
from smartcash.common.constants import (
    VERSION, APP_NAME, 
    DEFAULT_CONFIG_DIR, DEFAULT_DATA_DIR, 
    DEFAULT_MODEL_DIR, DEFAULT_LOGS_DIR
)

# Threading dan Multiprocessing
from smartcash.common.threadpools import (
    get_optimal_thread_count, 
    process_in_parallel, 
    process_with_stats
)

# Inisialisasi Proyek
from smartcash.common.initialization import (
    initialize_config
)

__all__ = [
    # Konfigurasi
    'get_config_manager', 'ConfigManager',
    'generate_default_config', 
    'ensure_base_config_exists', 
    'ensure_colab_config_exists',
    'sync_config_with_drive', 
    'sync_all_configs',
    
    # Lingkungan
    'get_environment_manager', 'EnvironmentManager',
    
    # Konfigurasi Layer
    'get_layer_config', 'LayerConfigManager',
    
    # Logging
    'get_logger', 'SmartCashLogger', 'LogLevel',
    
    # Utilitas
    'is_colab', 'is_notebook', 
    'get_system_info', 'generate_unique_id', 
    'format_time', 'get_timestamp', 
    
    # Tipe Data dan Antarmuka
    'ImageType', 'PathType', 'TensorType', 'ConfigType', 
    'ProgressCallback', 'LogCallback',
    'BoundingBox', 'Detection', 'ModelInfo', 'DatasetStats',
    
    # Exceptions
    'SmartCashError', 'ConfigError', 'DatasetError', 
    'ModelError', 'DetectionError', 
    'FileError', 'APIError', 'ValidationError',
    
    # Konstanta
    'VERSION', 'APP_NAME', 
    'DEFAULT_CONFIG_DIR', 'DEFAULT_DATA_DIR', 
    'DEFAULT_MODEL_DIR', 'DEFAULT_LOGS_DIR',
    
    # Threading dan Multiprocessing
    'get_optimal_thread_count', 
    'process_in_parallel', 
    'process_with_stats',
    
    # Inisialisasi Proyek
    'initialize_config'
]