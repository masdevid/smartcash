"""
File: smartcash/ui/pretrained/services/model_checker.py
Deskripsi: Service untuk mengecek keberadaan pretrained models di direktori
"""
import os
from typing import Dict, Any, Tuple, Optional, Callable, List
from functools import wraps

from smartcash.ui.utils.ui_logger import UILogger, get_module_logger

from smartcash.ui.pretrained.utils import (
    with_error_handling,
    log_errors
)

# Type aliases
ModelInfo = Dict[str, Any]

# Constants
DEFAULT_MODEL_TYPE = 'yolov5s'
MODEL_EXTENSIONS = ('.pt', '.yaml')
MIN_MODEL_SIZE = 1024 * 1024  # 1MB
MODEL_FILE_PATTERNS = [
    '{model_type}.pt',
    '{model_type}.yaml',
    '{model_type}_best.pt',
    '{model_type}_last.pt'
]

def _get_model_patterns(model_type: str) -> List[str]:
    """Get list of model file patterns for the given model type"""
    return [p.format(model_type=model_type) for p in MODEL_FILE_PATTERNS]

def _create_model_info(
    models_dir: str,
    model_type: str,
    exists: bool = False,
    files: Optional[List[Dict]] = None,
    error: Optional[str] = None
) -> ModelInfo:
    """Create a standardized model info dictionary"""
    return {
        'exists': exists,
        'files': files or [],
        'total_size': 0,
        'directory': models_dir,
        'model_type': model_type,
        **({'error': error} if error else {})
    }

@with_error_handling(
    component="pretrained",
    operation="check_model_exists",
    fallback_value=False
)
@log_errors(level="debug")
def check_model_exists(
    models_dir: str, 
    model_type: str = DEFAULT_MODEL_TYPE,
    logger: Optional[UILogger] = None,
    **kwargs
) -> bool:
    """🔍 Check apakah model pretrained ada di direktori
    
    Args:
        models_dir: Path direktori model
        model_type: Tipe model (default: yolov5s)
        logger: UILogger instance untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        bool: True jika model ada, False jika tidak
    """
    logger = logger or get_module_logger('smartcash.ui.pretrained.services.model_checker')
    
    if not models_dir or not os.path.exists(models_dir):
        logger.debug(f"Directory tidak ditemukan: {models_dir}")
        return False
    
    # Check for any model file
    for pattern in _get_model_patterns(model_type):
        if os.path.isfile(os.path.join(models_dir, pattern)):
            logger.debug(f"✅ Model found: {pattern}")
            return True
    
    logger.debug(f"❌ No model files found in {models_dir}")
    return False


def _process_model_file(model_path: str, pattern: str, logger: Optional[UILogger] = None) -> Optional[Dict]:
    """Process a single model file and return its info"""
    logger = logger or get_module_logger('smartcash.ui.pretrained.services.model_checker')
    try:
        file_size = os.path.getsize(model_path)
        if file_size < MIN_MODEL_SIZE:
            logger.warning(f"File terlalu kecil (hanya {file_size} bytes): {pattern}")
            return None
            
        return {
            'name': os.path.basename(model_path),
            'path': model_path,
            'size': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        logger.error(f"Error processing {pattern}: {str(e)}", exc_info=True)
        return None

def _log_message(message: str, logger: Optional[UILogger] = None, level: str = "info") -> None:
    """Log a message using the UILogger if available"""
    logger = logger or get_module_logger('smartcash.ui.pretrained.services.model_checker')
    
    if hasattr(logger, level):
        getattr(logger, level)(message)
    else:
        # Fallback to standard logging levels
        if level == "success":
            logger.info(f"✅ {message}")
        else:
            logger.info(message)

@with_error_handling(
    component="model_checker",
    operation="get_model_info",
    fallback_factory=lambda models_dir, model_type, **_: _create_model_info(
        models_dir, model_type, error="Error getting model info"
    )
)
@log_errors(level="error")
def get_model_info(
    models_dir: str, 
    model_type: str = DEFAULT_MODEL_TYPE,
    logger: Optional[UILogger] = None,
    **kwargs
) -> ModelInfo:
    """📊 Get detailed information tentang model di direktori
    
    Args:
        models_dir: Path direktori model
        model_type: Tipe model (default: yolov5s)
        logger: UILogger instance untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        Dictionary berisi informasi model
    """
    logger = logger or get_module_logger('smartcash.ui.pretrained.services.model_checker')
    
    info = _create_model_info(models_dir, model_type)
    
    if not os.path.exists(models_dir):
        logger.warning(f"Directory tidak ditemukan: {models_dir}")
        return info
    
    # Process all model files
    for pattern in _get_model_patterns(model_type):
        model_path = os.path.join(models_dir, pattern)
        if os.path.isfile(model_path):
            if file_info := _process_model_file(model_path, pattern, logger):
                info['files'].append(file_info)
                info['total_size'] += file_info['size']
    
    info['exists'] = len(info['files']) > 0
    info['total_size_mb'] = round(info['total_size'] / (1024 * 1024), 2)
    
    if info['exists']:
        logger.info(f"ℹ️ Found {len(info['files'])} model files ({info['total_size_mb']} MB total)")
    
    return info


@with_error_handling(
    component="pretrained",
    operation="check_both_locations",
    fallback_factory=lambda local_dir, drive_dir, model_type, **_: {
        'local': {'exists': False, 'error': 'Error checking locations'},
        'drive': {'exists': False, 'error': 'Error checking locations'},
        'recommendation': 'Error checking model locations',
        'error': True
    }
)
@log_errors(level="error")
def check_both_locations(
    local_dir: str, 
    drive_dir: str, 
    model_type: str = 'yolov5s',
    logger: Optional[UILogger] = None,
    **kwargs
) -> Dict[str, Any]:
    """🔍 Check model existence di kedua lokasi (local dan drive)
    
    Args:
        local_dir: Path direktori local
        drive_dir: Path direktori drive
        model_type: Tipe model (default: yolov5s)
        logger: UILogger instance untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        Dictionary berisi status kedua lokasi
    """
    logger = logger or get_module_logger('smartcash.ui.pretrained.services.model_checker')
    logger.info(f"🔍 Checking model in both locations: local={local_dir}, drive={drive_dir}")
    
    local_info = get_model_info(local_dir, model_type, logger=logger)
    drive_info = get_model_info(drive_dir, model_type, logger=logger)
    
    recommendation = _get_recommendation(local_info, drive_info)
    
    logger.info(f"✅ Checked both locations. Recommendation: {recommendation}")
    
    return {
        'local': local_info,
        'drive': drive_info,
        'recommendation': recommendation
    }


def _get_recommendation(local_info: Dict[str, Any], drive_info: Dict[str, Any]) -> str:
    """💡 Get recommendation berdasarkan status model
    
    Args:
        local_info: Info model di local
        drive_info: Info model di drive
        
    Returns:
        String recommendation
    """
    local_exists = local_info.get('exists', False)
    drive_exists = drive_info.get('exists', False)
    
    if local_exists and drive_exists:
        local_size = local_info.get('total_size_mb', 0)
        drive_size = drive_info.get('total_size_mb', 0)
        if local_size > drive_size:
            return f"✅ Model tersedia di kedua lokasi (local lebih baru: {local_size}MB vs {drive_size}MB)"
        elif drive_size > local_size:
            return f"✅ Model tersedia di kedua lokasi (drive lebih baru: {drive_size}MB vs {local_size}MB)"
        return "✅ Model tersedia di kedua lokasi (ukuran sama)"
    elif local_exists:
        return "💡 Model hanya ada di local"
    elif drive_exists:
        return "💡 Model hanya ada di drive"
    else:
        return "❌ Model tidak ditemukan di kedua lokasi"


def validate_model_file(model_path: str) -> Tuple[bool, Optional[str]]:
    """✅ Validate apakah file model valid
    
    Args:
        model_path: Path ke file model
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        if not os.path.isfile(model_path):
            return False, "File tidak ditemukan"
        
        # Check file size
        if (file_size := os.path.getsize(model_path)) < MIN_MODEL_SIZE:
            return False, f"File terlalu kecil ({file_size} bytes)"
        
        # Check file extension
        if not model_path.endswith(MODEL_EXTENSIONS):
            return False, f"Format file tidak didukung. Harus salah satu dari: {', '.join(MODEL_EXTENSIONS)}"
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"


@with_error_handling(
    component="model_checker",
    operation="create_models_directory",
    fallback_value=False
)
@log_errors(level="error")
def create_models_directory(
    models_dir: str,
    logger: Optional[UILogger] = None,
    **kwargs
) -> bool:
    """📁 Create direktori models jika belum ada
    
    Args:
        models_dir: Path direktori yang akan dibuat
        logger: Logger untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        True jika berhasil dibuat atau sudah ada, False jika gagal
    """
    logger = logger or get_module_logger('smartcash.ui.pretrained.services.model_checker')
    
    if os.path.exists(models_dir):
        logger.debug(f"📁 Directory sudah ada: {models_dir}")
        return True
        
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"✅ Directory berhasil dibuat: {models_dir}")
    return True