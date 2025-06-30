# File: smartcash/ui/pretrained/services/model_checker.py
"""
File: smartcash/ui/pretrained/services/model_checker.py
Deskripsi: Service untuk mengecek keberadaan pretrained models di direktori
"""

import os
from typing import Dict, Any, Tuple, Optional, Callable
from functools import wraps

from smartcash.ui.utils.error_utils import (
    with_error_handling,
    log_errors,
    create_error_context
)

# Type alias for logger bridge type
LoggerBridge = Callable[[str, str], None]

@with_error_handling(
    component="pretrained",
    operation="check_model_exists",
    fallback_value=False
)
@log_errors(level="debug")
def check_model_exists(
    models_dir: str, 
    model_type: str = 'yolov5s',
    logger_bridge: Optional[LoggerBridge] = None,
    **kwargs
) -> bool:
    """🔍 Check apakah model pretrained ada di direktori
    
    Args:
        models_dir: Path direktori model
        model_type: Tipe model (default: yolov5s)
        logger_bridge: Logger bridge untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        True jika model ada, False jika tidak
    """
    context = create_error_context(
        component="pretrained",
        operation="check_model_exists",
        details={
            "models_dir": models_dir,
            "model_type": model_type
        }
    )
    
    if not models_dir or not os.path.exists(models_dir):
        msg = f"Directory tidak ditemukan: {models_dir}"
        if logger_bridge:
            logger_bridge(msg, "debug")
        return False
    
    # Common model file patterns untuk YOLOv5
    model_patterns = [
        f'{model_type}.pt',
        f'{model_type}.yaml',
        f'{model_type}_best.pt',
        f'{model_type}_last.pt'
    ]
    
    # Check apakah minimal ada satu file model
    for pattern in model_patterns:
        model_path = os.path.join(models_dir, pattern)
        if os.path.isfile(model_path):
            if logger_bridge:
                logger_bridge(f"✅ Model found: {model_path}", "debug")
            return True
    
    if logger_bridge:
        logger_bridge(f"❌ No model files found in {models_dir}", "debug")
    return False


@with_error_handling(
    component="pretrained",
    operation="get_model_info",
    fallback_factory=lambda models_dir, model_type, **_: {
        'exists': False,
        'files': [],
        'total_size': 0,
        'directory': models_dir,
        'model_type': model_type,
        'error': True
    }
)
@log_errors(level="error")
def get_model_info(
    models_dir: str, 
    model_type: str = 'yolov5s',
    logger_bridge: Optional[LoggerBridge] = None,
    **kwargs
) -> Dict[str, Any]:
    """📊 Get detailed information tentang model di direktori
    
    Args:
        models_dir: Path direktori model
        model_type: Tipe model (default: yolov5s)
        logger_bridge: Logger bridge untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        Dictionary berisi informasi model
    """
    context = create_error_context(
        component="pretrained",
        operation="get_model_info",
        details={
            "models_dir": models_dir,
            "model_type": model_type
        }
    )
    
    info = {
        'exists': False,
        'files': [],
        'total_size': 0,
        'directory': models_dir,
        'model_type': model_type
    }
    
    if not os.path.exists(models_dir):
        if logger_bridge:
            logger_bridge(f"Directory tidak ditemukan: {models_dir}", "warning")
        return info
    
    # Scan files yang berhubungan dengan model
    model_patterns = [
        f'{model_type}.pt', 
        f'{model_type}.yaml', 
        f'{model_type}_best.pt', 
        f'{model_type}_last.pt'
    ]
    
    for pattern in model_patterns:
        model_path = os.path.join(models_dir, pattern)
        if os.path.isfile(model_path):
            try:
                file_size = os.path.getsize(model_path)
                info['files'].append({
                    'name': pattern,
                    'path': model_path,
                    'size': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
                info['total_size'] += file_size
                if logger_bridge:
                    logger_bridge(f"✅ Found model file: {pattern} ({round(file_size/(1024*1024), 2)} MB)", "debug")
            except Exception as e:
                if logger_bridge:
                    logger_bridge(f"⚠️ Error processing {pattern}: {str(e)}", "warning")
    
    info['exists'] = len(info['files']) > 0
    info['total_size_mb'] = round(info['total_size'] / (1024 * 1024), 2)
    
    if logger_bridge and info['exists']:
        logger_bridge(f"ℹ️ Found {len(info['files'])} model files ({info['total_size_mb']} MB total)", "info")
    
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
    logger_bridge: Optional[LoggerBridge] = None,
    **kwargs
) -> Dict[str, Any]:
    """🔍 Check model existence di kedua lokasi (local dan drive)
    
    Args:
        local_dir: Path direktori local
        drive_dir: Path direktori drive
        model_type: Tipe model (default: yolov5s)
        logger_bridge: Logger bridge untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        Dictionary berisi status kedua lokasi
    """
    context = create_error_context(
        component="pretrained",
        operation="check_both_locations",
        details={
            "local_dir": local_dir,
            "drive_dir": drive_dir,
            "model_type": model_type
        }
    )
    
    if logger_bridge:
        logger_bridge(f"🔍 Checking model in both locations: local={local_dir}, drive={drive_dir}", "info")
    
    local_info = get_model_info(local_dir, model_type, logger_bridge=logger_bridge)
    drive_info = get_model_info(drive_dir, model_type, logger_bridge=logger_bridge)
    
    recommendation = _get_recommendation(local_info, drive_info)
    
    if logger_bridge:
        logger_bridge(f"✅ Checked both locations. Recommendation: {recommendation}", "info")
    
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
        Tuple (is_valid, error_message)
    """
    try:
        if not os.path.isfile(model_path):
            return False, "File tidak ditemukan"
        
        # Check file size (model YOLOv5s should be > 1MB)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # < 1MB
            return False, f"File terlalu kecil ({file_size} bytes)"
        
        # Check file extension
        if not model_path.endswith(('.pt', '.yaml')):
            return False, "Format file tidak didukung"
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"


@with_error_handling(
    component="pretrained",
    operation="create_models_directory",
    fallback_value=False
)
@log_errors(level="error")
def create_models_directory(
    models_dir: str,
    logger_bridge: Optional[LoggerBridge] = None,
    **kwargs
) -> bool:
    """📁 Create direktori models jika belum ada
    
    Args:
        models_dir: Path direktori yang akan dibuat
        logger_bridge: Logger bridge untuk logging ke UI
        **kwargs: Additional arguments for error context
        
    Returns:
        True jika berhasil dibuat atau sudah ada, False jika gagal
    """
    context = create_error_context(
        component="pretrained",
        operation="create_models_directory",
        details={
            "models_dir": models_dir
        }
    )
    
    if os.path.exists(models_dir):
        if logger_bridge:
            logger_bridge(f"📁 Directory sudah ada: {models_dir}", "debug")
        return True
        
    os.makedirs(models_dir, exist_ok=True)
    
    if logger_bridge:
        logger_bridge(f"✅ Directory berhasil dibuat: {models_dir}", "info")
    
    return True