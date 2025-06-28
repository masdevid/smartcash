# File: smartcash/ui/pretrained/services/model_checker.py
"""
File: smartcash/ui/pretrained/services/model_checker.py
Deskripsi: Service untuk mengecek keberadaan pretrained models di direktori
"""

import os
from typing import Dict, Any, Tuple, Optional
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def check_model_exists(models_dir: str, model_type: str = 'yolov5s') -> bool:
    """🔍 Check apakah model pretrained ada di direktori
    
    Args:
        models_dir: Path direktori model
        model_type: Tipe model (default: yolov5s)
        
    Returns:
        True jika model ada, False jika tidak
    """
    try:
        if not models_dir or not os.path.exists(models_dir):
            logger.debug(f"📁 Directory tidak ditemukan: {models_dir}")
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
                logger.debug(f"✅ Model found: {model_path}")
                return True
        
        logger.debug(f"❌ No model files found in {models_dir}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error checking model existence: {str(e)}")
        return False


def get_model_info(models_dir: str, model_type: str = 'yolov5s') -> Dict[str, Any]:
    """📊 Get detailed information tentang model di direktori
    
    Args:
        models_dir: Path direktori model
        model_type: Tipe model (default: yolov5s)
        
    Returns:
        Dictionary berisi informasi model
    """
    try:
        info = {
            'exists': False,
            'files': [],
            'total_size': 0,
            'directory': models_dir,
            'model_type': model_type
        }
        
        if not os.path.exists(models_dir):
            return info
        
        # Scan files yang berhubungan dengan model
        model_patterns = [f'{model_type}.pt', f'{model_type}.yaml', f'{model_type}_best.pt', f'{model_type}_last.pt']
        
        for pattern in model_patterns:
            model_path = os.path.join(models_dir, pattern)
            if os.path.isfile(model_path):
                file_size = os.path.getsize(model_path)
                info['files'].append({
                    'name': pattern,
                    'path': model_path,
                    'size': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2)
                })
                info['total_size'] += file_size
        
        info['exists'] = len(info['files']) > 0
        info['total_size_mb'] = round(info['total_size'] / (1024 * 1024), 2)
        
        return info
        
    except Exception as e:
        logger.error(f"❌ Error getting model info: {str(e)}")
        return {
            'exists': False,
            'files': [],
            'total_size': 0,
            'directory': models_dir,
            'model_type': model_type,
            'error': str(e)
        }


def check_both_locations(local_dir: str, drive_dir: str, model_type: str = 'yolov5s') -> Dict[str, Any]:
    """🔍 Check model existence di kedua lokasi (local dan drive)
    
    Args:
        local_dir: Path direktori local
        drive_dir: Path direktori drive
        model_type: Tipe model (default: yolov5s)
        
    Returns:
        Dictionary berisi status kedua lokasi
    """
    try:
        local_info = get_model_info(local_dir, model_type)
        drive_info = get_model_info(drive_dir, model_type)
        
        return {
            'local': local_info,
            'drive': drive_info,
            'sync_needed': local_info['exists'] != drive_info['exists'],
            'recommendation': _get_recommendation(local_info, drive_info)
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking both locations: {str(e)}")
        return {
            'local': {'exists': False, 'error': str(e)},
            'drive': {'exists': False, 'error': str(e)},
            'sync_needed': False,
            'recommendation': 'download_required'
        }


def _get_recommendation(local_info: Dict[str, Any], drive_info: Dict[str, Any]) -> str:
    """💡 Get recommendation berdasarkan status model
    
    Args:
        local_info: Info model di local
        drive_info: Info model di drive
        
    Returns:
        String recommendation
    """
    if local_info['exists'] and drive_info['exists']:
        return 'models_available'
    elif local_info['exists'] and not drive_info['exists']:
        return 'sync_to_drive'
    elif not local_info['exists'] and drive_info['exists']:
        return 'sync_from_drive'
    else:
        return 'download_required'


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


def create_models_directory(models_dir: str) -> bool:
    """📁 Create direktori models jika belum ada
    
    Args:
        models_dir: Path direktori yang akan dibuat
        
    Returns:
        True jika berhasil dibuat atau sudah ada, False jika gagal
    """
    try:
        if os.path.exists(models_dir):
            logger.debug(f"📁 Directory sudah ada: {models_dir}")
            return True
        
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"📁 Directory berhasil dibuat: {models_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating directory {models_dir}: {str(e)}")
        return False