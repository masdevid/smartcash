"""
File: smartcash/ui/pretrained/utils/model_utils.py
Deskripsi: Utility functions untuk pretrained model operations
"""

import os
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def validate_model_file(file_path: str, min_size_mb: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate model file existence dan integrity.
    
    Args:
        file_path: Path to model file
        min_size_mb: Minimum file size in MB
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    try:
        if not os.path.exists(file_path):
            return False, f"File tidak ditemukan: {file_path}"
        
        # Check file size
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if min_size_mb and file_size_mb < min_size_mb:
            return False, f"File terlalu kecil: {file_size_mb:.1f}MB < {min_size_mb}MB"
        
        # Check jika file bisa dibaca
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB
        except Exception as e:
            return False, f"File tidak bisa dibaca: {str(e)}"
        
        return True, f"Valid ({file_size_mb:.1f}MB)"
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"

def get_model_info(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive model file information.
    
    Args:
        file_path: Path to model file
        
    Returns:
        Dict berisi model information
    """
    try:
        if not os.path.exists(file_path):
            return {
                'exists': False,
                'error': 'File not found'
            }
        
        stat = os.stat(file_path)
        file_size_mb = stat.st_size / (1024 * 1024)
        
        # Get file hash untuk integrity check
        file_hash = calculate_file_hash(file_path)
        
        return {
            'exists': True,
            'filename': os.path.basename(file_path),
            'size_bytes': stat.st_size,
            'size_mb': round(file_size_mb, 2),
            'modified_time': stat.st_mtime,
            'file_hash': file_hash,
            'extension': Path(file_path).suffix,
            'readable': check_file_readable(file_path)
        }
        
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }

def calculate_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
    """
    Calculate file hash untuk integrity verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256')
        
    Returns:
        File hash string atau None jika error
    """
    try:
        hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"❌ Error calculating hash: {str(e)}")
        return None

def check_file_readable(file_path: str) -> bool:
    """Check jika file bisa dibaca"""
    try:
        with open(file_path, 'rb') as f:
            f.read(1024)
        return True
    except Exception:
        return False

def cleanup_partial_downloads(directory: str) -> List[str]:
    """
    Cleanup partial atau corrupted download files.
    
    Args:
        directory: Directory to scan
        
    Returns:
        List of cleaned up files
    """
    cleaned_files = []
    
    try:
        if not os.path.exists(directory):
            return cleaned_files
        
        # Look for temporary atau partial files
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check for common partial download patterns
            should_cleanup = (
                filename.endswith('.tmp') or
                filename.endswith('.part') or
                filename.endswith('.download') or
                (filename.endswith(('.pt', '.bin', '.pth')) and os.path.getsize(file_path) < 1024)  # < 1KB
            )
            
            if should_cleanup:
                try:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                    logger.info(f"🗑️ Cleaned up: {filename}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cleanup {filename}: {str(e)}")
        
        return cleaned_files
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {str(e)}")
        return cleaned_files

def get_available_models(models_dir: str, models_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get status of all configured models.
    
    Args:
        models_dir: Models directory path
        models_config: Models configuration
        
    Returns:
        Dict berisi status semua models
    """
    results = {}
    
    try:
        for model_key, model_info in models_config.items():
            model_path = os.path.join(models_dir, model_info['filename'])
            
            # Get model info
            info = get_model_info(model_path)
            
            # Add config info
            info.update({
                'config': model_info,
                'model_key': model_key,
                'path': model_path
            })
            
            # Validate dengan min_size
            if info['exists']:
                is_valid, message = validate_model_file(
                    model_path, 
                    model_info.get('min_size_mb')
                )
                info['valid'] = is_valid
                info['validation_message'] = message
            else:
                info['valid'] = False
                info['validation_message'] = 'File not found'
            
            results[model_key] = info
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error getting available models: {str(e)}")
        return {}

def create_model_directories(models_dir: str, drive_dir: Optional[str] = None) -> bool:
    """
    Create necessary directories untuk model storage.
    
    Args:
        models_dir: Main models directory
        drive_dir: Optional drive directory
        
    Returns:
        bool: Success status
    """
    try:
        # Create main models directory
        os.makedirs(models_dir, exist_ok=True)
        logger.info(f"📁 Created models directory: {models_dir}")
        
        # Create drive directory jika specified dan drive mounted
        if drive_dir and os.path.exists('/content/drive'):
            os.makedirs(drive_dir, exist_ok=True)
            logger.info(f"📁 Created drive directory: {drive_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating directories: {str(e)}")
        return False

def format_file_size(size_bytes: int) -> str:
    """Format file size dalam human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_download_progress_message(downloaded: int, total: int, filename: str) -> str:
    """Generate progress message untuk download"""
    if total > 0:
        progress_pct = (downloaded / total) * 100
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        return f"📥 {filename}: {progress_pct:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
    else:
        downloaded_mb = downloaded / (1024 * 1024)
        return f"📥 {filename}: {downloaded_mb:.1f} MB downloaded"