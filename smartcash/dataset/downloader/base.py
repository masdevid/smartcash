"""
File: smartcash/dataset/downloader/base.py
Deskripsi: Fixed base components dengan proper imports dan directory creation utilities
"""

import time
import requests
import shutil
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from abc import ABC, abstractmethod
from smartcash.common.logger import get_logger


class BaseDownloaderComponent(ABC):
    """Base class untuk semua downloader components dengan fixed imports"""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger(self.__class__.__module__)
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback"""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress dengan safe execution"""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def _create_success_result(self, **kwargs) -> Dict[str, Any]:
        """Create standardized success result"""
        return {'status': 'success', **kwargs}
    
    def _create_error_result(self, message: str, **kwargs) -> Dict[str, Any]:
        """Create standardized error result"""
        return {'status': 'error', 'message': message, **kwargs}


class DirectoryManager:
    """Enhanced directory manager dengan safe creation"""
    
    @staticmethod
    def ensure_dataset_structure(base_path: Path) -> Dict[str, Any]:
        """Ensure complete dataset structure exists"""
        try:
            required_dirs = [
                base_path / 'downloads',
                base_path / 'train' / 'images',
                base_path / 'train' / 'labels',
                base_path / 'valid' / 'images',
                base_path / 'valid' / 'labels',
                base_path / 'test' / 'images',
                base_path / 'test' / 'labels'
            ]
            
            created_dirs = []
            for dir_path in required_dirs:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dir_path))
            
            return {
                'status': 'success',
                'created_directories': created_dirs,
                'total_created': len(created_dirs)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Failed to create directories: {str(e)}"
            }
    
    @staticmethod
    def ensure_directory(path: Path) -> bool:
        """Safe directory creation dengan error handling"""
        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_dataset_paths(base_path: Path) -> Dict[str, Any]:
        """Validate dan create missing dataset paths"""
        validation_result = {
            'valid': True,
            'missing_dirs': [],
            'created_dirs': [],
            'errors': []
        }
        
        required_structure = {
            'downloads': base_path / 'downloads',
            'train_images': base_path / 'train' / 'images',
            'train_labels': base_path / 'train' / 'labels',
            'valid_images': base_path / 'valid' / 'images',
            'valid_labels': base_path / 'valid' / 'labels',
            'test_images': base_path / 'test' / 'images',
            'test_labels': base_path / 'test' / 'labels'
        }
        
        for name, dir_path in required_structure.items():
            if not dir_path.exists():
                validation_result['missing_dirs'].append(name)
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    validation_result['created_dirs'].append(name)
                except Exception as e:
                    validation_result['errors'].append(f"Failed to create {name}: {str(e)}")
                    validation_result['valid'] = False
        
        return validation_result


class ProgressTracker:
    """Fixed progress tracker dengan better error handling"""
    
    def __init__(self):
        self._callback = None
        self.current_step = ""
        self.overall_progress = 0
        self.progress_bar = None  # Add progress_bar attribute for UI compatibility
    
    def set_callback(self, callback: Callable) -> None:
        self._callback = callback
    
    def update(self, step: str, progress: int, message: str = "") -> None:
        if self._callback:
            try:
                self._callback(step, progress, 100, message)
            except Exception:
                pass  # Silent fail untuk prevent blocking
        self.current_step = step
        self.overall_progress = progress


class RequestsHelper:
    """Fixed requests helper dengan better timeout handling"""
    
    @staticmethod
    def get_with_retry(url: str, params: Dict = None, timeout: int = 30, 
                      retry_count: int = 3) -> requests.Response:
        """GET request dengan retry logic"""
        for attempt in range(1, retry_count + 1):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == retry_count:
                    raise e
                time.sleep(attempt * 2)
    
    @staticmethod
    def download_with_progress(url: str, output_path: Path, 
                             progress_callback: Callable = None,
                             chunk_size: int = 65536) -> Dict[str, Any]:  # 64KB standardized
        """Download file dengan progress tracking"""
        try:
            # Ensure parent directory exists
            DirectoryManager.ensure_directory(output_path.parent)
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            progress_callback("download", progress, 100, 
                                           f"📥 {downloaded/1048576:.1f}/{total_size/1048576:.1f} MB")
            
            return {
                'status': 'success',
                'file_path': str(output_path),
                'size_bytes': downloaded,
                'size_mb': downloaded / 1048576
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Download failed: {str(e)}"}


class ValidationHelper:
    """Enhanced validation helper dengan directory checking"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
        """Validate config dengan required fields"""
        missing = [f for f in required_fields if not config.get(f, '').strip()]
        
        if missing:
            return {'valid': False, 'errors': [f"Field '{f}' wajib diisi" for f in missing]}
        
        # API key validation
        api_key = config.get('api_key', '')
        if api_key and len(api_key) < 10:
            return {'valid': False, 'errors': ['API key terlalu pendek']}
        
        return {'valid': True, 'errors': []}
    
    @staticmethod
    def validate_dataset_structure(dataset_dir: Path) -> Dict[str, Any]:
        """Enhanced dataset structure validation dengan auto-creation"""
        if not dataset_dir.exists():
            # Try to create base directory
            try:
                DirectoryManager.ensure_directory(dataset_dir)
            except Exception:
                return {'valid': False, 'message': 'Dataset directory tidak dapat dibuat'}
        
        # Ensure directory structure exists
        structure_result = DirectoryManager.ensure_dataset_structure(dataset_dir)
        
        if structure_result['status'] == 'error':
            return {'valid': False, 'message': structure_result['message']}
        
        # Check for split structure
        splits = []
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split
            images_dir = split_dir / 'images'
            if images_dir.exists():
                splits.append(split)
        
        # Check for flat structure
        has_flat = (dataset_dir / 'images').exists()
        
        return {
            'valid': True, 
            'splits': splits, 
            'has_flat': has_flat,
            'created_dirs': structure_result.get('created_directories', [])
        }


class FileHelper:
    """Enhanced file helper dengan better error handling"""
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure directory exists dengan error handling"""
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to create directory {path}: {str(e)}")
    
    @staticmethod
    def get_file_stats(file_path: Path) -> Dict[str, Any]:
        """Get file statistics"""
        try:
            if not file_path.exists():
                return {'exists': False, 'size_bytes': 0, 'size_mb': 0}
                
            stat = file_path.stat()
            return {
                'exists': True,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / 1048576,
                'modified': stat.st_mtime
            }
        except Exception:
            return {'exists': False, 'size_bytes': 0, 'size_mb': 0}
    
    @staticmethod
    def backup_directory(source_dir: Path) -> Dict[str, Any]:
        """Backup directory dengan timestamp"""
        try:
            from datetime import datetime
            
            if not source_dir.exists():
                return {'success': False, 'message': 'Source directory not found'}
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = source_dir.parent / f"{source_dir.name}_backup_{timestamp}"
            
            shutil.copytree(source_dir, backup_dir)
            return {'success': True, 'backup_path': str(backup_dir)}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    @staticmethod
    def cleanup_temp(temp_dir: Path) -> None:
        """Cleanup temporary directory"""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


class PathHelper:
    """Enhanced path helper dengan environment awareness"""
    
    @staticmethod
    def setup_download_paths(workspace: str, project: str, version: str) -> Dict[str, Path]:
        """Setup download paths berdasarkan environment"""
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.constants.paths import get_paths_for_environment
        
        env_manager = get_environment_manager()
        env_paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
        
        dataset_name = f"{workspace}_{project}_v{version}"
        base_downloads = Path(env_paths['downloads'])
        
        paths = {
            'temp_dir': base_downloads / f"{dataset_name}_temp",
            'final_dir': Path(env_paths['data_root']),
            'temp_zip': base_downloads / f"{dataset_name}_temp" / 'dataset.zip',
            'extract_dir': base_downloads / f"{dataset_name}_temp" / 'extracted'
        }
        
        # Create directories dengan error handling
        for path_key, path in paths.items():
            if path_key in ['temp_dir', 'final_dir']:
                try:
                    FileHelper.ensure_directory(path)
                    
                    # Ensure dataset structure for final_dir
                    if path_key == 'final_dir':
                        DirectoryManager.ensure_dataset_structure(path)
                        
                except Exception as e:
                    raise Exception(f"Failed to setup {path_key}: {str(e)}")
        
        return paths


# Factory functions with enhanced error handling
def create_progress_tracker() -> ProgressTracker:
    """Create progress tracker"""
    return ProgressTracker()

def create_validation_helper() -> ValidationHelper:
    """Create validation helper"""
    return ValidationHelper()

def create_file_helper() -> FileHelper:
    """Create file helper"""
    return FileHelper()

def create_path_helper() -> PathHelper:
    """Create path helper"""
    return PathHelper()

def create_directory_manager() -> DirectoryManager:
    """Create directory manager"""
    return DirectoryManager()


# Export
__all__ = [
    'BaseDownloaderComponent', 'DirectoryManager', 'ProgressTracker', 'RequestsHelper', 
    'ValidationHelper', 'FileHelper', 'PathHelper',
    'create_progress_tracker', 'create_validation_helper',
    'create_file_helper', 'create_path_helper', 'create_directory_manager'
]