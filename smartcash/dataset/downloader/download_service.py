"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: Complete download service dengan proper integration dan error handling
"""

import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.downloader.roboflow_client import RoboflowClient
from smartcash.dataset.downloader.file_processor import FileProcessor
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker
from smartcash.dataset.downloader.validators import DatasetValidator

class DownloadService:
    """Main download service dengan comprehensive features."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger('downloader.service')
        self.env_manager = get_environment_manager()
        
        # Extract API key dari config
        api_key = config.get('api_key', '')
        
        # Initialize components dengan proper parameters
        self.roboflow_client = RoboflowClient(api_key, logger=logger)
        self.file_processor = FileProcessor(logger)
        self.validator = DatasetValidator(logger)
        self.progress_tracker = DownloadProgressTracker()
        
        # Progress callback
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback untuk UI updates."""
        self._progress_callback = callback
        
        # Propagate ke components
        self.roboflow_client.set_progress_callback(callback)
        self.file_processor.set_progress_callback(callback)
        self.progress_tracker.set_callback(callback)
    
    def download_dataset(self, workspace: str, project: str, version: str, api_key: str,
                        output_format: str = 'yolov5pytorch', validate_download: bool = True,
                        organize_dataset: bool = True, backup_existing: bool = False) -> Dict[str, Any]:
        """Download dataset dengan comprehensive flow."""
        start_time = time.time()
        
        try:
            # Step 1: Validate parameters
            self._notify_progress('validate', 0, 100, "Validating parameters...")
            validation_result = self._validate_parameters(workspace, project, version, api_key, output_format)
            
            if not validation_result['valid']:
                return self._error_result(f"Validation failed: {'; '.join(validation_result['errors'])}")
            
            self._notify_progress('validate', 100, 100, "Parameters validated")
            
            # Step 2: Get dataset metadata
            self._notify_progress('connect', 0, 100, "Connecting to Roboflow...")
            metadata_result = self.roboflow_client.get_dataset_metadata(workspace, project, version, output_format)
            
            if metadata_result['status'] != 'success':
                return self._error_result(f"Failed to get dataset metadata: {metadata_result['message']}")
            
            metadata = metadata_result['data']
            download_url = metadata_result['download_url']
            
            self._notify_progress('connect', 100, 100, "Connected to Roboflow")
            
            # Step 3: Setup paths
            paths = self._setup_download_paths(workspace, project, version)
            
            # Step 4: Backup existing jika diminta
            if backup_existing:
                self._handle_backup_existing(paths)
            
            # Step 5: Download dataset
            self._notify_progress('download', 0, 100, "Starting download...")
            download_result = self.roboflow_client.download_dataset(download_url, paths['temp_dir'] / 'dataset.zip')
            
            if download_result['status'] != 'success':
                return self._error_result(f"Download failed: {download_result['message']}")
            
            # Step 6: Extract dan process files
            self._notify_progress('extract', 0, 100, "Extracting dataset...")
            extract_result = self.file_processor.extract_zip(
                Path(download_result['file_path']), paths['temp_dir'] / 'extracted'
            )
            
            if extract_result['status'] != 'success':
                return self._error_result(f"Extraction failed: {extract_result['message']}")
            
            # Step 7: Organize dataset structure jika diminta
            if organize_dataset:
                self._notify_progress('organize', 0, 100, "Organizing dataset structure...")
                organize_result = self.file_processor.organize_dataset(
                    paths['temp_dir'] / 'extracted', paths['final_dir']
                )
                
                if organize_result['status'] != 'success':
                    return self._error_result(f"Organization failed: {organize_result['message']}")
                
                final_stats = {
                    'total_images': organize_result['total_images'],
                    'total_labels': organize_result['total_labels'],
                    'splits': organize_result['splits']
                }
            else:
                # Move ke final directory tanpa organize
                import shutil
                if paths['final_dir'].exists():
                    shutil.rmtree(paths['final_dir'])
                shutil.move(str(paths['temp_dir'] / 'extracted'), str(paths['final_dir']))
                final_stats = {'total_images': 0, 'total_labels': 0, 'splits': {}}
            
            # Step 8: Validate hasil jika diminta
            if validate_download:
                self._notify_progress('organize', 80, 100, "Validating download results...")
                validate_result = self.validator.validate_extracted_dataset(paths['final_dir'])
                
                if not validate_result['valid']:
                    self.logger.warning(f"⚠️ Validation warnings: {', '.join(validate_result.get('issues', []))}")
                    if validate_result.get('issues'):
                        final_stats.update(validate_result)
            
            # Cleanup temp files
            self._cleanup_temp_files(paths['temp_dir'])
            
            # Success result
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'output_dir': str(paths['final_dir']),
                'stats': final_stats,
                'duration': duration,
                'drive_storage': self.env_manager.is_drive_mounted,
                'metadata': {
                    'workspace': workspace,
                    'project': project,
                    'version': version,
                    'format': output_format
                }
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Download service error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            
            return {
                'status': 'error',
                'message': error_msg,
                'duration': duration
            }
    
    def _validate_parameters(self, workspace: str, project: str, version: str, api_key: str, output_format: str) -> Dict[str, Any]:
        """Validate download parameters dengan comprehensive checking."""
        errors = []
        
        # Required fields
        if not workspace: errors.append("Workspace tidak boleh kosong")
        if not project: errors.append("Project tidak boleh kosong")  
        if not version: errors.append("Version tidak boleh kosong")
        if not api_key: errors.append("API key tidak boleh kosong")
        
        # API key format
        if api_key and len(api_key) < 10:
            errors.append("API key terlalu pendek")
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def _setup_download_paths(self, workspace: str, project: str, version: str) -> Dict[str, Path]:
        """Setup download paths berdasarkan environment."""
        from smartcash.common.constants.paths import get_paths_for_environment
        
        # Get environment paths
        env_paths = get_paths_for_environment(
            self.env_manager.is_colab,
            self.env_manager.is_drive_mounted
        )
        
        # Setup specific paths
        base_downloads = Path(env_paths['downloads'])
        dataset_name = f"{workspace}_{project}_v{version}"
        
        temp_dir = base_downloads / f"{dataset_name}_temp"
        final_dir = Path(env_paths['data_root'])
        
        # Ensure directories exist
        temp_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'temp_dir': temp_dir,
            'final_dir': final_dir,
            'dataset_name': dataset_name,
            'base_downloads': base_downloads
        }
    
    def _handle_backup_existing(self, paths: Dict[str, Path]) -> None:
        """Handle backup existing dataset jika ada."""
        final_dir = paths['final_dir']
        
        # Check jika ada dataset existing
        if self._has_existing_dataset(final_dir):
            self.logger.info("💾 Backing up existing dataset...")
            
            backup_result = self._backup_existing_dataset(final_dir)
            
            if backup_result['success']:
                self.logger.info(f"✅ Backup created: {backup_result['backup_path']}")
            else:
                self.logger.warning(f"⚠️ Backup failed: {backup_result['message']}")
    
    def _has_existing_dataset(self, dataset_dir: Path) -> bool:
        """Check apakah ada existing dataset."""
        if not dataset_dir.exists():
            return False
        
        # Check untuk train/valid/test structure
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists() and any(split_dir.iterdir()):
                return True
        
        return False
    
    def _backup_existing_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        """Backup existing dataset."""
        try:
            import shutil
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = dataset_dir.parent / f"dataset_backup_{timestamp}"
            
            shutil.copytree(dataset_dir, backup_dir)
            
            return {
                'success': True,
                'backup_path': str(backup_dir),
                'message': 'Backup berhasil'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Backup gagal: {str(e)}"
            }
    
    def _cleanup_temp_files(self, temp_dir: Path) -> None:
        """Cleanup temporary files."""
        try:
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.logger.debug(f"🗑️ Temp files cleaned: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup warning: {str(e)}")
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress melalui callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception as e:
                self.logger.error(f"❌ Progress callback error: {str(e)}")
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'status': 'error',
            'message': message,
            'duration': 0
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information untuk debugging."""
        return {
            'has_progress_callback': self._progress_callback is not None,
            'environment': {
                'is_colab': self.env_manager.is_colab,
                'drive_mounted': self.env_manager.is_drive_mounted
            },
            'components': {
                'roboflow_client': bool(self.roboflow_client),
                'file_processor': bool(self.file_processor),
                'validator': bool(self.validator),
                'progress_tracker': bool(self.progress_tracker)
            }
        }

# Factory function
def create_download_service(config: Dict[str, Any], logger=None) -> DownloadService:
    """Factory untuk create download service."""
    return DownloadService(config, logger)