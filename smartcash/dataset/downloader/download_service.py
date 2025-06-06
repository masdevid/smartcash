"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: Optimized download service dengan one-liner style dan enhanced integration
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
    """Optimized download service dengan one-liner methods dan enhanced features."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config, self.logger, self.env_manager = config, logger or get_logger('downloader.service'), get_environment_manager()
        
        # One-liner component initialization
        api_key = config.get('api_key', '')
        self.roboflow_client, self.file_processor, self.validator, self.progress_tracker = (
            RoboflowClient(api_key, logger=logger), FileProcessor(logger), 
            DatasetValidator(logger), DownloadProgressTracker()
        )
        
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback dengan one-liner propagation"""
        self._progress_callback = callback
        [component.set_progress_callback(callback) for component in [self.roboflow_client, self.file_processor] if hasattr(component, 'set_progress_callback')]
        hasattr(self.progress_tracker, 'set_callback') and self.progress_tracker.set_callback(callback)
    
    def download_dataset(self, workspace: str, project: str, version: str, api_key: str,
                        output_format: str = 'yolov5pytorch', validate_download: bool = True,
                        organize_dataset: bool = True, backup_existing: bool = False) -> Dict[str, Any]:
        """Download dataset dengan optimized flow dan one-liner operations"""
        start_time = time.time()
        
        try:
            # Step 1: Validation dengan one-liner check
            self._notify_progress('validate', 0, 100, "🔍 Validasi parameter...")
            validation_result = self._validate_parameters(workspace, project, version, api_key, output_format)
            not validation_result['valid'] and self._return_error(f"❌ Validasi gagal: {'; '.join(validation_result['errors'])}")
            self._notify_progress('validate', 100, 100, "✅ Parameter valid")
            
            # Step 2: Metadata dengan enhanced error handling
            self._notify_progress('connect', 0, 100, "🌐 Koneksi Roboflow...")
            metadata_result = self.roboflow_client.get_dataset_metadata(workspace, project, version, output_format)
            metadata_result['status'] != 'success' and self._return_error(f"❌ Metadata gagal: {metadata_result['message']}")
            metadata, download_url = metadata_result['data'], metadata_result['download_url']
            self._notify_progress('connect', 100, 100, "✅ Metadata diperoleh")
            
            # Step 3: Setup paths dengan environment awareness
            paths = self._setup_download_paths(workspace, project, version)
            
            # Step 4: Backup dengan conditional execution
            backup_existing and self._handle_backup_existing(paths)
            
            # Step 5: Download dengan progress tracking
            self._notify_progress('download', 0, 100, "📥 Memulai download...")
            download_result = self.roboflow_client.download_dataset(download_url, paths['temp_dir'] / 'dataset.zip')
            download_result['status'] != 'success' and self._return_error(f"❌ Download gagal: {download_result['message']}")
            
            # Step 6: Extract dengan enhanced progress
            self._notify_progress('extract', 0, 100, "📦 Ekstrak dataset...")
            extract_result = self.file_processor.extract_zip(Path(download_result['file_path']), paths['temp_dir'] / 'extracted')
            extract_result['status'] != 'success' and self._return_error(f"❌ Ekstrak gagal: {extract_result['message']}")
            
            # Step 7: Organize dengan conditional flow
            final_stats = (self._organize_dataset_flow(paths, organize_dataset) if organize_dataset 
                          else self._move_dataset_flow(paths))
            
            # Step 8: Validation dengan conditional execution
            validate_download and self._validate_download_results(paths['final_dir'], final_stats)
            
            # Cleanup dan success return
            self._cleanup_temp_files(paths['temp_dir'])
            return self._success_result(paths, final_stats, time.time() - start_time, workspace, project, version, output_format)
            
        except Exception as e:
            return self._error_result(str(e), time.time() - start_time)
    
    def _validate_parameters(self, workspace: str, project: str, version: str, api_key: str, output_format: str) -> Dict[str, Any]:
        """One-liner parameter validation"""
        errors = [msg for value, msg in [(workspace, "Workspace kosong"), (project, "Project kosong"), (version, "Version kosong"), (api_key, "API key kosong")] if not value]
        api_key and len(api_key) < 10 and errors.append("API key terlalu pendek")
        return {'valid': not errors, 'errors': errors}
    
    def _setup_download_paths(self, workspace: str, project: str, version: str) -> Dict[str, Path]:
        """Setup paths dengan one-liner environment detection"""
        from smartcash.common.constants.paths import get_paths_for_environment
        env_paths = get_paths_for_environment(self.env_manager.is_colab, self.env_manager.is_drive_mounted)
        
        # One-liner path creation
        base_downloads, dataset_name = Path(env_paths['downloads']), f"{workspace}_{project}_v{version}"
        temp_dir, final_dir = base_downloads / f"{dataset_name}_temp", Path(env_paths['data_root'])
        
        # One-liner directory creation
        [path.mkdir(parents=True, exist_ok=True) for path in [temp_dir, final_dir]]
        
        return {'temp_dir': temp_dir, 'final_dir': final_dir, 'dataset_name': dataset_name, 'base_downloads': base_downloads}
    
    def _handle_backup_existing(self, paths: Dict[str, Path]) -> None:
        """Handle backup dengan one-liner conditional execution"""
        final_dir = paths['final_dir']
        self._has_existing_dataset(final_dir) and (
            self.logger.info("💾 Backup existing dataset..."),
            backup_result := self._backup_existing_dataset(final_dir),
            self.logger.info(f"✅ Backup: {backup_result['backup_path']}") if backup_result['success'] 
            else self.logger.warning(f"⚠️ Backup gagal: {backup_result['message']}")
        )
    
    def _organize_dataset_flow(self, paths: Dict[str, Path], organize_dataset: bool) -> Dict[str, Any]:
        """Organize dataset dengan one-liner flow"""
        self._notify_progress('organize', 0, 100, "🗂️ Organisir struktur...")
        organize_result = self.file_processor.organize_dataset(paths['temp_dir'] / 'extracted', paths['final_dir'])
        organize_result['status'] != 'success' and self._return_error(f"❌ Organisasi gagal: {organize_result['message']}")
        return {'total_images': organize_result['total_images'], 'total_labels': organize_result['total_labels'], 'splits': organize_result['splits']}
    
    def _move_dataset_flow(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        """Move dataset tanpa organize dengan one-liner execution"""
        import shutil
        paths['final_dir'].exists() and shutil.rmtree(paths['final_dir'])
        shutil.move(str(paths['temp_dir'] / 'extracted'), str(paths['final_dir']))
        return {'total_images': 0, 'total_labels': 0, 'splits': {}}
    
    def _validate_download_results(self, final_dir: Path, final_stats: Dict[str, Any]) -> None:
        """Validate hasil download dengan one-liner conditional"""
        self._notify_progress('organize', 80, 100, "✅ Validasi hasil...")
        validate_result = self.validator.validate_extracted_dataset(final_dir)
        (not validate_result['valid'] and validate_result.get('issues') and 
         self.logger.warning(f"⚠️ Validasi issues: {', '.join(validate_result['issues'][:3])}") and
         final_stats.update(validate_result))
    
    def _has_existing_dataset(self, dataset_dir: Path) -> bool:
        """Check existing dataset dengan one-liner validation"""
        return (dataset_dir.exists() and 
                any((dataset_dir / split).exists() and any((dataset_dir / split).iterdir()) 
                    for split in ['train', 'valid', 'test']))
    
    def _backup_existing_dataset(self, dataset_dir: Path) -> Dict[str, Any]:
        """Backup dengan one-liner execution dan error handling"""
        try:
            import shutil
            from datetime import datetime
            timestamp, backup_dir = datetime.now().strftime("%Y%m%d_%H%M%S"), dataset_dir.parent / f"dataset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(dataset_dir, backup_dir)
            return {'success': True, 'backup_path': str(backup_dir), 'message': 'Backup berhasil'}
        except Exception as e:
            return {'success': False, 'message': f"Backup gagal: {str(e)}"}
    
    def _cleanup_temp_files(self, temp_dir: Path) -> None:
        """Cleanup dengan one-liner safe execution"""
        try:
            temp_dir.exists() and __import__('shutil').rmtree(temp_dir, ignore_errors=True) and self.logger.debug(f"🗑️ Temp cleaned: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup warning: {str(e)}")
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """One-liner progress notification dengan error protection"""
        self._progress_callback and (lambda: self._progress_callback(step, current, total, message))() if True else None
    
    def _success_result(self, paths: Dict[str, Path], final_stats: Dict[str, Any], duration: float, 
                       workspace: str, project: str, version: str, output_format: str) -> Dict[str, Any]:
        """One-liner success result creation"""
        return {
            'status': 'success', 'output_dir': str(paths['final_dir']), 'stats': final_stats, 'duration': duration,
            'drive_storage': self.env_manager.is_drive_mounted,
            'metadata': {'workspace': workspace, 'project': project, 'version': version, 'format': output_format}
        }
    
    def _error_result(self, message: str, duration: float) -> Dict[str, Any]:
        """One-liner error result dengan logging"""
        error_msg = f"Download service error: {message}"
        self.logger.error(f"❌ {error_msg}")
        return {'status': 'error', 'message': error_msg, 'duration': duration}
    
    def _return_error(self, message: str) -> None:
        """One-liner error return yang raise exception"""
        raise Exception(message)
    
    def get_service_info(self) -> Dict[str, Any]:
        """One-liner service info untuk debugging"""
        return {
            'has_progress_callback': self._progress_callback is not None,
            'environment': {'is_colab': self.env_manager.is_colab, 'drive_mounted': self.env_manager.is_drive_mounted},
            'components': {comp: bool(getattr(self, comp, None)) for comp in ['roboflow_client', 'file_processor', 'validator', 'progress_tracker']}
        }

# One-liner factory function
def create_download_service(config: Dict[str, Any], logger=None) -> DownloadService:
    """Factory untuk create optimized download service"""
    return DownloadService(config, logger)