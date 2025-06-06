"""
File: smartcash/dataset/downloader/download_service.py
Deskripsi: Updated download service dengan UUID file renaming integration dan enhanced flow
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
    """Enhanced download service dengan UUID file renaming dan optimized features."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config, self.logger, self.env_manager = config, logger or get_logger('downloader.service'), get_environment_manager()
        
        # One-liner component initialization dengan enhanced file processor
        api_key = config.get('api_key', '')
        self.roboflow_client, self.file_processor, self.validator, self.progress_tracker = (
            RoboflowClient(api_key, logger=logger), FileProcessor(logger), 
            DatasetValidator(logger), DownloadProgressTracker()
        )
        
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback dengan one-liner propagation"""
        self._progress_callback = callback
        for component in [self.roboflow_client, self.file_processor]:
            if hasattr(component, 'set_progress_callback'):
                component.set_progress_callback(callback)
        if hasattr(self.progress_tracker, 'set_callback'):
            self.progress_tracker.set_callback(callback)
    
    def download_dataset(self) -> Dict[str, Any]:
        """Download dataset dengan enhanced flow including UUID renaming"""
        start_time = time.time()
        
        try:
            # Validasi config terlebih dahulu
            if not self._validate_config():
                return self._error_result("Konfigurasi tidak valid", time.time() - start_time)
                
            # Ekstrak parameter dari config
            workspace = self.config.get('workspace', '')
            project = self.config.get('project', '')
            version = self.config.get('version', '')
            api_key = self.config.get('api_key', '')
            output_format = self.config.get('output_format', 'yolov5pytorch')
            validate_download = self.config.get('validate_download', True)
            organize_dataset = self.config.get('organize_dataset', True)
            backup_existing = self.config.get('backup_existing', False)
            rename_files = self.config.get('rename_files', True)  # New option
            
            # Step 1: Validation dengan one-liner check
            self._notify_progress('validate', 0, 100, "🔍 Validasi parameter...")
            validation_result = self._validate_parameters(workspace, project, version, api_key, output_format)
            if not validation_result['valid']:
                return self._return_error(f"❌ Validasi gagal: {'; '.join(validation_result['errors'])}")
            self._notify_progress('validate', 100, 100, "✅ Parameter valid")
            
            # Step 2: Metadata dengan enhanced error handling
            self._notify_progress('connect', 0, 100, "🌐 Koneksi Roboflow...")
            metadata_result = self.roboflow_client.get_dataset_metadata(workspace, project, version, output_format)
            if metadata_result['status'] != 'success':
                return self._return_error(f"❌ Metadata gagal: {metadata_result['message']}")
            metadata, download_url = metadata_result['data'], metadata_result['download_url']
            self._notify_progress('connect', 100, 100, "✅ Metadata diperoleh")
            
            # Step 3: Setup paths dengan environment awareness
            paths = self._setup_download_paths(workspace, project, version)
            
            # Step 4: Backup dengan conditional execution
            if backup_existing:
                self._handle_backup_existing(paths)
            
            # Step 5: Download dengan progress tracking
            self._notify_progress('download', 0, 100, "📥 Memulai download...")
            download_result = self.roboflow_client.download_dataset(download_url, paths['temp_dir'] / 'dataset.zip')
            if download_result['status'] != 'success':
                return self._return_error(f"❌ Download gagal: {download_result['message']}")
            
            # Step 6: Extract dengan enhanced progress
            self._notify_progress('extract', 0, 100, "📦 Ekstrak dataset...")
            extract_result = self.file_processor.extract_zip(Path(download_result['file_path']), paths['temp_dir'] / 'extracted')
            if extract_result['status'] != 'success':
                return self._return_error(f"❌ Ekstrak gagal: {extract_result['message']}")
            
            # Step 7: Organize dengan UUID renaming flow
            final_stats = (self._organize_with_renaming_flow(paths, organize_dataset, rename_files) if organize_dataset 
                          else self._move_dataset_flow(paths))
            
            # Step 8: Validation dengan UUID format checking
            if validate_download:
                self._validate_download_results_with_uuid(paths['final_dir'], final_stats, rename_files)
            
            # Cleanup dan success return
            self._cleanup_temp_files(paths['temp_dir'])
            return self._success_result_with_uuid(paths, final_stats, time.time() - start_time, workspace, project, version, output_format, rename_files)
            
        except Exception as e:
            self.logger.error(f"💥 Download error: {str(e)}")
            return self._error_result(str(e), time.time() - start_time)
    
    def _validate_config(self) -> bool:
        """Validasi konfigurasi yang diberikan saat inisialisasi"""
        required_fields = ['workspace', 'project', 'version', 'api_key']
        missing_fields = [field for field in required_fields if field not in self.config or not self.config[field]]
        
        if missing_fields:
            error_msg = f"Konfigurasi tidak lengkap: {', '.join(missing_fields)} tidak ditemukan"
            self.logger.error(f"❌ {error_msg}")
            return False
            
        # Validasi panjang API key
        api_key = self.config.get('api_key', '')
        if len(api_key) < 10:
            self.logger.error("❌ API key terlalu pendek")
            return False
            
        return True
        
    def _validate_parameters(self, workspace: str, project: str, version: str, api_key: str, output_format: str) -> Dict[str, Any]:
        """One-liner parameter validation"""
        errors = [msg for value, msg in [(workspace, "Workspace kosong"), (project, "Project kosong"), (version, "Version kosong"), (api_key, "API key kosong")] if not value]
        if api_key and len(api_key) < 10:
            errors.append("API key terlalu pendek")
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
        if self._has_existing_dataset(final_dir):
            self.logger.info("💾 Backup existing dataset...")
            backup_result = self._backup_existing_dataset(final_dir)
            if backup_result['success']:
                self.logger.info(f"✅ Backup: {backup_result['backup_path']}")
            else:
                self.logger.warning(f"⚠️ Backup warning: {backup_result['message']}")
    
    def _organize_with_renaming_flow(self, paths: Dict[str, Path], organize_dataset: bool, rename_files: bool) -> Dict[str, Any]:
        """Enhanced organize flow dengan UUID renaming"""
        self._notify_progress('organize', 0, 100, "🗂️ Organisir struktur dengan UUID renaming...")
        
        if rename_files:
            organize_result = self.file_processor.organize_dataset_with_renaming(
                paths['temp_dir'] / 'extracted', paths['final_dir']
            )
        else:
            organize_result = self.file_processor.organize_dataset(
                paths['temp_dir'] / 'extracted', paths['final_dir']
            )
        
        if organize_result['status'] != 'success':
            return self._return_error(f"❌ Organisasi gagal: {organize_result['message']}")
        
        # Enhanced stats dengan UUID info
        stats = {
            'total_images': organize_result['total_images'], 
            'total_labels': organize_result['total_labels'], 
            'splits': organize_result['splits'],
            'uuid_renamed': organize_result.get('uuid_renamed', False)
        }
        
        if rename_files:
            naming_stats = self.file_processor.get_naming_statistics()
            stats.update({'naming_stats': naming_stats})
            self.logger.info(f"🔄 UUID renaming: {naming_stats.get('total_files', 0)} files processed")
        
        return stats
    
    def _move_dataset_flow(self, paths: Dict[str, Path]) -> Dict[str, Any]:
        """Move dataset tanpa organize dengan one-liner execution"""
        import shutil
        if paths['final_dir'].exists():
            shutil.rmtree(paths['final_dir'])
        shutil.move(str(paths['temp_dir'] / 'extracted'), str(paths['final_dir']))
        return {'total_images': 0, 'total_labels': 0, 'splits': {}, 'uuid_renamed': False}
    
    def _validate_download_results_with_uuid(self, final_dir: Path, final_stats: Dict[str, Any], uuid_format: bool) -> None:
        """Validate hasil download dengan UUID format checking"""
        self._notify_progress('organize', 80, 100, "✅ Validasi hasil dengan UUID check...")
        validate_result = self.file_processor.validate_dataset_structure(final_dir)
        
        if not validate_result['valid']:
            if validate_result.get('issues'):
                self.logger.warning(f"⚠️ Validasi issues: {', '.join(validate_result['issues'][:3])}")
            final_stats.update(validate_result)
        
        # Enhanced validation untuk UUID format
        if uuid_format and not validate_result.get('uuid_format', True):
            self.logger.warning("⚠️ Beberapa file tidak konsisten dengan format UUID")
            final_stats['uuid_issues'] = validate_result.get('issues', [])
    
    def _has_existing_dataset(self, dataset_dir: Path) -> bool:
        """Check existing dataset dengan validasi yang aman"""
        if not dataset_dir.exists():
            return False
            
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists() and any(split_dir.iterdir()):
                return True
                
        return False
    
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
            if temp_dir.exists():
                __import__('shutil').rmtree(temp_dir, ignore_errors=True)
                self.logger.debug(f"🗑️ Temp cleaned: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup warning: {str(e)}")
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """One-liner progress notification dengan error protection"""
        if self._progress_callback:
            self._progress_callback(step, current, total, message)
    
    def _success_result_with_uuid(self, paths: Dict[str, Path], final_stats: Dict[str, Any], duration: float, 
                                 workspace: str, project: str, version: str, output_format: str, uuid_renamed: bool) -> Dict[str, Any]:
        """Enhanced success result dengan UUID info"""
        return {
            'status': 'success', 'output_dir': str(paths['final_dir']), 'stats': final_stats, 'duration': duration,
            'drive_storage': self.env_manager.is_drive_mounted, 'uuid_renamed': uuid_renamed,
            'metadata': {'workspace': workspace, 'project': project, 'version': version, 'format': output_format},
            'naming_info': final_stats.get('naming_stats', {}) if uuid_renamed else {}
        }
    
    def _error_result(self, message: str, duration: float) -> Dict[str, Any]:
        """One-liner error result dengan logging"""
        error_msg = f"Download service error: {message}"
        self.logger.error(f"❌ {error_msg}")
        return {'status': 'error', 'message': error_msg, 'duration': duration}
    
    def _return_error(self, message: str) -> None:
        """Return error dengan one-liner exception raising"""
        self.logger.error(message)
        raise Exception(message)
        
    def _notify_error(self, message: str) -> Dict[str, Any]:
        """Notify error dan return error result tanpa raise exception"""
        self.logger.error(message)
        return self._error_result(message, 0)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Enhanced service info dengan UUID support info"""
        return {
            'has_progress_callback': self._progress_callback is not None,
            'environment': {'is_colab': self.env_manager.is_colab, 'drive_mounted': self.env_manager.is_drive_mounted},
            'components': {comp: bool(getattr(self, comp, None)) for comp in ['roboflow_client', 'file_processor', 'validator', 'progress_tracker']},
            'uuid_support': True, 'file_naming_enabled': hasattr(self.file_processor, 'naming_manager')
        }

# Factory function dengan enhanced error handling dan UUID support
def create_download_service(config: Dict[str, Any], logger=None) -> Optional[DownloadService]:
    """Factory untuk create enhanced download service dengan UUID renaming support"""
    logger = logger or get_logger('downloader.factory')
    
    try:
        # Validasi config dasar
        if not config:
            logger.error("❌ Config kosong")
            return None
        
        # Validasi field yang diperlukan
        required_fields = ['workspace', 'project', 'version', 'api_key']
        default_values = {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3'
        }
        
        # Terapkan nilai default jika field tidak ada atau kosong
        for field in required_fields:
            if field not in config or not config[field]:
                if field in default_values:
                    config[field] = default_values[field]
                    logger.info(f"🔄 Menggunakan nilai default untuk {field}: {config[field]}")
        
        # Default UUID renaming ke True jika tidak disebutkan
        if 'rename_files' not in config:
            config['rename_files'] = True
            logger.info("🔄 UUID file renaming diaktifkan secara default")
        
        # Periksa lagi setelah menerapkan nilai default
        missing_fields = [field for field in required_fields if field not in config or not config[field]]
        
        if missing_fields:
            logger.error(f"❌ Konfigurasi tidak lengkap: {', '.join(missing_fields)} tidak ditemukan")
            return None
            
        # Buat instance dengan error handling
        service = DownloadService(config, logger)
        
        # Validasi komponen penting
        if not service.roboflow_client or not service.file_processor:
            logger.error("❌ Komponen download service tidak lengkap")
            return None
            
        return service
    except Exception as e:
        logger.error(f"❌ Error saat membuat download service: {str(e)}")
        return None