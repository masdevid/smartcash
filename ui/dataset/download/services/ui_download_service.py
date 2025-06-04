"""
File: smartcash/ui/dataset/download/services/ui_download_service.py
Deskripsi: Service untuk UI download integration dengan progress tracking dan observer support
"""

from typing import Dict, Any, Callable, Optional
import time
from smartcash.dataset.services.downloader.roboflow_downloader import RoboflowDownloader
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

class UIDownloadService:
    """Service untuk UI download dengan comprehensive progress tracking dan observer integration."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.observer_manager = ui_components.get('observer_manager')
        self.start_time = None
        
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download dataset dengan comprehensive progress tracking."""
        self.start_time = time.time()
        
        try:
            # Emit download start event
            self._emit_progress_event('DOWNLOAD_START', message="🚀 Memulai download dataset", operation='download')
            
            # Step 1: Setup downloader (10%)
            self._update_progress(10, "🔧 Setup downloader...")
            downloader = RoboflowDownloader(logger=self.logger)
            self._setup_downloader_callbacks(downloader)
            
            # Step 2: Download dataset (10-60%)
            self._update_progress(15, "📥 Downloading dataset dari Roboflow...")
            download_result = downloader.download_dataset(
                workspace=params['workspace'],
                project=params['project'], 
                version=params['version'],
                api_key=params['api_key'],
                output_dir=params['output_dir']
            )
            
            if not download_result.get('success', False):
                error_msg = download_result.get('message', 'Download gagal')
                self._emit_progress_event('DOWNLOAD_ERROR', message=error_msg, error_details=error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Step 3: Organize dataset (60-90%)
            self._update_progress(65, "📁 Mengorganisir dataset ke struktur final...")
            organize_result = self._organize_downloaded_dataset(download_result, params)
            
            if not organize_result.get('success', False):
                error_msg = organize_result.get('message', 'Organisasi dataset gagal')
                self._emit_progress_event('DOWNLOAD_ERROR', message=error_msg, error_details=error_msg)
                return {'status': 'error', 'message': error_msg}
            
            # Step 4: Final validation (90-100%)
            self._update_progress(95, "✅ Validasi hasil download...")
            final_stats = self._validate_final_result(organize_result)
            
            # Success completion
            duration = time.time() - self.start_time
            success_result = {
                'status': 'success',
                'message': 'Download dan organisasi berhasil',
                'duration': duration,
                'stats': final_stats,
                'output_dir': params['output_dir'],
                'drive_storage': self._is_drive_storage()
            }
            
            self._emit_progress_event('DOWNLOAD_COMPLETE', message="🎉 Download selesai!", **success_result)
            return success_result
            
        except Exception as e:
            error_msg = f"Download service error: {str(e)}"
            self.logger and self.logger.error(f"💥 {error_msg}")
            self._emit_progress_event('DOWNLOAD_ERROR', message=error_msg, error_details=str(e))
            return {'status': 'error', 'message': error_msg}
    
    def _setup_downloader_callbacks(self, downloader: RoboflowDownloader) -> None:
        """Setup callbacks untuk downloader progress."""
        def download_progress_callback(step: str, current: int, total: int, message: str):
            """Callback untuk download progress dari RoboflowDownloader."""
            if step == 'download':
                # Map download progress ke overall progress (15-60%)
                download_progress = 15 + int((current / max(total, 1)) * 45)
                self._update_progress(download_progress, f"📥 {message}")
                
                # Emit step progress untuk detailed tracking
                self._emit_progress_event('DOWNLOAD_STEP_PROGRESS', 
                                        step_name='download',
                                        progress=int((current / max(total, 1)) * 100),
                                        message=message)
            
            elif step == 'extract':
                # Map extract progress ke overall progress (50-65%)
                extract_progress = 50 + int((current / max(total, 1)) * 15)
                self._update_progress(extract_progress, f"📦 {message}")
                
                self._emit_progress_event('DOWNLOAD_STEP_PROGRESS',
                                        step_name='extract', 
                                        progress=int((current / max(total, 1)) * 100),
                                        message=message)
        
        # Set callback ke downloader
        downloader.set_progress_callback(download_progress_callback)
    
    def _organize_downloaded_dataset(self, download_result: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Organize downloaded dataset ke struktur final."""
        try:
            organizer = DatasetOrganizer(logger=self.logger)
            
            # Setup organizer callback
            def organize_progress_callback(step: str, current: int, total: int, message: str):
                if step == 'organize':
                    # Map organize progress ke overall progress (65-90%)
                    organize_progress = 65 + int((current / max(total, 1)) * 25)
                    self._update_progress(organize_progress, f"📁 {message}")
                    
                    self._emit_progress_event('DOWNLOAD_STEP_PROGRESS',
                                            step_name='organize',
                                            progress=int((current / max(total, 1)) * 100), 
                                            message=message)
            
            organizer.set_progress_callback(organize_progress_callback)
            
            # Execute organization
            source_dir = download_result.get('extracted_path', download_result.get('output_dir'))
            organize_result = organizer.organize_dataset_structure(
                source_dir=source_dir,
                ensure_yolo_format=True
            )
            
            if organize_result.get('status') == 'success':
                self._emit_progress_event('DOWNLOAD_STEP_COMPLETE', 
                                        step_name='organize',
                                        message="Dataset berhasil diorganisir")
                return {'success': True, 'organize_result': organize_result}
            else:
                return {'success': False, 'message': organize_result.get('message', 'Organize gagal')}
                
        except Exception as e:
            error_msg = f"Error organizing dataset: {str(e)}"
            self.logger and self.logger.error(f"💥 {error_msg}")
            return {'success': False, 'message': error_msg}
    
    def _validate_final_result(self, organize_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate final result dan generate statistics."""
        try:
            from smartcash.ui.dataset.download.utils.dataset_checker import check_complete_dataset_status
            
            # Check dataset status setelah organize
            dataset_status = check_complete_dataset_status()
            final_dataset = dataset_status.get('final_dataset', {})
            
            # Generate stats
            stats = {
                'total_images': final_dataset.get('total_images', 0),
                'total_labels': final_dataset.get('total_labels', 0),
                'train_images': final_dataset.get('splits', {}).get('train', {}).get('images', 0),
                'valid_images': final_dataset.get('splits', {}).get('valid', {}).get('images', 0),
                'test_images': final_dataset.get('splits', {}).get('test', {}).get('images', 0),
                'organize_stats': organize_result.get('organize_result', {}).get('stats', {})
            }
            
            return stats
            
        except Exception as e:
            self.logger and self.logger.warning(f"⚠️ Error validating results: {str(e)}")
            return {'total_images': 0, 'total_labels': 0}
    
    def _update_progress(self, progress: int, message: str, color: str = None) -> None:
        """Update progress dengan UI integration."""
        # Emit progress event untuk observer
        self._emit_progress_event('DOWNLOAD_PROGRESS', 
                                progress=progress, 
                                message=message)
        
        # Direct UI update jika ada
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('overall', progress, message, color)
        elif 'tracker' in self.ui_components:
            self.ui_components['tracker'].update('overall', progress, message, color)
    
    def _emit_progress_event(self, event_type: str, **kwargs) -> None:
        """Emit progress event ke observer manager."""
        if self.observer_manager and hasattr(self.observer_manager, 'notify'):
            try:
                self.observer_manager.notify(event_type, sender=self, **kwargs)
            except Exception as e:
                self.logger and self.logger.debug(f"🔔 Observer notify error: {str(e)}")
    
    def _is_drive_storage(self) -> bool:
        """Check apakah menggunakan Google Drive storage."""
        try:
            env_manager = self.ui_components.get('env_manager')
            return env_manager and env_manager.is_drive_mounted if env_manager else False
        except Exception:
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status untuk debugging."""
        return {
            'ui_components_count': len(self.ui_components),
            'has_logger': self.logger is not None,
            'has_observer_manager': self.observer_manager is not None,
            'progress_methods': {
                'update_progress': 'update_progress' in self.ui_components,
                'tracker': 'tracker' in self.ui_components
            },
            'env_manager': 'env_manager' in self.ui_components,
            'current_operation': 'download' if self.start_time else None
        }


class DownloadServiceFactory:
    """Factory untuk create UIDownloadService dengan dependency injection."""
    
    @staticmethod
    def create_service(ui_components: Dict[str, Any], 
                      logger: Optional[Any] = None,
                      observer_manager: Optional[Any] = None) -> UIDownloadService:
        """Create UIDownloadService dengan optional dependency override."""
        # Override dependencies jika disediakan
        if logger:
            ui_components['logger'] = logger
        if observer_manager:
            ui_components['observer_manager'] = observer_manager
        
        return UIDownloadService(ui_components)
    
    @staticmethod
    def create_minimal_service(logger: Optional[Any] = None) -> UIDownloadService:
        """Create minimal service untuk testing atau standalone use."""
        minimal_components = {
            'logger': logger,
            'observer_manager': None,
            'update_progress': lambda *args: None,
            'tracker': None
        }
        return UIDownloadService(minimal_components)


# Helper functions untuk backward compatibility
def create_download_service(ui_components: Dict[str, Any]) -> UIDownloadService:
    """Create download service dengan standard configuration."""
    return DownloadServiceFactory.create_service(ui_components)

def download_dataset_with_ui(ui_components: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Download dataset dengan UI integration - one-liner wrapper."""
    return create_download_service(ui_components).download_dataset(params)

def get_download_service_info(service: UIDownloadService) -> Dict[str, Any]:
    """Get download service information untuk debugging."""
    return {
        'service_status': service.get_service_status(),
        'service_class': service.__class__.__name__,
        'active_operation': service.start_time is not None
    }