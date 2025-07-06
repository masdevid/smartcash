"""
File: smartcash/ui/pretrained/handlers/progress_handlers.py
Deskripsi: Handlers untuk progress tracking di pretrained module
"""

from typing import Dict, Any, Optional, Callable, Union
from functools import partial

from smartcash.ui.utils import get_logger, UILogger
from smartcash.ui.pretrained.utils.progress_adapter import PretrainedProgressAdapter, create_progress_adapter
from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader
from smartcash.ui.pretrained.services.model_syncer import PretrainedModelSyncer
from smartcash.ui.pretrained.services.model_checker import check_model_exists

# Get module logger
logger = get_logger(__name__)

# Type aliases
LoggerBridge = Any  # Type alias for logger bridge

class ProgressHandlers:
    """Class untuk menangani progress tracking di pretrained module"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize progress handlers
        
        Args:
            ui_components: Dictionary berisi UI components
            config: Konfigurasi sistem
        """
        self.ui_components = ui_components
        self.config = config
        self.progress_tracker = None
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Initialize services with the module logger
        self.downloader = PretrainedModelDownloader()
        self.syncer = PretrainedModelSyncer()
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message"""
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def _log_success(self, message: str, **kwargs) -> None:
        """Log success message"""
        self.logger.info(f"✅ {message}", **kwargs)
    
    def _log_to_output(self, message: str, level: str = "info") -> None:
        """Log message to output console"""
        try:
            if 'output_console' in self.ui_components:
                # Use the global log_to_ui function which handles the UI output
                log_to_ui(self.ui_components, message, level=level)
        except Exception as e:
            self.logger.error(f"Gagal menulis ke output console: {e}", exc_info=True)
    
    def _update_ui_progress(self, progress: int, message: str = "") -> None:
        """Update UI progress indicators"""
        try:
            if 'main_progress' in self.ui_components:
                self.ui_components['main_progress'].value = progress
            if 'sub_progress' in self.ui_components and message:
                self.ui_components['sub_progress'].description = message
            if 'status' in self.ui_components and message:
                self.ui_components['status'].value = message
                
            self._log_to_output(f"📊 {progress}% - {message}", "info")
            self._log_debug(f"Progress: {progress}% - {message}")
            
        except Exception as e:
            self._log_error(f"Gagal memperbarui UI progress: {str(e)}")
    
    def _update_ui_status(self, message: str) -> None:
        """Update UI status message"""
        try:
            if 'status' in self.ui_components:
                self.ui_components['status'].value = message
            self._log_to_output(message, "info")
            self._log_info(f"Status: {message}")
        except Exception as e:
            self._log_error(f"Gagal memperbarui status UI: {str(e)}")
    
    def _init_progress_tracker(self) -> None:
        """Initialize progress tracker with UI components"""
        try:
            # Create progress tracker with UI components
            self.progress_tracker = create_progress_adapter({
                'progress_bar': self.ui_components.get('main_progress')
            })
            
            # Configure callbacks
            self.downloader.set_progress_callbacks(
                progress_cb=self.progress_tracker,
                status_cb=self._update_ui_status
            )
            
        except Exception as e:
            self._log_error(f"Gagal menginisialisasi progress tracker: {str(e)}")
            # Fallback to basic callbacks
            self.downloader.set_progress_callbacks(
                progress_cb=self._update_ui_progress,
                status_cb=self._update_ui_status
            )
    
    def _recheck_model_status(self) -> None:
        """Re-check model status and update UI"""
        try:
            pretrained_config = self.config.get('pretrained_models', {})
            models_dir = pretrained_config.get('models_dir', '/content/models')
            drive_dir = pretrained_config.get('drive_models_dir', '/data/pretrained')
            model_type = 'yolov5s'  # Fixed to YOLOv5s only
            
            # Check model existence
            local_exists = check_model_exists(models_dir, model_type)
            drive_exists = check_model_exists(drive_dir, model_type)
            
            # Update UI based on model status
            if local_exists and drive_exists:
                self._update_ui_status("✅ Model tersedia secara lokal dan di drive")
            elif local_exists:
                self._update_ui_status("ℹ️ Model hanya tersedia secara lokal")
            elif drive_exists:
                self._update_ui_status("ℹ️ Model hanya tersedia di drive")
            else:
                self._update_ui_status("❌ Model tidak ditemukan")
                
        except Exception as e:
            self._log_error(f"Gagal memeriksa status model: {str(e)}")
    
    def get_download_sync_handler(self) -> Callable:
        """Get handler for download/sync button click"""
        def on_download_sync_click(b) -> None:
            try:
                self._log_info("Memulai operasi download & sinkronisasi...")
                self._log_to_output("🚀 Memulai operasi download & sinkronisasi...", "info")
                
                # Initialize progress tracker
                self._init_progress_tracker()
                
                # Get config
                pretrained_config = self.config.get('pretrained_models', {})
                models_dir = pretrained_config.get('models_dir', '/content/models')
                drive_dir = pretrained_config.get('drive_models_dir', '/data/pretrained')
                model_type = 'yolov5s'  # Fixed to YOLOv5s only
                
                # Get current model status
                local_exists = check_model_exists(models_dir, model_type)
                drive_exists = check_model_exists(drive_dir, model_type)
                
                # Determine operation based on current status
                if not local_exists and not drive_exists:
                    # Download new model
                    self._update_ui_status("⬇️ Mengunduh model YOLOv5s...")
                    success = self.downloader.download_yolov5s(
                        models_dir, 
                        self.progress_tracker.get_progress_callback() if self.progress_tracker else None,
                        self.progress_tracker.get_status_callback() if self.progress_tracker else None
                    )
                    
                    if success:
                        # Sync to drive after download
                        self._update_ui_status("📤 Menyinkronkan ke drive...")
                        self.syncer.sync_to_drive(
                            models_dir, 
                            drive_dir, 
                            self.progress_tracker,
                            self.progress_tracker.get_status_callback() if self.progress_tracker else None
                        )
                
                elif local_exists and not drive_exists:
                    # Sync to drive
                    self._update_ui_status("📤 Menyinkronkan model ke drive...")
                    self.syncer.sync_to_drive(
                        models_dir, 
                        drive_dir, 
                        self.progress_tracker,
                        self.progress_tracker.get_status_callback() if self.progress_tracker else None
                    )
                
                elif not local_exists and drive_exists:
                    # Sync from drive
                    self._update_ui_status("📥 Menyinkronkan model dari drive...")
                    self.syncer.sync_from_drive(
                        drive_dir, 
                        models_dir, 
                        self.progress_tracker,
                        self.progress_tracker.get_status_callback() if self.progress_tracker else None
                    )
                
                else:
                    # Both exist - bi-directional sync
                    self._update_ui_status("🔄 Melakukan sinkronisasi dua arah...")
                    self.syncer.bi_directional_sync(
                        models_dir, 
                        drive_dir, 
                        self.progress_tracker,
                        self.progress_tracker.get_status_callback() if self.progress_tracker else None
                    )
                
                # Re-check model status after operation
                self._recheck_model_status()
                
                # Final success message
                if self.progress_tracker:
                    self.progress_tracker.update_progress(100, "Operasi berhasil diselesaikan")
                self._update_ui_status("✅ Operasi berhasil diselesaikan")
                self._log_info("Operasi download & sinkronisasi selesai")
                self._log_to_output("✅ Operasi download & sinkronisasi selesai", "success")
                
            except Exception as e:
                error_msg = f"❌ Gagal melakukan operasi download/sinkronisasi: {str(e)}"
                self._log_error(error_msg, exc_info=True)
                self._log_to_output(error_msg, "error")
                self._update_ui_status(error_msg)
                raise
        
        return on_download_sync_click


def setup_progress_handlers(
    ui_components: Dict[str, Any], 
    config: Dict[str, Any], 
    logger_bridge: Optional[LoggerBridge] = None
) -> None:
    """Setup progress tracking handlers
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
        logger_bridge: Logger bridge instance untuk logging (opsional)
    """
    try:
        # Initialize progress handlers
        progress_handler = ProgressHandlers(ui_components, config, logger_bridge)
        
        # Setup download/sync button handler
        if 'download_sync_button' in ui_components:
            ui_components['download_sync_button'].on_click(
                progress_handler.get_download_sync_handler()
            )
        
        # Initial model status check
        progress_handler._recheck_model_status()
        
        if logger_bridge and hasattr(logger_bridge, 'info'):
            logger_bridge.info("Progress handlers setup completed")
            
    except Exception as e:
        error_msg = f"Error setting up progress handlers: {str(e)}"
        if logger_bridge and hasattr(logger_bridge, 'error'):
            logger_bridge.error(error_msg, exc_info=True)
