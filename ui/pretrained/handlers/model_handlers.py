"""
File: smartcash/ui/pretrained/handlers/model_handlers.py
Deskripsi: Handlers untuk operasi terkait model pretrained
"""

import logging
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from pathlib import Path
import shutil

if TYPE_CHECKING:
    from smartcash.common.logger import LoggerBridge
else:
    LoggerBridge = Any  # For runtime type hints

from smartcash.ui.pretrained.services.model_checker import check_model_exists
from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader

class ModelOperationError(Exception):
    """Custom exception untuk operasi model"""
    pass

class ModelOperationHandler:
    """Handler untuk operasi-operasi terkait model"""
    
    def __init__(self, ui_components: Dict[str, Any], logger_bridge: Optional[LoggerBridge] = None):
        """Initialize model operation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
            logger_bridge: LoggerBridge instance untuk logging terpusat. 
                         Jika tidak disediakan, akan mencoba mengambil dari ui_components.
        """
        self.ui_components = ui_components
        self.logger_bridge = logger_bridge or ui_components.get('logger_bridge')
        self.last_models_dir = None
        
        # Initialize downloader with logger bridge
        self.downloader = PretrainedModelDownloader()
        if hasattr(self.downloader, 'set_logger_bridge') and self.logger_bridge:
            self.downloader.set_logger_bridge(self.logger_bridge)
            
        # Log initialization
        self._log_debug("ModelOperationHandler initialized")
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'debug'):
            self.logger_bridge.debug(message, **kwargs)
            
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'info'):
            self.logger_bridge.info(message, **kwargs)
            
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'warning'):
            self.logger_bridge.warning(message, **kwargs)
            
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message using logger_bridge if available"""
        if self.logger_bridge and hasattr(self.logger_bridge, 'error'):
            self.logger_bridge.error(message, exc_info=exc_info, **kwargs)
    
    def set_ui_operation_state(self, in_progress: bool) -> None:
        """Set UI state selama operasi berlangsung
        
        Args:
            in_progress: Apakah operasi sedang berlangsung
            
        Raises:
            ModelOperationError: Jika gagal mengubah state UI
        """
        try:
            self._log_debug(f"Setting UI operation state: in_progress={in_progress}")
            # Clear logs jika memulai operasi baru
            if in_progress and 'log_output' in self.ui_components:
                if hasattr(self.ui_components['log_output'], 'clear_output'):
                    self.ui_components['log_output'].clear_output()
            
            # Toggle state tombol
            button_keys = ['download_button', 'sync_button']
            for key in button_keys:
                if key in self.ui_components and hasattr(self.ui_components[key], 'disabled'):
                    self.ui_components[key].disabled = in_progress
            
            # Nonaktifkan input fields
            input_keys = ['model_dir_input', 'yolo_url_input', 'efficientnet_url_input']
            for key in input_keys:
                if key in self.ui_components and hasattr(self.ui_components[key], 'disabled'):
                    self.ui_components[key].disabled = in_progress
            
            if self.logger_bridge and in_progress:
                self.logger_bridge.debug("UI operation state updated", {"in_progress": in_progress})
                
        except Exception as e:
            error_msg = f"Gagal mengubah state UI: {str(e)}"
            if self.logger_bridge:
                self.logger_bridge.error(error_msg, exc_info=True)
            raise ModelOperationError(error_msg) from e
    
    def update_download_progress(self, progress: int, message: str = "") -> None:
        """Update progress download di UI
        
        Args:
            progress: Persentase progress (0-100)
            message: Pesan status opsional
            
        Raises:
            ModelOperationError: Jika gagal mengupdate progress
        """
        try:
            # Update progress tracker jika tersedia
            if 'progress_tracker' in self.ui_components:
                if hasattr(self.ui_components['progress_tracker'], 'update_progress'):
                    self.ui_components['progress_tracker'].update_progress(progress, message)
            
            # Update status panel jika tersedia
            status_panel = self.ui_components.get('status_panel')
            if status_panel is not None:
                if message:
                    status_panel.value = f"⏳ {message} ({progress}%)"
                else:
                    status_panel.value = f"⏳ Mengunduh... ({progress}%)"
                
                # Tampilkan pesan selesai
                if progress >= 100:
                    status_panel.value = "✅ Download selesai!"
            
            # Log progress jika logger tersedia
            if self.logger_bridge and progress % 10 == 0:  # Log setiap 10%
                self.logger_bridge.debug("Download progress updated", {
                    "progress": progress,
                    "message": message
                })
                
        except Exception as e:
            error_msg = f"Gagal mengupdate progress: {str(e)}"
            if self.logger_bridge:
                self.logger_bridge.error(error_msg, exc_info=True)
            raise ModelOperationError(error_msg) from e
    
    def cleanup_old_models_dir(self, old_dir: str, new_dir: str) -> None:
        """Bersihkan direktori model lama jika berbeda dengan yang baru
        
        Args:
            old_dir: Path direktori model lama
            new_dir: Path direktori model baru
            
        Raises:
            ModelOperationError: Jika gagal membersihkan direktori lama
        """
        if not old_dir or old_dir == new_dir:
            return
            
        try:
            old_path = Path(old_dir)
            if old_path.exists() and old_path.is_dir():
                if self.logger_bridge:
                    self.logger_bridge.info(f"Membersihkan direktori model lama: {old_dir}")
                
                shutil.rmtree(old_dir, ignore_errors=True)
                
                if self.logger_bridge:
                    self.logger_bridge.info("Direktori model lama berhasil dibersihkan")
                    
        except Exception as e:
            error_msg = f"Gagal membersihkan direktori model lama: {str(e)}"
            if self.logger_bridge:
                self.logger_bridge.warning(error_msg, exc_info=True)
            raise ModelOperationError(error_msg) from e
    
    def check_and_download_model(self, config: Dict[str, Any]) -> bool:
        """Cek dan download model jika diperlukan
        
        Args:
            config: Konfigurasi sistem
            
        Returns:
            bool: True jika model tersedia atau berhasil didownload
            
        Raises:
            ModelOperationError: Jika terjadi error saat proses
        """
        try:
            self.set_ui_operation_state(True)
            
            pretrained_config = config.get('pretrained_models', {})
            models_dir = pretrained_config.get('models_dir', '/data/pretrained')
            model_type = 'yolov5s'  # Hardcoded sesuai permintaan
            
            if self.logger_bridge:
                self.logger_bridge.info("Memeriksa ketersediaan model", {
                    "model_type": model_type,
                    "models_dir": models_dir
                })
            
            # Bersihkan direktori lama jika berbeda
            if self.last_models_dir and self.last_models_dir != models_dir:
                self.cleanup_old_models_dir(self.last_models_dir, models_dir)
            self.last_models_dir = models_dir
            
            # Cek ketersediaan model
            if check_model_exists(models_dir, model_type):
                if self.logger_bridge:
                    self.logger_bridge.info("Model sudah tersedia", {"model_type": model_type})
                self.update_download_progress(100, "Model sudah tersedia")
                return True
                
            # Download model
            model_url = None
            if 'model_urls' in pretrained_config and pretrained_config['model_urls']:
                model_url = pretrained_config['model_urls'].get('yolov5s')
                if self.logger_bridge:
                    self.logger_bridge.info("Menggunakan URL kustom untuk download", {"url": model_url})
            
            if self.logger_bridge:
                self.logger_bridge.info("Memulai proses download model")
                
            success = self.downloader.download_yolov5s(
                models_dir=models_dir,
                progress_callback=self.update_download_progress,
                status_callback=lambda m: self._update_status(m),
                model_url=model_url
            )
            
            if success and self.logger_bridge:
                self.logger_bridge.info("Model berhasil didownload", {"model_type": model_type})
            
            return success
            
        except Exception as e:
            error_msg = f"Gagal memeriksa/mendownload model: {str(e)}"
            if self.logger_bridge:
                self.logger_bridge.error(error_msg, exc_info=True)
            raise ModelOperationError(error_msg) from e
            
        finally:
            try:
                self.set_ui_operation_state(False)
            except Exception as e:
                error_msg = f"Gagal mengembalikan state UI: {str(e)}"
                if self.logger_bridge:
                    self.logger_bridge.error(error_msg, exc_info=True)
                # Tetap lanjutkan walaupun gagal mengembalikan state UI
    
    def _update_status(self, message: str) -> None:
        """Update status message di UI
        
        Args:
            message: Pesan status
        """
        if 'status_panel' in self.ui_components:
            self.ui_components['status_panel'].value = message
