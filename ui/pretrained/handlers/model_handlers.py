"""
File: smartcash/ui/pretrained/handlers/model_handlers.py
Deskripsi: Handlers untuk operasi terkait model pretrained
"""

import logging
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING, List, Union
from pathlib import Path
import shutil

from smartcash.ui.utils.ui_logger import UILogger, get_module_logger

from smartcash.ui.pretrained.services.model_checker import check_model_exists
from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader
from smartcash.ui.pretrained.utils import (
    with_error_handling,
    log_errors,
    get_logger
)
from smartcash.ui.utils.error_utils import create_error_context

@with_error_handling(component="model_handlers", operation="ModelOperationError")
class ModelOperationError(Exception):
    """Custom exception untuk operasi model"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize ModelOperationError
        
        Args:
            message: Pesan error yang deskriptif
            error_code: Kode error opsional untuk penanganan spesifik
            details: Detail tambahan tentang error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

@with_error_handling(component="model_handlers", operation="ModelOperationHandler")
class ModelOperationHandler:
    """Handler untuk operasi-operasi terkait model"""
    
    def __init__(self, ui_components: Dict[str, Any], logger: Optional[UILogger] = None):
        """Initialize model operation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
            logger: UILogger instance untuk logging terpusat. 
                   Jika tidak disediakan, akan mencoba mengambil dari ui_components.
        """
        self.ui_components = ui_components
        self.logger = logger or ui_components.get('logger')
        if not self.logger:
            self.logger = get_module_logger('smartcash.ui.pretrained.handlers.model_handlers')
        self.last_models_dir = None
        
        # Initialize downloader with logger
        self.downloader = PretrainedModelDownloader()
        if hasattr(self.downloader, 'set_logger'):
            self.downloader.set_logger(self.logger)
            
        # Log initialization
        self._log_debug("ModelOperationHandler initialized")
    
    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log message using UILogger
        
        Args:
            message: Pesan yang akan dicatat
            level: Tingkat log ('debug', 'info', 'warning', 'error')
            **kwargs: Argumen tambahan untuk logging
        """
        if not self.logger:
            self.logger = get_module_logger('smartcash.ui.pretrained.handlers.model_handlers')
            
        log_func = getattr(self.logger, level.lower(), None)
        if callable(log_func):
            log_func(message, **kwargs)
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message using UILogger"""
        self._log(message, "debug", **kwargs)
            
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message using UILogger"""
        self._log(message, "info", **kwargs)
            
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message using UILogger"""
        self._log(message, "warning", **kwargs)
            
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message using UILogger"""
        self._log(message, "error", exc_info=exc_info, **kwargs)
        
    def _create_error_context(self, **kwargs) -> Dict[str, Any]:
        """Buat konteks error yang konsisten
        
        Returns:
            Dict berisi konteks error yang relevan
        """
        return create_error_context(
            component="pretrained",
            handler="ModelOperationHandler",
            **kwargs
        )
    
    @with_error_handling(
        component="pretrained",
        operation="set_ui_operation_state"
    )
    @log_errors(level="error")
    def set_ui_operation_state(self, in_progress: bool) -> None:
        """Set UI state selama operasi berlangsung
        
        Args:
            in_progress: Apakah operasi sedang berlangsung
            
        Raises:
            ModelOperationError: Jika gagal mengubah state UI
        """
        error_context = self._create_error_context(
            operation="set_ui_operation_state",
            in_progress=in_progress
        )
        self._log_debug(f"Mengubah state UI operation: in_progress={in_progress}", extra=error_context)
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
                self.logger_bridge.debug(f"UI operation state updated - in_progress: {in_progress}")
                
        except Exception as e:
            error_msg = f"Gagal mengubah state UI: {str(e)}"
            if self.logger_bridge:
                self.logger_bridge.error(error_msg, exc_info=True)
            raise ModelOperationError(error_msg) from e
    
    @with_error_handling(
        component="pretrained",
        operation="update_download_progress"
    )
    @log_errors(level="error")
    def update_download_progress(self, progress: int, message: str = "") -> None:
        """Update progress download di UI
        
        Args:
            progress: Persentase progress (0-100)
            message: Pesan status opsional
            
        Raises:
            ModelOperationError: Jika gagal mengupdate progress
        """
        error_context = self._create_error_context(
            operation="update_download_progress",
            progress=progress
        )
        
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
        
        # Log progress setiap 10%
        if progress % 10 == 0:
            self._log_debug(
                f"Progress update: {progress}% - {message}",
                extra={"progress": progress, "message": message, **error_context}
            )
    
    @with_error_handling(
        component="pretrained",
        operation="cleanup_old_models_dir"
    )
    @log_errors(level="warning")
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
            
        error_context = self._create_error_context(
            operation="cleanup_old_models_dir",
            old_dir=old_dir,
            new_dir=new_dir
        )
        
        old_path = Path(old_dir)
        if old_path.exists() and old_path.is_dir():
            self._log_info(f"Membersihkan direktori model lama: {old_dir}", extra=error_context)
            
            shutil.rmtree(old_dir, ignore_errors=True)
            
            self._log_info("Direktori model lama berhasil dibersihkan", extra=error_context)
    
    @with_error_handling(
        component="pretrained",
        operation="check_and_download_model"
    )
    @log_errors(level="error")
    def check_and_download_model(self, config: Dict[str, Any]) -> bool:
        """Cek dan download model jika diperlukan
        
        Args:
            config: Konfigurasi sistem yang berisi pengaturan model
            
        Returns:
            bool: True jika model tersedia atau berhasil didownload
            
        Raises:
            ModelOperationError: Jika terjadi error saat proses
        """
        # Setup error context
        error_context = self._create_error_context(
            operation="check_and_download_model",
            config_keys=list(config.keys()) if config else []
        )
        
        try:
            self.set_ui_operation_state(True)
            
            pretrained_config = config.get('pretrained_models', {})
            models_dir = pretrained_config.get('models_dir', '/data/pretrained')
            model_type = 'yolov5s'  # Hardcoded sesuai permintaan
            
            self._log_info(
                f"Memeriksa ketersediaan model - Tipe: {model_type}, Direktori: {models_dir}",
                extra=error_context
            )
            
            # Bersihkan direktori lama jika berbeda
            if self.last_models_dir and self.last_models_dir != models_dir:
                self.cleanup_old_models_dir(self.last_models_dir, models_dir)
            self.last_models_dir = models_dir
            
            # Cek ketersediaan model
            if check_model_exists(models_dir, model_type):
                self._log_info(
                    f"Model {model_type} sudah tersedia di {models_dir}",
                    extra={"model_type": model_type, **error_context}
                )
                self.update_download_progress(100, "Model sudah tersedia")
                return True
                
            # Download model
            model_url = None
            if 'model_urls' in pretrained_config and pretrained_config['model_urls']:
                model_url = pretrained_config['model_urls'].get('yolov5s')
                self._log_info(
                    f"Menggunakan URL kustom untuk download: {model_url}",
                    extra={"model_url": model_url, **error_context}
                )
            
            self._log_info("Memulai proses download model", extra=error_context)
                
            success = self.downloader.download_yolov5s(
                models_dir=models_dir,
                progress_callback=self.update_download_progress,
                status_callback=lambda m: self._update_status(m),
                model_url=model_url
            )
            
            if success:
                self._log_info(
                    f"Model {model_type} berhasil didownload ke {models_dir}",
                    extra={"model_type": model_type, **error_context}
                )
            else:
                self._log_warning(
                    f"Gagal mendownload model {model_type}",
                    extra={"model_type": model_type, **error_context}
                )
            
            return success
            
        finally:
            try:
                self.set_ui_operation_state(False)
            except Exception as e:
                self._log_error(
                    "Gagal mengembalikan state UI setelah operasi model",
                    exc_info=True,
                    extra=error_context
                )
                # Tetap lanjutkan walaupun gagal mengembalikan state UI
    
    def _update_status(self, message: str) -> None:
        """Update status message di UI
        
        Args:
            message: Pesan status
        """
        if 'status_panel' in self.ui_components:
            self.ui_components['status_panel'].value = message
