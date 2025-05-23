"""
File: smartcash/ui/dataset/preprocessing/handlers/logger_handler.py
Deskripsi: Handler untuk logging dengan integrasi UILoggerBridge dan namespace management yang proper
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger, LogLevel
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge, log_to_service

class LoggerHandler:
    """Handler untuk logging dalam modul preprocessing dengan namespace management."""
    
    # Namespace constants
    PREPROCESSING_NAMESPACE = "smartcash.ui.dataset.preprocessing"
    MODULE_SHORT_NAME = "PREPROC"
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi logger handler dengan komponen UI."""
        self.ui_components = ui_components
        self.namespace = self.PREPROCESSING_NAMESPACE
        
        # Setup core logger
        self.core_logger = get_logger(self.namespace)
        
        # Setup UI logger bridge jika UI components tersedia
        self.ui_logger_bridge = None
        if self._has_ui_components():
            self._setup_ui_logger_bridge()
        
        # Set initialization flag
        self.ui_components['preprocessing_initialized'] = True
        self.ui_components['logger_namespace'] = self.namespace
    
    def _has_ui_components(self) -> bool:
        """Cek apakah UI components memiliki output widgets yang diperlukan."""
        return any(key in self.ui_components for key in ['log_output', 'status', 'output'])
    
    def _setup_ui_logger_bridge(self) -> None:
        """Setup UI logger bridge untuk integrasi dengan UI components."""
        try:
            # Buat UI logger bridge
            self.ui_logger_bridge = create_ui_logger_bridge(
                self.ui_components,
                namespace=self.namespace
            )
            
            # Simpan reference ke UI components
            self.ui_components['logger'] = self.ui_logger_bridge
            self.ui_components['ui_logger_bridge'] = self.ui_logger_bridge
            
            self.core_logger.debug(f"🔗 UI Logger Bridge berhasil disetup untuk {self.MODULE_SHORT_NAME}")
            
        except Exception as e:
            self.core_logger.warning(f"⚠️ Error setup UI logger bridge: {str(e)}")
            # Fallback ke service logger
            self.ui_components['logger'] = self.core_logger
    
    def log(self, message: str, level: str = 'info', icon: str = '') -> None:
        """
        Log pesan dengan level dan icon yang ditentukan.
        
        Args:
            message: Pesan yang akan di-log
            level: Level log (debug, info, success, warning, error, critical)
            icon: Icon emoji untuk pesan (opsional)
        """
        # Cek apakah preprocessing sudah diinisialisasi
        if not self._is_preprocessing_initialized():
            return  # Skip logging untuk mencegah muncul di modul lain
        
        # Format pesan dengan namespace
        formatted_message = self._format_message(message, icon)
        
        # Log ke UI jika UI logger bridge tersedia
        if self.ui_logger_bridge and self._has_ui_components():
            self._log_to_ui(formatted_message, level, icon)
        
        # Log ke service logger dengan namespace prefix
        self._log_to_service(formatted_message, level)
    
    def _is_preprocessing_initialized(self) -> bool:
        """Cek apakah preprocessing module sudah diinisialisasi."""
        return self.ui_components.get('preprocessing_initialized', False)
    
    def _format_message(self, message: str, icon: str = '') -> str:
        """Format pesan dengan icon jika ada."""
        if icon and not message.startswith(icon):
            return f"{icon} {message}"
        return message
    
    def _log_to_ui(self, message: str, level: str, icon: str = '') -> None:
        """Log ke UI menggunakan UI logger bridge."""
        try:
            # Map level ke method yang sesuai
            level_methods = {
                'debug': self.ui_logger_bridge.debug,
                'info': self.ui_logger_bridge.info,
                'success': self.ui_logger_bridge.success,
                'warning': self.ui_logger_bridge.warning,
                'error': self.ui_logger_bridge.error,
                'critical': self.ui_logger_bridge.critical
            }
            
            # Dapatkan method yang sesuai
            log_method = level_methods.get(level, self.ui_logger_bridge.info)
            
            # Log ke UI
            log_method(message)
            
        except Exception as e:
            # Fallback ke direct UI logging
            try:
                from smartcash.ui.utils.ui_logger import log_to_ui
                log_to_ui(self.ui_components, message, level, icon)
            except Exception as fallback_error:
                self.core_logger.warning(f"⚠️ Error logging ke UI: {str(e)}, fallback error: {str(fallback_error)}")
    
    def _log_to_service(self, message: str, level: str) -> None:
        """Log ke service logger dengan namespace prefix."""
        try:
            # Tambahkan prefix namespace untuk filtering
            prefixed_message = f"[{self.MODULE_SHORT_NAME}] {message}"
            
            # Log menggunakan service helper
            log_to_service(self.core_logger, prefixed_message, level)
            
        except Exception as e:
            # Direct fallback ke core logger
            self.core_logger.info(f"[{self.MODULE_SHORT_NAME}] {message}")
    
    # Convenience methods untuk different log levels
    def debug(self, message: str, icon: str = '🔍') -> None:
        """Log debug message."""
        self.log(message, 'debug', icon)
    
    def info(self, message: str, icon: str = 'ℹ️') -> None:
        """Log info message."""
        self.log(message, 'info', icon)
    
    def success(self, message: str, icon: str = '✅') -> None:
        """Log success message."""
        self.log(message, 'success', icon)
    
    def warning(self, message: str, icon: str = '⚠️') -> None:
        """Log warning message."""
        self.log(message, 'warning', icon)
    
    def error(self, message: str, icon: str = '❌') -> None:
        """Log error message."""
        self.log(message, 'error', icon)
    
    def critical(self, message: str, icon: str = '🔥') -> None:
        """Log critical message."""
        self.log(message, 'critical', icon)
    
    def progress(self, message: str, percentage: Optional[int] = None, icon: str = '🔄') -> None:
        """Log progress message dengan formatting khusus."""
        if percentage is not None:
            formatted_message = f"{message} ({percentage}%)"
        else:
            formatted_message = message
        self.log(formatted_message, 'info', icon)
    
    def step(self, step_name: str, step_number: int, total_steps: int, icon: str = '📋') -> None:
        """Log step progress dengan formatting khusus."""
        message = f"Step {step_number}/{total_steps}: {step_name}"
        self.log(message, 'info', icon)
    
    def create_progress_callback(self) -> Callable:
        """
        Buat callback function untuk progress tracking yang kompatibel dengan backend services.
        
        Returns:
            Fungsi callback yang bisa digunakan oleh backend services
        """
        def progress_callback(**kwargs):
            try:
                # Extract parameters
                progress = kwargs.get('progress', 0) 
                total = kwargs.get('total', 100)
                message = kwargs.get('message', '')
                step = kwargs.get('step', '')
                
                # Format progress message
                if total > 0:
                    percentage = int((progress / total) * 100)
                    if message:
                        self.progress(message, percentage)
                    elif step:
                        self.step(step, progress, total)
                    else:
                        self.progress(f"Progress: {progress}/{total}", percentage)
                else:
                    if message:
                        self.info(message)
                
                # Cek apakah proses harus dihentikan
                if self.ui_components.get('stop_requested', False):
                    self.warning("Proses dihentikan oleh pengguna", "⏹️")
                    return False
                
                return True
                
            except Exception as e:
                self.error(f"Error saat progress callback: {str(e)}")
                return True  # Continue despite error
        
        return progress_callback
    
    def create_status_callback(self) -> Callable:
        """
        Buat callback untuk status updates dari backend services.
        
        Returns:
            Fungsi callback untuk status updates
        """
        def status_callback(status: str, message: str = '', **kwargs):
            try:
                # Map status ke log level
                status_map = {
                    'idle': 'info',
                    'running': 'info', 
                    'success': 'success',
                    'error': 'error',
                    'warning': 'warning',
                    'complete': 'success'
                }
                
                log_level = status_map.get(status.lower(), 'info')
                
                # Format message
                if message:
                    formatted_message = message
                else:
                    formatted_message = f"Status: {status}"
                
                # Log dengan level yang sesuai
                self.log(formatted_message, log_level)
                
                # Update status panel jika tersedia
                self._update_status_panel(status, formatted_message)
                
            except Exception as e:
                self.error(f"Error saat status callback: {str(e)}")
        
        return status_callback
    
    def _update_status_panel(self, status: str, message: str) -> None:
        """Update status panel jika tersedia."""
        try:
            if 'status_panel' in self.ui_components:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status)
        except Exception as e:
            self.core_logger.debug(f"Status panel update gagal: {str(e)}")
    
    def set_log_level(self, level: str) -> None:
        """
        Set minimum log level.
        
        Args:
            level: Level minimum (debug, info, warning, error, critical)
        """
        try:
            # Map string ke LogLevel enum
            level_map = {
                'debug': LogLevel.DEBUG,
                'info': LogLevel.INFO,
                'warning': LogLevel.WARNING,
                'error': LogLevel.ERROR,
                'critical': LogLevel.CRITICAL
            }
            
            log_level = level_map.get(level.lower(), LogLevel.INFO)
            
            # Set level pada core logger
            self.core_logger.set_level(log_level)
            
            self.debug(f"Log level diset ke: {level}")
            
        except Exception as e:
            self.warning(f"Error set log level: {str(e)}")
    
    def add_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Tambahkan callback untuk setiap log message.
        
        Args:
            callback: Function yang menerima (level, message)
        """
        try:
            # Wrapper untuk callback agar kompatibel dengan LogLevel enum
            def wrapped_callback(log_level: LogLevel, message: str):
                level_str = log_level.name.lower()
                callback(level_str, message)
            
            # Tambahkan ke core logger
            self.core_logger.add_callback(wrapped_callback)
            
            self.debug("Log callback berhasil ditambahkan")
            
        except Exception as e:
            self.warning(f"Error menambahkan log callback: {str(e)}")
    
    def remove_callback(self, callback: Callable) -> None:
        """
        Hapus callback dari logger.
        
        Args:
            callback: Function callback yang akan dihapus
        """
        try:
            self.core_logger.remove_callback(callback)
            self.debug("Log callback berhasil dihapus")
        except Exception as e:
            self.warning(f"Error menghapus log callback: {str(e)}")
    
    def get_namespace_info(self) -> Dict[str, str]:
        """
        Dapatkan informasi namespace logger.
        
        Returns:
            Dictionary informasi namespace
        """
        return {
            'namespace': self.namespace,
            'short_name': self.MODULE_SHORT_NAME,
            'initialized': str(self._is_preprocessing_initialized()),
            'ui_bridge_available': str(self.ui_logger_bridge is not None),
            'ui_components_available': str(self._has_ui_components())
        }
    
    def cleanup_resources(self) -> None:
        """Cleanup resources saat logger handler tidak digunakan lagi."""
        try:
            # Cleanup UI logger bridge
            if self.ui_logger_bridge and hasattr(self.ui_logger_bridge, 'cleanup'):
                self.ui_logger_bridge.cleanup()
            
            self.core_logger.debug(f"🧹 Logger handler resources berhasil dibersihkan")
            
        except Exception as e:
            self.core_logger.warning(f"⚠️ Error saat cleanup logger handler: {str(e)}")

# Factory function untuk membuat logger handler
def create_logger_handler(ui_components: Dict[str, Any]) -> LoggerHandler:
    """
    Factory function untuk membuat logger handler.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance LoggerHandler yang siap digunakan
    """
    return LoggerHandler(ui_components)

# Helper function untuk setup logger pada UI components
def setup_preprocessing_logger(ui_components: Dict[str, Any]) -> LoggerHandler:
    """
    Setup logger handler untuk preprocessing dan simpan ke UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance LoggerHandler yang sudah disetup
    """
    # Buat logger handler
    logger_handler = create_logger_handler(ui_components)
    
    # Simpan ke UI components
    ui_components['logger_handler'] = logger_handler
    ui_components['log_message'] = logger_handler.log  # Shortcut function
    
    # Setup shortcuts untuk convenience methods
    ui_components['log_debug'] = logger_handler.debug
    ui_components['log_info'] = logger_handler.info
    ui_components['log_success'] = logger_handler.success
    ui_components['log_warning'] = logger_handler.warning
    ui_components['log_error'] = logger_handler.error
    ui_components['log_progress'] = logger_handler.progress
    ui_components['log_step'] = logger_handler.step
    
    return logger_handler

# Compatibility function untuk mengganti logger_helper.log_message
def log_message(ui_components: Dict[str, Any], message: str, level: str = 'info', icon: str = '') -> None:
    """
    Compatibility function untuk log_message dari logger_helper.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan di-log
        level: Level log
        icon: Icon emoji
    """
    # Cek apakah logger handler sudah ada
    if 'logger_handler' in ui_components:
        logger_handler = ui_components['logger_handler']
        logger_handler.log(message, level, icon)
    else:
        # Fallback: buat logger handler on-the-fly
        logger_handler = create_logger_handler(ui_components)
        logger_handler.log(message, level, icon)

# Compatibility function untuk is_initialized dari logger_helper  
def is_initialized(ui_components: Dict[str, Any]) -> bool:
    """
    Compatibility function untuk is_initialized dari logger_helper.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika preprocessing sudah diinisialisasi
    """
    return ui_components.get('preprocessing_initialized', False)