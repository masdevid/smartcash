# File: smartcash/ui/pretrained/handlers/config_handler.py
"""
File: smartcash/ui/pretrained/handlers/config_handler.py
Deskripsi: Config handler untuk pretrained models dengan konsistensi pattern dan DRY utilities
"""

from typing import Dict, Any, Optional, Type, cast, Union
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.pretrained.handlers.config_extractor import extract_pretrained_config
from smartcash.ui.pretrained.handlers.config_updater import update_pretrained_ui
from smartcash.common.config import ConfigManager, get_config_manager

# Type alias for logger bridge to support different logger implementations
LoggerBridge = Any

class PretrainedConfigHandler(ConfigHandler):
    """Pretrained config handler dengan consistent pattern dan error handling"""
    
    def __init__(self, module_name: str = 'pretrained', parent_module: str = 'ui'):
        """Initialize pretrained config handler
        
        Args:
            module_name: Nama modul (default: 'pretrained')
            parent_module: Nama modul induk (default: 'ui')
        """
        super().__init__(module_name, parent_module)
        self.config_manager: ConfigManager = get_config_manager()
        self.config_filename: str = 'pretrained_config.yaml'
        self._ui_components: Optional[Dict[str, Any]] = None
        self._logger_bridge: Optional[LoggerBridge] = None
    
    @property
    def logger_bridge(self) -> LoggerBridge:
        """Get logger bridge dari UI components atau fallback ke parent logger
        
        Returns:
            LoggerBridge: Instance logger bridge yang tersedia
            
        Raises:
            RuntimeError: Jika logger bridge tidak tersedia
        """
        if self._logger_bridge is not None:
            return self._logger_bridge
            
        if self._ui_components and 'logger_bridge' in self._ui_components:
            self._logger_bridge = self._ui_components['logger_bridge']
            
        if self._logger_bridge is None and hasattr(self, 'logger'):
            # Fallback to parent logger if available
            self._logger_bridge = cast(LoggerBridge, self.logger)
            
        if self._logger_bridge is None:
            raise RuntimeError("Logger bridge not available")
            
        return self._logger_bridge
        
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components and update logger bridge reference
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Raises:
            ValueError: Jika ui_components bukan dictionary
        """
        if not isinstance(ui_components, dict):
            raise ValueError("UI components must be a dictionary")
            
        self._ui_components = ui_components
        if 'logger_bridge' in ui_components:
            self._logger_bridge = cast(LoggerBridge, ui_components['logger_bridge'])
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message using logger_bridge
        
        Args:
            message: Pesan log
            **kwargs: Argumen tambahan untuk logger
        """
        try:
            self.logger_bridge.debug(message, **kwargs)
        except Exception as e:
            # Fallback to print if logging fails
            print(f"[DEBUG] {message}")
    
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message using logger_bridge
        
        Args:
            message: Pesan log
            **kwargs: Argumen tambahan untuk logger
        """
        try:
            self.logger_bridge.info(message, **kwargs)
        except Exception as e:
            print(f"[INFO] {message}")
    
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message using logger_bridge
        
        Args:
            message: Pesan peringatan
            **kwargs: Argumen tambahan untuk logger
        """
        try:
            self.logger_bridge.warning(message, **kwargs)
        except Exception as e:
            print(f"[WARNING] {message}")
    
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message using logger_bridge
        
        Args:
            message: Pesan error
            exc_info: Apakah menyertakan info exception
            **kwargs: Argumen tambahan untuk logger
        """
        try:
            self.logger_bridge.error(message, exc_info=exc_info, **kwargs)
        except Exception as e:
            print(f"[ERROR] {message}")
    
    def _log_operation_success(self, message: str) -> None:
        """Log successful operation with consistent formatting
        
        Args:
            message: Pesan keberhasilan operasi
        """
        self._log_info(f"✅ {message}")
    
    def _log_operation_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Log operation error with consistent formatting
        
        Args:
            message: Pesan error
            exc: Exception yang terkait (opsional)
        """
        self._log_error(f"❌ {message}", exc_info=exc is not None)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI dengan enhanced error handling
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Returns:
            Dictionary berisi konfigurasi yang diekstrak
            
        Raises:
            ValueError: Jika ui_components tidak valid
            Exception: Jika terjadi kesalahan saat ekstraksi
        """
        self._log_debug("Memulai ekstraksi konfigurasi dari UI")
        
        if not ui_components or not isinstance(ui_components, dict):
            error_msg = "UI components tidak valid"
            self._log_operation_error(error_msg)
            raise ValueError(error_msg)
            
        try:
            config = extract_pretrained_config(ui_components)
            self._log_operation_success("Konfigurasi berhasil diekstrak")
            return config
            
        except Exception as e:
            error_msg = f"Gagal mengekstrak konfigurasi: {str(e)}"
            self._log_operation_error(error_msg, e)
            raise  # Re-raise to parent CommonInitializer error handler
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan config dan error handling
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Dictionary konfigurasi
            
        Raises:
            ValueError: Jika parameter tidak valid
            Exception: Jika terjadi kesalahan saat update UI
        """
        self._log_debug("Memperbarui UI dengan konfigurasi")
        
        if not ui_components or not isinstance(ui_components, dict):
            error_msg = "UI components tidak valid"
            self._log_operation_error(error_msg)
            raise ValueError(error_msg)
            
        if not config or not isinstance(config, dict):
            error_msg = "Konfigurasi tidak valid"
            self._log_operation_error(error_msg)
            raise ValueError(error_msg)
            
        try:
            update_pretrained_ui(ui_components, config)
            self._log_operation_success("UI berhasil diperbarui dengan konfigurasi")
            
        except Exception as e:
            error_msg = f"Gagal memperbarui UI: {str(e)}"
            self._log_operation_error(error_msg, e)
            raise  # Re-raise to parent CommonInitializer error handler
    

    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk pretrained models
        
        Returns:
            Dictionary berisi konfigurasi default
            
        Raises:
            ImportError: Jika modul defaults tidak ditemukan
            Exception: Untuk error lainnya
        """
        self._log_debug("Mengambil konfigurasi default")
        
        try:
            from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
            config = get_default_pretrained_config()
            self._log_debug("Konfigurasi default berhasil diambil")
            return config
            
        except ImportError as e:
            error_msg = f"Modul defaults tidak ditemukan: {str(e)}"
            self._log_operation_error(error_msg, e)
            raise
            
        except Exception as e:
            error_msg = f"Gagal mengambil konfigurasi default: {str(e)}"
            self._log_operation_error(error_msg, e)
            raise  # Re-raise to parent CommonInitializer error handler