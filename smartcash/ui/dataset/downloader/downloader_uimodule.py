"""
Dataset Downloader UIModule - New Core Pattern
Following new UIModule architecture with clean implementation.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime

from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs
from smartcash.ui.dataset.downloader.components.downloader_ui import create_downloader_ui_components
from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config
from smartcash.ui.dataset.downloader.operations import (
    create_download_operation,
    create_check_operation,
    create_cleanup_operation
)

class DownloaderUIModule(BaseUIModule):
    """
    Dataset Downloader UIModule following BaseUIModule pattern.
    
    Features:
    - 📥 Dataset download from Roboflow
    - 🔍 Dataset validation and checking
    - 🧹 Dataset cleanup operations
    - 📊 Real-time progress tracking
    - 🇮🇩 Bahasa Indonesia interface
    """
    
    def __init__(self):
        """Initialize downloader module."""
        super().__init__(
            module_name='downloader',
            parent_module='dataset',
            enable_environment=True  # Enable environment management features
        )
        
        # Set required components for validation
        self._required_components = [
            'main_container',
            'header_container',
            'form_container', 
            'action_container',
            'operation_container'
        ]
        
        self.logger.debug("✅ DownloaderUIModule initialized")
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Downloader module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = BaseUIModule.initialize(self)
            
            if success:
                # Set UI components in config handler for extraction
                if self._config_handler and hasattr(self._config_handler, 'set_ui_components'):
                    self._config_handler.set_ui_components(self._ui_components)
                
                # Post-initialization logging (now that operation container is ready)
                self._log_initialization_complete()
                
                # Set global instance for convenience access
                global _downloader_module_instance
                _downloader_module_instance = self
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Downloader module: {e}")
            return False
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for Downloader module (BaseUIModule requirement)."""
        try:
            self.logger.debug("Creating Downloader UI components...")
            ui_components = create_downloader_ui_components(module_config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
        
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Downloader module-specific button handlers."""
        # Call parent method to get base handlers (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add Downloader-specific handlers
        downloader_handlers = {
            'download': self._operation_download,
            'check': self._operation_check,
            'cleanup': self._operation_cleanup
        }
        
        handlers.update(downloader_handlers)
        return handlers
    
    def _log_initialization_complete(self) -> None:
        """Log initialization completion to operation container (after it's ready)."""
        try:
            # Ensure components are ready before proceeding
            if not self.ensure_components_ready():
                self.log("⚠️ Beberapa komponen UI belum siap, beberapa fitur mungkin terbatas", 'warning')
            
            # Log environment info if environment support is enabled
            if self.has_environment_support:
                env_type = "Google Colab" if self.is_colab else "Lokal/Jupyter"
                self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
                
                # Safely access environment_paths attributes
                if hasattr(self, 'environment_paths') and self.environment_paths is not None:
                    if hasattr(self.environment_paths, 'data_root') and self.environment_paths.data_root:
                        self.log(f"📁 Direktori kerja: {self.environment_paths.data_root}", 'info')
                    else:
                        self.log("ℹ️ Direktori kerja default akan digunakan", 'info')
            
            # Update status panel
            self.log("📊 Status: Siap untuk download dataset", 'info')
            
        except Exception as e:
            # Use logger fallback if operation container logging fails
            self.logger.error(f"Gagal mencatat inisialisasi selesai: {e}", exc_info=True)
            self.log(f"⚠️ Terjadi kesalahan saat inisialisasi: {str(e)}", 'error')
    
    
    def _operation_download(self, button=None) -> None:
        """Handle download operation."""
        try:
            # Disable all operation buttons
            self.disable_all_buttons()
            
            self.update_operation_status("Memulai download dataset...", "info")
            self.log("📥 Memulai operasi download dataset", "info")
            
            # Create download operation with merged config
            ui_config = self._extract_ui_config()
            merged_config = {**self.config, **ui_config}
            
            operation = create_download_operation(self, merged_config)
            result = operation.execute()
            
            if result.get("success", False):
                file_count = result.get("file_count", 0)
                total_size = result.get("total_size", "0B")
                self.update_operation_status(f"Download selesai: {file_count} file ({total_size})", "success")
                self.log("✅ Download dataset berhasil diselesaikan", "success")
            else:
                error_msg = result.get("error", "Download gagal")
                self.update_operation_status(f"Download gagal: {error_msg}", "error")
                self.log(f"❌ Download gagal: {error_msg}", "error")
                
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            self.update_operation_status(f"Error download: {e}", "error")
            self.log(f"❌ Error download: {e}", "error")
        finally:
            # Re-enable buttons
            self.enable_all_buttons()
    
    def _operation_check(self, button=None) -> None:
        """Handle check operation."""
        try:
            # Disable all operation buttons
            self.disable_all_buttons()
            
            self.update_operation_status("Memeriksa status dataset...", "info")
            self.log("🔍 Memulai operasi pengecekan dataset", "info")
            
            # Create check operation
            operation = create_check_operation(self, self.config)
            result = operation.execute()
            
            if result.get("success", False):
                file_count = result.get("file_count", 0)
                total_size = result.get("total_size", "0B")
                self.update_operation_status(f"Pengecekan selesai: {file_count} file ({total_size})", "success")
                self.log("✅ Pengecekan dataset berhasil diselesaikan", "success")
            else:
                error_msg = result.get("error", "Pengecekan gagal")
                self.update_operation_status(f"Pengecekan gagal: {error_msg}", "error")
                self.log(f"❌ Pengecekan gagal: {error_msg}", "error")
                
        except Exception as e:
            self.logger.error(f"Check failed: {e}")
            self.update_operation_status(f"Error pengecekan: {e}", "error")
            self.log(f"❌ Error pengecekan: {e}", "error")
        finally:
            # Re-enable buttons
            self.enable_all_buttons()
    
    def _operation_cleanup(self, button=None) -> None:
        """Handle cleanup operation."""
        try:
            # Disable all operation buttons
            self.disable_all_buttons()
            
            self.update_operation_status("Memulai pembersihan dataset...", "info")
            self.log("🧹 Memulai operasi pembersihan dataset", "info")
            
            # Create cleanup operation
            operation = create_cleanup_operation(self, self.config)
            
            # Get cleanup targets first
            targets_result = operation.get_cleanup_targets()
            
            if not targets_result.get("success", False):
                error_msg = targets_result.get("error", "Gagal mendapatkan target pembersihan")
                self.update_operation_status(f"Pembersihan gagal: {error_msg}", "error")
                self.log(f"❌ Pembersihan gagal: {error_msg}", "error")
                return
            
            if targets_result.get("total_files", 0) == 0:
                self.update_operation_status("Tidak ada file untuk dibersihkan", "info")
                self.log("✅ Tidak ada file untuk dibersihkan", "info")
                return
                
            # Show confirmation dialog and execute cleanup
            def confirm_cleanup():
                try:
                    result = operation.execute(targets_result.get("targets"))
                    
                    if result.get("success", False):
                        deleted_files = result.get("deleted_files", 0)
                        freed_space = result.get("freed_space", "0B")
                        self.update_operation_status(f"Pembersihan selesai: {deleted_files} file dihapus ({freed_space})", "success")
                        self.log("✅ Pembersihan berhasil diselesaikan", "success")
                    else:
                        error_msg = result.get("error", "Pembersihan gagal")
                        self.update_operation_status(f"Pembersihan gagal: {error_msg}", "error")
                        self.log(f"❌ Pembersihan gagal: {error_msg}", "error")
                except Exception as e:
                    self.logger.error(f"Cleanup execution failed: {e}")
                    self.update_operation_status(f"Error pembersihan: {e}", "error")
                    self.log(f"❌ Error pembersihan: {e}", "error")
                finally:
                    # Re-enable buttons
                    self.enable_all_buttons()
            
            # Show confirmation dialog
            operation.show_cleanup_confirmation(targets_result.get("targets"), confirm_cleanup)
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            self.update_operation_status(f"Error pembersihan: {e}", "error")
            self.log(f"❌ Error pembersihan: {e}", "error")
            # Re-enable buttons on error
            self.enable_all_buttons()
    
    def create_config_handler(self, config: Dict[str, Any]):
        """Create config handler instance for Downloader module (BaseUIModule requirement)."""
        from smartcash.ui.dataset.downloader.configs.downloader_config_handler import DownloaderConfigHandler
        handler = DownloaderConfigHandler(config)
        return handler
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Downloader module (BaseUIModule requirement)."""
        return get_default_downloader_config()

# Global module instance for singleton pattern
_downloader_module_instance: Optional[DownloaderUIModule] = None

def create_downloader_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> DownloaderUIModule:
    """
    Create a new Downloader UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        DownloaderUIModule instance
    """
    module = DownloaderUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module

def get_downloader_uimodule() -> Optional[DownloaderUIModule]:
    """Get the current Downloader UIModule instance."""
    global _downloader_module_instance
    return _downloader_module_instance

def reset_downloader_uimodule() -> None:
    """Reset the global Downloader UIModule instance."""
    global _downloader_module_instance
    if _downloader_module_instance:
        try:
            _downloader_module_instance.cleanup()
        except:
            pass
    _downloader_module_instance = None


# ==================== FACTORY FUNCTIONS ====================

# Create standardized display function using enhanced factory
from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory

# Create the initialize function using enhanced factory pattern
initialize_downloader_ui = EnhancedUIModuleFactory.create_display_function(DownloaderUIModule)

# ==================== CONVENIENCE FUNCTIONS ====================