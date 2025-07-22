"""
Dataset Downloader UIModule - New Core Pattern
Following new UIModule architecture with clean implementation.
"""
from typing import Dict, Any, Optional, List, Tuple
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
        
        self.log_debug("✅ DownloaderUIModule initialized")
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for Downloader module (BaseUIModule requirement)."""
        try:
            self.log_debug("Creating Downloader UI components...")
            ui_components = create_downloader_ui_components(module_config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.log_debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.log_error(f"Failed to create UI components: {e}")
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
    
    
    def _extract_ui_config(self) -> Dict[str, Any]:
        """Extract configuration from UI components.
        
        Returns:
            Dictionary containing the extracted configuration
        """
        try:
            if not hasattr(self, '_config_handler') or not self._config_handler:
                self.log("❌ Config handler not available for UI config extraction", "warning")
                return {}
                
            # Delegate to config handler's extract_config method
            if hasattr(self._config_handler, 'extract_config'):
                return self._config_handler.extract_config(self._ui_components)
                
            self.log("❌ Config handler does not implement extract_config", "warning")
            return {}
            
        except Exception as e:
            self.log(f"❌ Error extracting UI config: {e}", "error")
            return {}
    
    def _operation_download(self, button=None) -> None:
        """Handle download operation."""
        try:
            # Disable all operation buttons
            self.disable_all_buttons()
            
            self.log("📥 Memulai download dataset...", "info")
            
            # Create download operation with merged config
            ui_config = self._extract_ui_config()
            merged_config = {**self.config, **ui_config}
            
            operation = create_download_operation(self, merged_config)
            result = operation.execute()
            
            if result.get("success", False):
                file_count = result.get("file_count", 0)
                total_size = result.get("total_size", "0B")
                self.log(f"✅ Download selesai: {file_count} file ({total_size})", "info")
            else:
                error_msg = result.get("error", "Download gagal")
                self.log(f"❌ Download gagal: {error_msg}", "error")
                
        except Exception as e:
            self.log(f"❌ Error download: {e}", "error")
        finally:
            # Re-enable buttons
            self.enable_all_buttons()
    
    def _operation_check(self, button=None) -> None:
        """Handle check operation."""
        try:
            # Disable all operation buttons
            self.disable_all_buttons()
            
            self.log("🔍 Memeriksa status dataset...", "info")
            
            # Create check operation
            operation = create_check_operation(self, self.config)
            result = operation.execute()
            
            if result.get("success", False):
                file_count = result.get("file_count", 0)
                total_size = result.get("total_size", "0B")
                self.log(f"✅ Pengecekan selesai: {file_count} file ({total_size})", "info")
            else:
                error_msg = result.get("error", "Pengecekan gagal")
                self.log(f"❌ Pengecekan gagal: {error_msg}", "error")
                
        except Exception as e:
            self.log(f"❌ Error pengecekan: {e}", "error")
        finally:
            # Re-enable buttons
            self.enable_all_buttons()
    
    def _operation_cleanup(self, button=None) -> None:
        """Handle cleanup operation."""
        try:
            # Disable all operation buttons
            self.disable_all_buttons()
            
            self.log("🧹 Memulai pembersihan dataset...", "info")
            
            # Create cleanup operation
            operation = create_cleanup_operation(self, self.config)
            
            # Get cleanup targets first
            targets_result = operation.get_cleanup_targets()
            
            if not targets_result.get("success", False):
                error_msg = targets_result.get("error", "Gagal mendapatkan target pembersihan")
                self.log(f"❌ Pembersihan gagal: {error_msg}", "error")
                return
            
            if targets_result.get("total_files", 0) == 0:
                self.log("ℹ️ Tidak ada file untuk dibersihkan", "info")
                return
                
            # Show confirmation dialog and execute cleanup
            def confirm_cleanup():
                try:
                    result = operation.execute(targets_result.get("targets"))
                    
                    if result.get("success", False):
                        deleted_files = result.get("deleted_files", 0)
                        freed_space = result.get("freed_space", "0B")
                        self.log(f"✅ Pembersihan selesai: {deleted_files} file dihapus ({freed_space})", "info")
                    else:
                        error_msg = result.get("error", "Pembersihan gagal")
                        self.log(f"❌ Pembersihan gagal: {error_msg}", "error")
                except Exception as e:
                    self.log(f"❌ Error pembersihan: {e}", "error")
                finally:
                    # Re-enable buttons
                    self.enable_all_buttons()
            
            # Show confirmation dialog
            operation.show_cleanup_confirmation(targets_result.get("targets"), confirm_cleanup)
            
        except Exception as e:
            self.log(f"❌ Error pembersihan: {e}", "error")
            # Re-enable buttons on error
            self.enable_all_buttons()
    
    def _get_selected_packages(self) -> List[str]:
        """Get list of selected packages from UI."""
        # Implementation here
        return []
        
    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup any downloader-specific resources
            if hasattr(self, '_download_status'):
                self._download_status.clear()
            
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                # Close individual widgets
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Downloader module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Downloader module cleanup failed: {e}")
    
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion
    
    def create_config_handler(self, config: Dict[str, Any]):
        """Create config handler instance for Downloader module (BaseUIModule requirement)."""
        from smartcash.ui.dataset.downloader.configs.downloader_config_handler import DownloaderConfigHandler
        handler = DownloaderConfigHandler(config)
        return handler
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Downloader module (BaseUIModule requirement)."""
        return get_default_downloader_config()
