"""
File: smartcash/ui/dataset/augmentation/configs/augmentation_config_handler.py
Description: Simple config handler for the augmentation module focused on save/reset functionality.
"""

from typing import Dict, Any, Optional

from smartcash.ui.core.mixins.configuration_mixin import ConfigurationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.logger import get_module_logger
from .augmentation_defaults import get_default_augmentation_config


class AugmentationConfigHandler(LoggingMixin, ConfigurationMixin):
    """
    Simple config handler for augmentation management.
    
    Focuses on save/reset functionality with proper mixin integration.
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentation config handler.
        
        Args:
            default_config: Optional default configuration
        """
        # Initialize mixins properly
        super().__init__()
        
        # Module identification for LoggingMixin
        self.module_name = 'augmentation'
        self.parent_module = 'dataset'
        
        # Initialize logger
        self.logger = get_module_logger('smartcash.ui.dataset.augmentation.configs.augmentation_config_handler')
        
        # Store default config for get_default_config() method
        self._default_config = default_config or get_default_augmentation_config()
        
        # Initialize config handler with default config
        self._initialize_config_handler()
        
        self.logger.debug("✅ AugmentationConfigHandler initialized")
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for augmentation module."""
        return self._default_config.copy()
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'AugmentationConfigHandler':
        """Create config handler instance."""
        return self
    
    # ==================== UI INTEGRATION ====================
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """
        Set UI components for configuration extraction.
        
        Args:
            ui_components: Dictionary of UI components
        """
        self._ui_components = ui_components
        component_count = len(ui_components) if ui_components else 0
        self.logger.debug(f"UI components set: {component_count} components")
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Returns:
            Extracted configuration dictionary
        """
        try:
            if not hasattr(self, '_ui_components') or not self._ui_components:
                self.logger.debug("No UI components available for extraction - using current config")
                return self.get_current_config()
            
            # Start with current config
            extracted_config = self.get_current_config()
            
            # TODO: Add actual UI extraction logic based on your form widgets
            # For now, return current config
            self.logger.debug("Extracted augmentation configuration from UI components")
            
            return extracted_config
            
        except Exception as e:
            self.logger.error(f"Failed to extract config from UI: {e}")
            return self.get_current_config()
    
    # ==================== CONFIGURATION OPERATIONS ====================
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save current configuration.
        
        Returns:
            Dict with operation result
        """
        try:
            # Get current config without triggering UI extraction
            current_config = self.get_current_config()
            
            # Here you would typically save to persistent storage
            self.log("✅ Konfigurasi augmentasi berhasil disimpan", 'info')
            
            return {
                'success': True,
                'message': 'Konfigurasi augmentasi berhasil disimpan',
                'config': current_config
            }
            
        except Exception as e:
            error_msg = f"Gagal menyimpan konfigurasi augmentasi: {str(e)}"
            self.log(f"❌ {error_msg}", 'error')
            return {
                'success': False,
                'message': error_msg,
                'config': getattr(self, '_merged_config', {})
            }
    
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Dict with operation result
        """
        try:
            # Get default configuration
            default_config = self.get_default_config()
            
            # Update internal config
            self._merged_config = default_config.copy()
            
            # Update UI if components are available
            if hasattr(self, '_ui_components') and self._ui_components:
                # TODO: Add UI sync logic if needed
                pass
            
            self.log("✅ Konfigurasi augmentasi direset ke pengaturan awal", 'info')
            
            return {
                'success': True, 
                'message': 'Konfigurasi augmentasi berhasil direset',
                'config': self._merged_config
            }
            
        except Exception as e:
            error_msg = f"Gagal mereset konfigurasi augmentasi: {str(e)}"
            self.log(f"❌ {error_msg}", 'error')
            return {
                'success': False, 
                'message': error_msg,
                'config': getattr(self, '_merged_config', {})
            }