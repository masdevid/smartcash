"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Preprocessing config handler dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.dataset.preprocessing.handlers.base_preprocessing_handler import BasePreprocessingHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui

class PreprocessingConfigHandler(ConfigHandler, BasePreprocessingHandler):
    """Preprocessing config handler dengan centralized error handling.
    
    Provides configuration management for preprocessing module:
    - Config extraction from UI components
    - UI updates from loaded config
    - Default config generation
    - Config persistence with inheritance support
    """
    
    @handle_ui_errors(error_component_title="Config Handler Initialization Error", log_error=True)
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None, 
                 persistence_enabled: bool = True):
        """Initialize preprocessing config handler.
        
        Args:
            ui_components: Dictionary containing UI components
            persistence_enabled: Whether to enable config persistence
        """
        # Initialize both parent classes
        BasePreprocessingHandler.__init__(self, ui_components=ui_components)
        ConfigHandler.__init__(self, module_name='preprocessing', parent_module='dataset',
                              persistence_enabled=persistence_enabled)
        
        # Store UI components reference
        self.ui_components = ui_components or {}
    
    @handle_ui_errors(error_component_title="Config Extraction Error", log_error=True)
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components.
        
        Args:
            ui_components: Dictionary containing UI components
            
        Returns:
            Extracted configuration dictionary
        """
        try:
            return extract_preprocessing_config(ui_components)
        except Exception as e:
            self.logger.error(f"❌ Error extracting config: {str(e)}")
            return self.get_default_config()
    
    @handle_ui_errors(error_component_title="UI Update Error", log_error=True)
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dari loaded config.
        
        Args:
            ui_components: Dictionary containing UI components
            config: Configuration dictionary to apply
        """
        try:
            update_preprocessing_ui(ui_components, config)
            self.logger.info("🔄 UI updated with config")
        except Exception as e:
            self.logger.error(f"❌ Error updating UI: {str(e)}")
    
    @handle_ui_errors(error_component_title="Default Config Error", log_error=True)
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk preprocessing.
        
        Returns:
            Default configuration dictionary
        """
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading defaults: {str(e)}")
            return {'preprocessing': {'enabled': True}, 'performance': {'batch_size': 32}}
    
    @handle_ui_errors(error_component_title="Config Save Error", log_error=True)
    def save_config(self, ui_components: Optional[Dict[str, Any]] = None) -> bool:
        """Save config dari UI components.
        
        Args:
            ui_components: Optional dictionary containing UI components
                          (uses self.ui_components if None)
            
        Returns:
            True if save successful, False otherwise
        """
        # Use provided UI components or fall back to instance variable
        ui_components = ui_components or self.ui_components
        
        # If persistence is disabled, just update in-memory state
        if not self.persistence_enabled:
            try:
                self._config_state.config = self.extract_config(ui_components)
                self.logger.info("✅ Config saved to memory (persistence disabled)")
                return True
            except Exception as e:
                self.logger.error(f"❌ Error saving config to memory: {str(e)}")
                return False
        
        try:
            # Extract config from UI
            ui_config = self.extract_config(ui_components)
            
            # Save config
            success = self.config_manager.save_config(ui_config, self.config_filename)
            
            if success:
                self.logger.info(f"✅ Config saved to {self.config_filename}")
                self._refresh_ui_after_save(ui_components)
                return True
            else:
                self.logger.error("❌ Failed to save config")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saving config: {str(e)}")
            return False
    
    @handle_ui_errors(error_component_title="Config Reset Error", log_error=True)
    def reset_config(self, ui_components: Optional[Dict[str, Any]] = None) -> bool:
        """Reset config ke defaults.
        
        Args:
            ui_components: Optional dictionary containing UI components
                          (uses self.ui_components if None)
            
        Returns:
            True if reset successful, False otherwise
        """
        # Use provided UI components or fall back to instance variable
        ui_components = ui_components or self.ui_components
        
        # If persistence is disabled, just update in-memory state
        if not self.persistence_enabled:
            try:
                default_config = self.get_default_config()
                self._config_state.config = default_config
                self.update_ui(ui_components, default_config)
                self.logger.info("✅ Config reset to defaults (persistence disabled)")
                return True
            except Exception as e:
                self.logger.error(f"❌ Error resetting config in memory: {str(e)}")
                return False
        
        try:
            # Get default config
            default_config = self.get_default_config()
            
            # Save default config
            success = self.config_manager.save_config(default_config, self.config_filename)
            
            if success:
                self.logger.info("🔄 Config reset to defaults")
                self.update_ui(ui_components, default_config)
                return True
            else:
                self.logger.error("❌ Failed to reset config")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error resetting config: {str(e)}")
            return False
    
    @handle_ui_errors(error_component_title="UI Refresh Error", log_error=True)
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any]) -> None:
        """Refresh UI setelah save.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        try:
            saved_config = self.load_config()
            if saved_config:
                self.update_ui(ui_components, saved_config)
                self.logger.debug("🔄 UI refreshed with saved config")
        except Exception as e:
            self.logger.warning(f"⚠️ Error refreshing UI: {str(e)}")
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components reference.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        self.ui_components = ui_components
        
        # Setup log redirection if log_output is available
        if 'log_output' in ui_components:
            self.setup_log_redirection(ui_components)
