"""
Dependency Configuration Handler for SmartCash UI.

Implements the ConfigHandler pattern to manage dependency configurations with:
- Configuration extraction from UI components
- UI state management based on configuration
- Configuration persistence with inheritance support
- Validation and error handling

Design Pattern:
- Follows the Template Method pattern through ConfigHandler base class
- Implements Strategy pattern for config extraction and UI updates
- Uses Composition for config management (ConfigManager)
- Follows Open/Closed Principle for extensibility
"""

from typing import Dict, Any, List, Optional
import copy
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.common.logger import get_logger, safe_log_to_ui
from smartcash.common.config.manager import get_config_manager

# Import from defaults
from smartcash.ui.setup.dependency.handlers.defaults import PACKAGE_CATEGORIES

# Type aliases
ConfigDict = Dict[str, Any]

# Constants
MODULE_NAME = 'dependency'
PARENT_MODULE = 'setup'
CONFIG_FILENAME = 'dependency_config.yaml'

class DependencyConfigHandler(ConfigHandler):
    """Manages dependency configurations with UI integration and inheritance support.
    
    Handles config lifecycle: load/save (with _base_ inheritance), UI sync, and validation.
    Uses 'dependency_config.yaml' by default.
    
    Example:
        handler = DependencyConfigHandler()
        config = handler.load_config()
        handler.update_ui(ui_components, config)
        handler.save_config(ui_components)  # Saves with extracted config
    """
    
    def __init__(self, module_name: str = MODULE_NAME, 
                 parent_module: str = PARENT_MODULE):
        """Initialize the configuration handler.
        
        Args:
            module_name: Name of the module (default: 'dependency')
            parent_module: Parent module name (default: 'setup')
        """
        super().__init__(module_name, parent_module)
        self._package_categories = PACKAGE_CATEGORIES
        self.config_manager = get_config_manager()
        self.config_filename = CONFIG_FILENAME
        self._ui_components: Dict[str, Any] = {}
        
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Extracted configuration dictionary
        """
        from smartcash.ui.setup.dependency.handlers.config_extractor import extract_dependency_config
        return extract_dependency_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components from configuration.
        
        Args:
            ui_components: Dictionary of UI components to update
            config: Configuration dictionary to apply
        """
        from smartcash.ui.setup.dependency.handlers.config_updater import update_dependency_ui
        update_dependency_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with fallback to minimal config.
        
        Returns:
            Default configuration dictionary
        """
        try:
            # Try to get default config from parent
            config = super().get_default_config()
            
            # Ensure required fields exist
            config.setdefault('version', '1.0.0')
            config.setdefault('module_name', self.module_name)
            config.setdefault('selected_packages', self.get_default_selected_packages())
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error getting default config: {e}")
            return {
                'module_name': self.module_name,
                'version': '1.0.0',
                'selected_packages': self.get_default_selected_packages()
            }
    
    def get_default_selected_packages(self) -> List[str]:
        """Get list of package keys that are selected by default.
        
        Returns:
            List of package keys that should be selected by default
        """
        try:
            selected = []
            
            # Get all packages that are marked as default or required
            for category in self._package_categories:
                for pkg in category.get('packages', []):
                    pkg_key = pkg.get('key')
                    if not pkg_key:
                        continue
                        
                    # Select if marked as default or required
                    if pkg.get('default', False) or pkg.get('required', False):
                        selected.append(pkg_key)
            
            # Remove duplicates while preserving order
            seen = set()
            return [x for x in selected if not (x in seen or seen.add(x))]
            
        except Exception as e:
            self.logger.error(f"Error getting default selected packages: {e}", exc_info=True)
            return ['torch', 'torchvision', 'numpy', 'pandas']
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load config with inheritance handling.
        
        Args:
            config_filename: Optional custom config filename
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            filename = config_filename or self.config_filename
            config = self.config_manager.load_config(filename)
            
            if not config:
                safe_log_to_ui(self._ui_components, "⚠️ Config is empty, using default", "warning")
                return self.get_default_config()
            
            # Handle inheritance
            if '_base_' in config:
                base_config = self.config_manager.load_config(config['_base_']) or {}
                merged_config = self._merge_configs(base_config, config)
                safe_log_to_ui(self._ui_components, f"📂 Config loaded from {filename} with inheritance", "info")
                return merged_config
            
            safe_log_to_ui(self._ui_components, f"📂 Config loaded from {filename}", "info")
            return config
            
        except Exception as e:
            error_msg = f"Error loading config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            safe_log_to_ui(self._ui_components, f"❌ {error_msg}", "error")
            return self.get_default_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save configuration to file.
        
        Args:
            ui_components: Dictionary of UI components
            config_filename: Optional custom config filename
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            filename = config_filename or self.config_filename
            config = self.extract_config(ui_components)
            
            success = self.config_manager.save_config(config, filename)
            
            if success:
                safe_log_to_ui(ui_components, f"✅ Config saved to {filename}", "success")
                self._refresh_ui_after_save(ui_components, filename)
                return True
            else:
                safe_log_to_ui(ui_components, "❌ Failed to save config", "error")
                return False
                
        except Exception as e:
            error_msg = f"Error saving config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            safe_log_to_ui(ui_components, f"❌ {error_msg}", "error")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset config to defaults.
        
        Args:
            ui_components: Dictionary of UI components
            config_filename: Optional custom config filename
            
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                safe_log_to_ui(ui_components, "🔄 Config reset to defaults", "success")
                self.update_ui(ui_components, default_config)
                return True
            else:
                safe_log_to_ui(ui_components, "❌ Failed to reset config", "error")
                return False
                
        except Exception as e:
            error_msg = f"Error resetting config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            safe_log_to_ui(ui_components, f"❌ {error_msg}", "error")
            return False
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations with deep merge.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Configuration with overrides
            
        Returns:
            Merged configuration dictionary
        """
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key == '_base_':
                continue
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge helper for nested dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary with overrides
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str) -> None:
        """Refresh UI after saving configuration.
        
        Args:
            ui_components: Dictionary of UI components
            filename: Name of the saved config file
        """
        try:
            saved_config = self.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                safe_log_to_ui(ui_components, "🔄 UI refreshed with saved config", "info")
        except Exception as e:
            error_msg = f"Error refreshing UI: {str(e)}"
            self.logger.warning(error_msg)
            safe_log_to_ui(ui_components, f"⚠️ {error_msg}", "warning")