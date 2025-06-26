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
        
        # Ensure config directory exists and initialize default config
        self._ensure_config_directory()
        self._initialize_default_config()
    
    def _ensure_config_directory(self) -> None:
        """Ensure the config directory exists."""
        try:
            config_dir = self.config_manager.get_config_path().parent
            config_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Config directory: {config_dir}")
        except Exception as e:
            error_msg = f"❌ Failed to create config directory: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            safe_log_to_ui(self._ui_components, error_msg, "error")
    
    def _initialize_default_config(self) -> None:
        """Initialize default config if it doesn't exist or is invalid."""
        try:
            config_path = self.config_manager.get_config_path(self.config_filename)
            needs_update = False
            
            # Get default config with all required fields
            default_config = self.get_default_config()
            
            if not config_path.exists():
                needs_update = True
                self.logger.info(f"ℹ️ Config file not found at {config_path}, creating default...")
            else:
                # Check if existing config has all required fields
                try:
                    existing_config = self.config_manager.load_config(self.config_filename) or {}
                    required_fields = ['dependencies', 'install_options']
                    missing_fields = [field for field in required_fields if field not in existing_config]
                    
                    if missing_fields:
                        self.logger.warning(f"⚠️ Missing required fields in config: {missing_fields}")
                        # Merge existing config with defaults, preserving existing values
                        merged_config = self._merge_configs(default_config, existing_config)
                        default_config = merged_config
                        needs_update = True
                        
                except Exception as load_error:
                    self.logger.error(f"❌ Error loading existing config: {str(load_error)}", exc_info=True)
                    needs_update = True
            
            if needs_update:
                # Ensure the directory exists before saving
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the config
                success = self.config_manager.save_config(default_config, self.config_filename)
                if success:
                    self.logger.info(f"✅ {'Created' if not config_path.exists() else 'Updated'} config at {config_path}")
                    # Verify the config was saved correctly
                    saved_config = self.config_manager.load_config(self.config_filename)
                    if not saved_config:
                        self.logger.error("❌ Failed to verify saved config - file may be empty")
                else:
                    self.logger.error(f"❌ Failed to save config to {config_path}")
                    # Try one more time with direct file write as fallback
                    self._save_config_fallback(config_path, default_config)
                    
        except Exception as e:
            self.logger.error(f"❌ Critical error in _initialize_default_config: {str(e)}", exc_info=True)
            # Try fallback save even if there was an error
            try:
                self._save_config_fallback(self.config_manager.get_config_path(self.config_filename), default_config)
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback save also failed: {str(fallback_error)}")
    
    def _save_config_fallback(self, config_path: str, config: Dict[str, Any]) -> bool:
        """Fallback method to save config using direct file operations."""
        try:
            import yaml
            import json
            from pathlib import Path
            
            # Ensure directory exists
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Try YAML first, fall back to JSON if that fails
            try:
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            except Exception as yaml_error:
                self.logger.warning(f"YAML save failed, trying JSON: {str(yaml_error)}")
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            self.logger.info(f"✅ Used fallback method to save config to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Fallback save failed for {config_path}: {str(e)}")
            return False
        
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
            Default configuration dictionary with required fields
        """
        try:
            # Get default config from parent
            config = super().get_default_config()
            
            # Ensure required fields exist with proper structure
            config.update({
                'version': '1.0.0',
                'module_name': self.module_name,
                'dependencies': self._get_default_dependencies(),
                'install_options': {
                    'run_analysis_on_startup': True,
                    'auto_install': False,
                    'upgrade_strategy': 'if_needed',
                    'timeout': 300,  # 5 minutes
                    'retries': 3
                },
                'selected_packages': self.get_default_selected_packages(),
                'ui': {
                    'show_advanced': False,
                    'theme': 'light'
                }
            })
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error getting default config: {e}", exc_info=True)
            # Fallback to minimal valid config
            return {
                'module_name': self.module_name,
                'version': '1.0.0',
                'dependencies': {},
                'install_options': {
                    'run_analysis_on_startup': True,
                    'auto_install': False,
                    'upgrade_strategy': 'if_needed'
                },
                'selected_packages': self.get_default_selected_packages()
            }
            
    def _get_default_dependencies(self) -> Dict[str, Any]:
        """Get default dependencies configuration.
        
        Returns:
            Dictionary of default package dependencies
        """
        try:
            from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependencies
            return get_default_dependencies()
        except Exception as e:
            self.logger.error(f"Error getting default dependencies: {e}", exc_info=True)
            return {}
    
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
        """Load config with inheritance handling and required fields validation.
        
        Args:
            config_filename: Optional custom config filename
            
        Returns:
            Loaded configuration dictionary with all required fields
        """
        try:
            # Get the full config path
            filename = config_filename or self.config_filename
            config_path = self.config_manager.get_config_path(filename)
            
            # Log where we're looking for the config
            self.logger.info(f"🔍 Looking for config at: {config_path}")
            
            # Load the config
            config = self.config_manager.load_config(filename) or {}
            
            # Check for required fields
            required_fields = ['dependencies', 'install_options']
            missing_fields = [field for field in required_fields if field not in config]
            
            if not config or missing_fields:
                # Get default config to ensure we have all required fields
                default_config = self.get_default_config()
                
                if not config:
                    msg = f"⚠️ Config is empty at {config_path}, using default"
                    safe_log_to_ui(self._ui_components, msg, "warning")
                    self.logger.warning(msg)
                    return default_config
                else:
                    # Merge with defaults to ensure all required fields exist
                    msg = f"⚠️ Config missing required fields {missing_fields}, merging with defaults"
                    safe_log_to_ui(self._ui_components, msg, "warning")
                    self.logger.warning(msg)
                    
                    # Only merge missing fields, preserving existing values
                    for field in required_fields:
                        if field not in config and field in default_config:
                            config[field] = default_config[field]
                    
                    # Save the fixed config for next time
                    try:
                        self.config_manager.save_config(config, filename)
                    except Exception as save_error:
                        self.logger.error(f"Failed to save fixed config: {str(save_error)}")
            
            # Handle inheritance after ensuring required fields exist
            if '_base_' in config:
                try:
                    base_config = self.config_manager.load_config(config['_base_']) or {}
                    # Ensure base config has required fields
                    for field in required_fields:
                        if field not in base_config and field in self.get_default_config():
                            base_config[field] = self.get_default_config()[field]
                    
                    merged_config = self._merge_configs(base_config, config)
                    msg = f"📂 Config loaded from {config_path} with inheritance"
                    safe_log_to_ui(self._ui_components, msg, "info")
                    self.logger.info(msg)
                    return merged_config
                except Exception as inherit_error:
                    self.logger.error(f"Error processing inherited config: {str(inherit_error)}", exc_info=True)
                    # Continue with current config if inheritance fails
            
            msg = f"📂 Config loaded from {config_path}"
            safe_log_to_ui(self._ui_components, msg, "info")
            self.logger.info(msg)
            return config
            
        except Exception as e:
            error_msg = f"Error loading config from {getattr(config_path, 'as_posix', lambda: str(config_path))()}: {str(e)}"
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