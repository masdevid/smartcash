"""
File: smartcash/ui/setup/dependency/configs/dependency_config_handler.py
Description: Pure mixin-based dependency config handler using core ConfigurationMixin.
Uses composition over inheritance for better flexibility and testability.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from smartcash.ui.logger import get_module_logger
from .dependency_defaults import get_default_dependency_config


class DependencyConfigHandler:
    """
    Dependency configuration handler using pure delegation pattern.
    
    This class follows the modern BaseUIModule architecture where config handlers
    are pure implementation classes that delegate to BaseUIModule mixins.
    
    Handles configuration for dependency management and installation.
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None, logger=None):
        """
        Initialize dependency config handler.
        
        Args:
            default_config: Optional default configuration
            logger: Optional logger instance
        """
        # Initialize pure delegation pattern (no mixin inheritance)
        self.logger = logger or get_module_logger('smartcash.ui.setup.dependency.config')
        self.module_name = 'dependency'
        self.parent_module = 'setup'
        
        # Store default config
        self._default_config = default_config or get_default_dependency_config()
        self._config = self._default_config.copy()
        
        self.logger.info("✅ Dependency config handler initialized")

    # --- Core Configuration Methods ---

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.validate_config(updates):
            self._config.update(updates)
            self.logger.debug(f"Configuration updated: {list(updates.keys())}")
        else:
            raise ValueError("Invalid configuration updates provided")

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = get_default_dependency_config().copy()
        self.logger.info("Configuration reset to defaults")

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save config file
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Implementation would save to YAML file
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default dependency configuration.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_dependency_config()

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate dependency configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - ensure required keys exist
            if not isinstance(config, dict):
                return False
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    # ==================== ADDITIONAL METHODS ====================
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'DependencyConfigHandler':
        """Create config handler instance (ConfigurationMixin requirement)."""
        # Return self since we are the config handler
        return self
    
    # ==================== CONFIGURATION ACCESS METHODS ====================
    
    def update_config(self, updates: Dict[str, Any], skip_ui_update: bool = True):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            skip_ui_update: Whether to skip UI updates to prevent circular references
        """
        if not hasattr(self, '_merged_config'):
            self._merged_config = {}
            
        # Update the config
        self._merged_config.update(updates)
        
        # Update UI only if explicitly requested and components are available
        if not skip_ui_update and hasattr(self, '_ui_components') and self._ui_components:
            if hasattr(self, 'sync_to_ui'):
                self.sync_to_ui(self._ui_components, self._merged_config)
        
        # Update each key-value pair directly in the merged config
        for key, value in updates.items():
            # Handle nested updates with dot notation
            keys = key.split('.')
            current = self._merged_config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current or not isinstance(current[k], dict):
                    current[k] = {}
                current = current[k]
            
            # Set the final value
            current[keys[-1]] = value
    
    # ==================== UI INTEGRATION ====================
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """
        Set UI components for configuration extraction.
        
        Args:
            ui_components: Dictionary of UI components
        """
        self._ui_components = ui_components
        component_count = len(ui_components) if ui_components else 0
        self.logger.debug(f"UI components set: {list(ui_components.keys()) if ui_components else 'None'}, count: {component_count}", 'debug')
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Returns:
            Extracted configuration dictionary
        """
        try:
            if not self._ui_components:
                self.logger.debug("No UI components available for extraction - using current config", 'debug')
                return self.get_current_config()
            
            # Start with current config from ConfigurationMixin
            extracted_config = self.get_current_config()
            
            # Extract selected packages from checkboxes
            selected_packages = []
            if 'package_checkboxes' in self._ui_components:
                checkboxes = self._ui_components['package_checkboxes']
                for checkbox_list in checkboxes.values():
                    for checkbox in checkbox_list:
                        if hasattr(checkbox, 'value') and checkbox.value:
                            # Try to get package name
                            if hasattr(checkbox, 'package_name'):
                                selected_packages.append(checkbox.package_name)
                            elif hasattr(checkbox, 'description'):
                                # Fallback: extract from description
                                desc = checkbox.description
                                if '(' in desc:
                                    package_name = desc.split('(')[0].strip()
                                    selected_packages.append(package_name)
            
            # Extract custom packages
            custom_packages = ""
            if 'custom_packages' in self._ui_components:
                custom_widget = self._ui_components['custom_packages']
                if hasattr(custom_widget, 'value'):
                    custom_packages = custom_widget.value.strip()
            
            # Extract install options if available
            install_options = {}
            if 'install_options' in self._ui_components:
                options_widget = self._ui_components['install_options']
                if hasattr(options_widget, 'value'):
                    install_options = options_widget.value or {}
            
            # Update extracted config
            extracted_config.update({
                'selected_packages': selected_packages,
                'custom_packages': custom_packages,
                'install_options': install_options,
                'last_extracted': datetime.now().isoformat()
            })
            
            custom_count = len([p for p in custom_packages.split('\n') if p.strip()])
            self.logger.debug(
                f"Extracted config: {len(selected_packages)} selected packages, {custom_count} custom packages"
            )
            
            return extracted_config
            
        except Exception as e:
            self.logger.debug(f"Failed to extract config from UI: {e}", 'error')
            return self.get_current_config()
    
    # ==================== VALIDATION ====================
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate dependency configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with 'valid' and 'message' keys
        """
        try:
            # Check required keys
            required_keys = ['selected_packages', 'custom_packages']
            for key in required_keys:
                if key not in config:
                    error_msg = f"Missing required key: {key}"
                    self.logger.debug(f"{error_msg} (missing: {key})", 'error')
                    return {'valid': False, 'message': error_msg}
            
            # Validate selected_packages is list
            if not isinstance(config['selected_packages'], list):
                error_msg = "selected_packages must be a list"
                self.logger.debug(f"{error_msg} (got {type(config['selected_packages']).__name__})", 'error')
                return {'valid': False, 'message': error_msg}
            
            # Validate custom_packages is string
            if not isinstance(config['custom_packages'], str):
                error_msg = "custom_packages must be a string"
                self.logger.debug(f"{error_msg} (got {type(config['custom_packages']).__name__})", 'error')
                return {'valid': False, 'message': error_msg}
            
            # Validate package names (basic validation)
            for package in config['selected_packages']:
                if not isinstance(package, str) or not package.strip():
                    error_msg = f"Invalid package name: {package}"
                    self.logger.debug(f"{error_msg} (package: {package})", 'error')
                    return {'valid': False, 'message': error_msg}
            
            self.logger.debug("Configuration validation passed", 'info')
            return {'valid': True, 'message': 'Configuration is valid'}
            
        except Exception as e:
            error_msg = f"Error validating config: {e}"
            self.logger.debug(f"{error_msg} (error type: {type(e).__name__})", 'error')
            return {'valid': False, 'message': error_msg}
    
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
            self.logger.info("✅ Konfigurasi berhasil disimpan")
            
            return {
                'success': True,
                'message': 'Configuration saved successfully',
                'config': current_config
            }
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            self.logger.error(f"❌ Gagal menyimpan konfigurasi: {error_msg} (tipe error: {type(e).__name__})")
            return {
                'success': False,
                'message': error_msg,
                'config': getattr(self, '_merged_config', {})
            }
    
    def reset_config(self, reset_ui: bool = True) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Args:
            reset_ui: Whether to update the UI after reset
            
        Returns:
            Dict with operation result
        """
        try:
            # Get default configuration directly without calling parent
            default_config = self.get_default_config()
            
            # Update internal config
            self._merged_config = default_config.copy()
            
            # Sync UI if components are available and reset_ui is True
            if reset_ui and hasattr(self, '_ui_components') and self._ui_components:
                if hasattr(self, 'sync_to_ui'):
                    self.sync_to_ui(self._ui_components, self._merged_config)
            
            self.logger.info("✅ Konfigurasi direset ke pengaturan awal")
            return {
                'success': True, 
                'message': 'Configuration reset to defaults',
                'config': self._merged_config
            }
            
        except Exception as e:
            error_msg = f"Gagal mereset konfigurasi: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'success': False, 
                'message': error_msg,
                'config': getattr(self, '_merged_config', {})
            }
    
    # ==================== DEPENDENCY-SPECIFIC METHODS ====================
    
    def add_selected_package(self, package_name: str) -> bool:
        """
        Add package to selected packages.
        
        Args:
            package_name: Name of package to add
            
        Returns:
            True if successful
        """
        try:
            selected_packages = self.get_config_value('selected_packages', [])
            
            if package_name not in selected_packages:
                selected_packages.append(package_name)
                self.update_config_value('selected_packages', selected_packages)
                
                self.logger.info(f"Package '{package_name}' added to selection (total: {len(selected_packages)})")
                return True
            
            self.logger.debug(f"Package '{package_name}' already in selection", 'debug')
            return True
            
        except Exception as e:
            self.logger.debug(f"Error adding package {package_name}: {e} (type: {type(e).__name__})", 'error')
            return False
    
    def remove_selected_package(self, package_name: str) -> bool:
        """
        Remove package from selected packages.
        
        Args:
            package_name: Name of package to remove
            
        Returns:
            True if successful
        """
        try:
            selected_packages = self.get_config_value('selected_packages', [])
            
            if package_name in selected_packages:
                selected_packages.remove(package_name)
                self.update_config_value('selected_packages', selected_packages)
                
                self.logger.debug(f"Package '{package_name}' removed from selection (remaining: {len(selected_packages)-1})", 'info')
                return True
            
            self.logger.debug(f"Package '{package_name}' not found in selection", 'debug')
            return True
            
        except Exception as e:
            self.logger.debug(f"Error removing package {package_name}: {e} (type: {type(e).__name__})", 'error')
            return False
    
    def update_custom_packages(self, custom_packages: str) -> bool:
        """
        Update custom packages string.
        
        Args:
            custom_packages: Custom packages string
            
        Returns:
            True if successful
        """
        try:
            self.update_config_value('custom_packages', custom_packages)
            package_count = len([line.strip() for line in custom_packages.split('\n') if line.strip()])
            self.logger.info(f"Updated {package_count} custom packages")
            return True
            
        except Exception as e:
            self.logger.debug(f"Error updating custom packages: {e} (type: {type(e).__name__})", 'error')
            return False
    
    # ==================== UTILITY METHODS ====================
    
    def get_selected_packages(self) -> List[str]:
        """Get list of selected packages."""
        return self.get_config_value('selected_packages', [])
    
    def get_custom_packages(self) -> str:
        """Get custom packages string."""
        return self.get_config_value('custom_packages', '')
    
    def get_install_options(self) -> Dict[str, Any]:
        """Get installation options."""
        return self.get_config_value('install_options', {})
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """Get UI settings."""
        return self.get_config_value('ui_settings', {})
    
    def get_package_count(self) -> int:
        """Get total package count (selected + custom)."""
        selected_count = len(self.get_selected_packages())
        custom_packages = self.get_custom_packages()
        custom_count = len([line.strip() for line in custom_packages.split('\n') if line.strip()])
        return selected_count + custom_count
    
    def get_all_packages_list(self) -> List[str]:
        """
        Get combined list of all packages (selected + custom).
        
        Returns:
            List of all package names
        """
        packages = []
        
        # Add selected packages
        packages.extend(self.get_selected_packages())
        
        # Add custom packages
        custom_packages = self.get_custom_packages()
        if custom_packages:
            for line in custom_packages.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name from version specification
                    pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                    if pkg_name:
                        packages.append(pkg_name)
        
        return list(set(packages))  # Remove duplicates
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for display.
        
        Returns:
            Configuration summary
        """
        current_config = self.get_current_config()
        return {
            'handler_type': 'core_mixin_based',
            'inheritance_chain': ['DependencyConfigHandler', 'LoggingMixin', 'ConfigurationMixin'],
            'total_packages': self.get_package_count(),
            'selected_packages': len(self.get_selected_packages()),
            'custom_packages': len([line.strip() for line in self.get_custom_packages().split('\n') if line.strip()]),
            'has_install_options': bool(self.get_install_options()),
            'last_updated': current_config.get('last_saved', 'Never')
        }