"""
File: smartcash/ui/setup/dependency/handlers/config_handler.py

Dependency Configuration Handler for SmartCash UI.

This module provides a configuration handler for the dependency management
interface, handling extraction, validation, and application of configuration.
"""

import os
from typing import Dict, Any, Optional, TypeVar, List, Tuple
from dataclasses import dataclass, field
from smartcash.ui.handlers.config_handlers import BaseConfigHandler
from smartcash.common.logger import get_logger, safe_log_to_ui

# Import from defaults to access package categories
from .defaults import (
    PACKAGE_CATEGORIES,
    CONFIG_VERSION,
    CONFIG_SCHEMA_VERSION,
    PackageConfig,
    PackageCategory,
    DEFAULT_CONFIG
)

# Type aliases
ConfigDict = Dict[str, Any]
UIComponents = Dict[str, Any]
PackageKey = str
T = TypeVar('T')

# Constants
MODULE_NAME = 'dependency'
PARENT_MODULE = 'setup'

@dataclass
class ConfigState:
    """Holds the current state of the configuration."""
    config: ConfigDict = field(default_factory=dict)
    ui_components: UIComponents = field(default_factory=dict)
    last_error: Optional[str] = None
    package_categories: List[PackageCategory] = field(default_factory=list)

class DependencyConfigHandler(BaseConfigHandler):
    """Configuration handler for dependency management.
    
    Handles extraction, validation, and application of dependency configurations
    with proper error handling and logging.
    """
    
    def __init__(self, module_name: str = MODULE_NAME, 
                 parent_module: str = PARENT_MODULE):
        """Initialize the configuration handler.
        
        Args:
            module_name: Name of the module (default: 'dependency')
            parent_module: Parent module name (default: 'setup')
        """
        super().__init__(module_name, parent_module)
        self._state = ConfigState()
        self.logger = get_logger(f"smartcash.ui.{parent_module}.{module_name}")
        
        # Initialize package categories from defaults
        self._state.package_categories = PACKAGE_CATEGORIES
    
    def get_default_selected_packages(self) -> List[PackageKey]:
        """Get list of package keys that are selected by default.
        
        Returns:
            List of package keys that should be selected by default
        """
        try:
            selected = []
            
            # Get all packages that are marked as default or required
            for category in PACKAGE_CATEGORIES:
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
    
    def get_default_config(self) -> ConfigDict:
        """Get the default configuration for the dependency installer.
        
        Returns:
            Dictionary containing the default configuration with all settings.
        """
        try:
            # Create a deep copy of categories to avoid modifying the original
            categories_copy = []
            for cat in PACKAGE_CATEGORIES:
                cat_copy = {
                    'name': cat['name'],
                    'icon': cat.get('icon', '📦'),
                    'description': cat.get('description', ''),
                    'packages': []
                }
                
                # Copy packages with proper formatting
                for pkg in cat.get('packages', []):
                    pkg_copy = {
                        'key': pkg.get('key', pkg.get('name', '')),
                        'name': pkg.get('name', ''),
                        'description': pkg.get('description', ''),
                        'pip_name': pkg.get('pip_name', pkg.get('name', '')),
                        'default': pkg.get('default', False),
                        'min_version': pkg.get('min_version'),
                        'max_version': pkg.get('max_version')
                    }
                    # Remove None values
                    pkg_copy = {k: v for k, v in pkg_copy.items() if v is not None}
                    cat_copy['packages'].append(pkg_copy)
                    
                categories_copy.append(cat_copy)
            
            return {
                'version': CONFIG_VERSION,
                'schema_version': CONFIG_SCHEMA_VERSION,
                'auto_update': True,
                'check_on_startup': True,
                'selected_packages': self.get_default_selected_packages(),
                'package_manager': 'pip',
                'python_path': 'python',
                'use_venv': True,
                'venv_path': '.venv',
                'upgrade_strategy': 'eager',
                'timeout': 300,  # 5 minutes
                'retries': 3,
                'http_retries': 3,
                'prefer_binary': False,
                'trusted_hosts': [],
                'extra_index_urls': [],
                'constraints': [],
                'environment_variables': {},
                'post_install_commands': [],
                'log_level': 'INFO',
                'log_file': 'dependency_installer.log',
                'max_workers': min(4, (os.cpu_count() or 1) * 2),  # 2x CPU cores, max 4
                'categories': categories_copy
            }
            
        except Exception as e:
            self.logger.error(f"Error generating default config: {e}", exc_info=True)
            return self.get_minimal_config()
    
    def get_minimal_config(self) -> ConfigDict:
        """Get a minimal configuration for basic functionality.
        
        This configuration is used as a fallback or for testing purposes.
        It includes only essential settings with minimal dependencies.
        """
        try:
            # Get core packages that are marked as required
            required_packages = [
                pkg.get('key', pkg.get('name', ''))
                for cat in (self._state.package_categories or [])
                for pkg in cat.get('packages', [])
                if pkg.get('required', False) or pkg.get('default', False)
            ]
            
            # Ensure we have at least some basic packages
            if not required_packages:
                required_packages = ['numpy', 'pandas', 'matplotlib']
                
            return {
                'version': CONFIG_VERSION,
                'schema_version': CONFIG_SCHEMA_VERSION,
                'auto_update': False,  # Disable auto-updates in minimal mode
                'check_on_startup': False,  # Don't check on startup
                'selected_packages': required_packages,
                'package_manager': 'pip',
                'python_path': 'python',
                'use_venv': True,
                'venv_path': '.venv',
                'upgrade_strategy': 'only-if-needed',
                'timeout': 120,  # Shorter timeout for minimal config
                'retries': 2,  # Fewer retries
                'http_retries': 2,
                'prefer_binary': True,  # Prefer binaries for faster installation
                'trusted_hosts': ['pypi.org', 'files.pythonhosted.org'],
                'extra_index_urls': [],
                'constraints': [],
                'environment_variables': {},
                'post_install_commands': [],
                'log_level': 'WARNING',  # Higher log level to reduce noise
                'log_file': 'dependency_minimal.log',
                'max_workers': 2,  # Fewer workers
                'categories': [
                    {
                        'name': 'Core Dependencies',
                        'icon': '⚙️',
                        'description': 'Essential packages for minimal functionality',
                        'packages': [
                            pkg for cat in (self._state.package_categories or [])
                            for pkg in cat.get('packages', [])
                            if pkg.get('required', False) or pkg.get('default', False)
                        ][:5]  # Limit to first 5 packages
                    }
                ] if self._state.package_categories else [],
                'minimal': True  # Flag to indicate this is a minimal config
            }
            
        except Exception as e:
            self.logger.error(f"Error generating minimal config: {e}", exc_info=True)
            # Return absolute minimum configuration on error
            return {
                'version': CONFIG_VERSION,
                'schema_version': CONFIG_SCHEMA_VERSION,
                'auto_update': False,
                'check_on_startup': False,
                'selected_packages': ['numpy', 'pandas'],
                'categories': [],
                'error': str(e)
            }
    
    def get_default_dependency_config(self) -> ConfigDict:
        """Get the complete default configuration for the dependency installer.
        
        This method ensures that the configuration is properly initialized and validated.
        """
        try:
            # Get the default configuration
            config = self.get_default_config()
            
            # Ensure required fields exist
            required_fields = [
                'version', 'schema_version', 'selected_packages', 'categories',
                'package_manager', 'python_path', 'use_venv', 'venv_path'
            ]
            
            for field in required_fields:
                if field not in config:
                    self.logger.warning(f"Missing required field in config: {field}")
                    config[field] = None
            
            # Ensure categories is a list
            if not isinstance(config.get('categories'), list):
                self.logger.warning("Invalid or missing 'categories' in config, initializing...")
                config['categories'] = []
                
            # Ensure selected_packages is a list
            if not isinstance(config.get('selected_packages'), list):
                self.logger.warning("Invalid or missing 'selected_packages' in config, initializing...")
                config['selected_packages'] = self.get_default_selected_packages()
            
            # Set default values for critical settings if missing
            config.setdefault('auto_update', True)
            config.setdefault('check_on_startup', True)
            config.setdefault('package_manager', 'pip')
            config.setdefault('python_path', 'python')
            config.setdefault('use_venv', True)
            config.setdefault('venv_path', '.venv')
            config.setdefault('max_workers', min(4, (os.cpu_count() or 1) * 2))
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error in get_default_dependency_config: {e}", exc_info=True)
            return self.get_minimal_config()
    
    def extract_config(self, ui_components: UIComponents) -> ConfigDict:
        """Extract configuration from UI components with safe fallback.
        
        Args:
            ui_components: Dictionary of UI components to extract from
            
        Returns:
            Extracted and validated configuration dictionary
        """
        self._state.ui_components = ui_components
        
        try:
            from .config_extractor import extract_dependency_config
            self._state.config = extract_dependency_config(ui_components)
            self._state.last_error = None
            return self._state.config
            
        except Exception as e:
            self._state.last_error = str(e)
            self.logger.warning(f"Extract config error: {str(e)}", exc_info=True)
            self._state.config = self.get_default_config()
            return self._state.config
    
    def update_ui(self, ui_components: UIComponents, config: ConfigDict) -> None:
        """Update UI components with the current configuration.
        
        Args:
            ui_components: Dictionary of UI components to update
            config: Configuration to apply to the UI
        """
        if not isinstance(ui_components, dict) or not isinstance(config, dict):
            self.logger.error("Invalid UI components or config provided")
            return
            
        self._state.ui_components = ui_components
        self._state.config = config.copy()
        
        try:
            from .config_updater import update_dependency_ui
            update_dependency_ui(ui_components, config)
            self._log_to_ui("Dependency config updated", "success", ui_components)
            self._state.last_error = None
            
        except Exception as e:
            self._state.last_error = str(e)
            self.logger.error(f"Update UI error: {str(e)}", exc_info=True)
            self._log_to_ui(f"Update error: {str(e)}", "error", ui_components)
    
    def get_default_config(self) -> ConfigDict:
        """Get the default configuration with safe fallback.
        
        Returns:
            Default configuration dictionary
        """
        try:
            # Return the default config from defaults.py
            return DEFAULT_CONFIG.copy()
            
        except Exception as e:
            self._state.last_error = str(e)
            self.logger.error(f"Error getting default config: {e}", exc_info=True)
            self.logger.info("Using fallback default config")
            return {
                'module_name': MODULE_NAME,
                'dependencies': {
                    'torch': {'version': 'latest', 'required': True},
                    'torchvision': {'version': 'latest', 'required': True},
                    'ultralytics': {'version': 'latest', 'required': True}
                },
                'install_options': {
                    'force_reinstall': False,
                    'upgrade': True,
                    'quiet': False
                }
            }
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk current config"""
        return self._current_config.copy()
    
    def _log_to_ui(self, message: str, level: str, ui_components: Optional[Dict[str, Any]] = None):
        """Safe logging ke UI components"""
        target_ui = ui_components or self._ui_components
        if target_ui:
            safe_log_to_ui(target_ui, message, level)
        
        # Log ke standard logger juga
        if level == 'error':
            self.logger.error(message)
        elif level == 'warning':
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """Set UI components reference untuk logging"""
        self._ui_components = ui_components