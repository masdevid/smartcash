"""
Handler for configuration synchronization using SimpleConfigManager.

This module provides the ConfigHandler class which manages configuration
synchronization between different sources with proper error handling.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypedDict

from smartcash.common.environment import get_environment_manager
from smartcash.ui.handlers.error_handler import handle_ui_errors
from smartcash.ui.handlers.config_handlers import ConfigHandler as BaseConfigHandler
from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin

class SyncResult(TypedDict, total=False):
    """Type definition for configuration sync results.
    
    Attributes:
        synced_count: Number of configs successfully synced
        configs_synced: List of config names that were synced
        success: Overall success status
        errors: List of error messages if any
        details: Additional operation details
    """
    synced_count: int
    configs_synced: List[str]
    success: bool
    errors: List[str]
    details: Dict[str, Any]



class ConfigHandler(BaseConfigHandler, BaseConfigMixin):
    """Handler for configuration synchronization in the environment setup.
    
    This handler manages configuration synchronization using SimpleConfigManager,
    providing a consistent interface for configuration operations with proper
    error handling and logging.
    
    Attributes:
        config_manager: SimpleConfigManager instance for config operations
        _last_sync_result: Result of the last sync operation
    """
    
    # Default configuration for the handler
    DEFAULT_CONFIG = {
        'auto_sync': True,
        'max_retries': 3,
        'handlers': {
            'config': {},
            'drive': {},
            'folder': {},
            'status': {},
            'setup': {}
        },
        # These will be set in __init__ based on environment
        'config_dir': '',
        'repo_config_dir': '/smartcash/configs'  # Default repo config path
    }
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary containing the extracted configuration
        """
        try:
            config = {}
            
            # Extract configuration from UI components if needed
            # This is a minimal implementation to satisfy the abstract method
            
            self.logger.debug("Extracted config from UI components")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to extract config: {str(e)}", exc_info=True)
            return {}

    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components from configuration.
        
        Args:
            ui_components: Dictionary of UI components to update
            config: Configuration dictionary to apply
        """
        try:
            # Update UI components from configuration
            # This is a minimal implementation to satisfy the abstract method
            
            self.logger.debug("Updated UI components from config")
            
        except Exception as e:
            self.logger.error(f"Failed to update UI: {str(e)}", exc_info=True)
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None, 
                 repo_config_dir: Optional[Union[str, Path]] = None, **kwargs):
        """Initialize the ConfigHandler.
        
        Args:
            config_dir: Directory for local configuration files (str or Path). 
                       Defaults to '/configs' which should be a symlink to '/drive/MyDrive/SmartCash/configs' in Colab.
            repo_config_dir: Directory for repository configuration files (str or Path).
                           Defaults to '/smartcash/configs'.
            **kwargs: Additional keyword arguments including:
                - config: Optional configuration dictionary to use instead of loading from file
        """
        # Initialize environment manager
        env_manager = get_environment_manager()
        
        # Set default paths based on environment
        if config_dir is None:
            # Use base_dir/configs as the default config directory
            config_dir = env_manager.base_dir / 'configs'
            
        if repo_config_dir is None:
            # Default to /smartcash/configs in the repository
            repo_config_dir = Path('/smartcash/configs')
            
        # Ensure paths are Path objects
        config_dir = Path(config_dir) if not isinstance(config_dir, Path) else config_dir
        repo_config_dir = Path(repo_config_dir) if not isinstance(repo_config_dir, Path) else repo_config_dir
            
        # Store paths in config
        config = {
            'config_dir': str(config_dir),
            'repo_config_dir': str(repo_config_dir),
            'auto_sync': True,
            'max_retries': 3,
            'retry_delay': 2.0
        }
        
        # Update with any provided config values
        if 'config' in kwargs:
            config.update(kwargs.pop('config'))
            
        # Store config for later use
        self.config = config
            
        # Call parent constructor with persistence disabled
        super().__init__(
            module_name='config',  # Explicit module name for consistency
            parent_module='env_config',
            use_shared_config=False,
            persistence_enabled=False,
            **kwargs
        )
        
        # Initialize BaseConfigMixin
        BaseConfigMixin.__init__(self, config_handler=None, **kwargs)
        
        # Initialize last sync result
        self._last_sync_result = None
        
        # Log initialization after parent class has set up logger
        self.logger.debug(f"Environment: colab={env_manager.is_colab}, drive_mounted={env_manager.is_drive_mounted}")
        self.logger.debug(f"Using config_dir: {self.config['config_dir']}")
        self.logger.debug(f"Using repo_config_dir: {self.config['repo_config_dir']}")
        self.logger.debug("ConfigHandler initialized with centralized configuration")
        
    @property
    def last_sync_result(self) -> Optional[SyncResult]:
        """Get the result of the last sync operation.
        
        Returns:
            The last sync result, or None if no sync has been performed
        """
        return self._last_sync_result
        
    def get_config_handler(self) -> 'ConfigHandler':
        """Get the underlying ConfigHandler instance.
        
        Returns:
            The ConfigHandler instance used by this handler
        """
        return self._config_handler
        
    async def initialize(self, **kwargs) -> Dict[str, Any]:
        """Initialize the config handler.
        
        Args:
            **kwargs: Additional initialization parameters
            
        Returns:
            Dictionary containing initialization result
        """
        if not self._initialized:
            # Initialize the config handler with default config
            default_config = self._config_handler.get_default_config()
            self._config_handler.update_ui({}, default_config)
            self._initialized = True
            
        return {
            'status': True,
            'message': 'Config handler initialized successfully',
            'initialized': True
        }
    
    @handle_ui_errors(operation="sync_configurations")
    async def sync_configurations(self) -> Dict[str, Any]:
        """Synchronize configurations from the repository to local storage.
        
        This is a convenience method that calls sync_configs_from_repo with default parameters.
        
        Returns:
            Dictionary containing the sync operation results with 'status' key
        """
        try:
            # Update progress if tracker is available
            # Progress tracking removed - handled by workflow manager
            
            result = await self.sync_configs_from_repo()
            
            # Update progress based on result
            if hasattr(self, 'progress_tracker'):
                if result.get('status'):
                    self.progress_tracker.set_progress(
                        100, 
                        f"Successfully synced {result.get('synced_count', 0)} configurations"
                    )
                else:
                    self.progress_tracker.set_error(
                        f"Sync failed: {', '.join(result.get('errors', ['Unknown error']))}"
                    )
            
            return {
                'status': result.get('status', False),
                'message': 'Sync completed',
                'synced_count': result.get('synced_count', 0),
                'errors': result.get('errors', [])
            }
            
        except Exception as e:
            error_msg = f"Error in sync_configurations: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Progress tracking removed - handled by workflow manager
            return {
                'status': False,
                'message': error_msg,
                'errors': [error_msg]
            }
    
    @handle_ui_errors(operation="sync_configs_from_repo")
    async def sync_configs_from_repo(self, force: bool = False) -> Dict[str, Any]:
        """Synchronize configuration files from repository to local config directory.
        
        Args:
            force: If True, overwrite existing configs even if they haven't changed
            
        Returns:
            Dictionary containing sync operation results with 'status' key
        """
        result: Dict[str, Any] = {
            'status': False,
            'synced_count': 0,
            'configs_synced': [],
            'errors': [],
            'details': {}
        }
        
        try:
            # Update progress
            # Progress tracking removed - handled by workflow manager
            
            # Get available configs from repository
            repo_configs = await self.get_available_configs()
            if not repo_configs:
                error_msg = "No configurations found in repository"
                self.logger.error(error_msg)
                result['errors'].append(error_msg)
                return result
            
            # Process each config file
            total_files = len(repo_configs)
            synced_count = 0
            
            for idx, config_file in enumerate(repo_configs, 1):
                try:
                    # Skip if config file exists and we're not forcing
                    if not force and os.path.exists(os.path.join(self.config['config_dir'], config_file)):
                        self.logger.debug(f"Skipping existing config: {config_file}")
                        continue
                    
                    # Copy config from repository
                    src_path = os.path.join(self.config['repo_config_dir'], config_file)
                    dest_path = os.path.join(self.config['config_dir'], config_file)
                    
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Copy the file
                    import shutil
                    shutil.copy2(src_path, dest_path)
                    
                    synced_count += 1
                    result['configs_synced'].append(config_file)
                    self.logger.info(f"Successfully synced config: {config_file}")
                    
                except Exception as e:
                    error_msg = f"Failed to sync config {config_file}: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    result['errors'].append(error_msg)
            
            # Update result
            result['synced_count'] = synced_count
            result['total_files'] = total_files
            result['status'] = len(result['errors']) == 0
            
            return result
            
        except Exception as e:
            error_msg = f"Error in sync_configs_from_repo: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result['errors'].append(error_msg)
            return result
    
    async def get_available_configs(self) -> List[str]:
        """Get list of available config files in the repository.
        
        Returns:
            List of configuration file paths relative to the repository config directory
            
        Raises:
            RuntimeError: If there's an error discovering repository configs
        """
        try:
            # Get repository configs
            self.logger.debug(f"Discovering configs in: {self.config['repo_config_dir']}")
            repo_configs = []
            
            # Check if repo directory exists
            if not os.path.exists(self.config['repo_config_dir']):
                error_msg = f"Repository config directory not found: {self.config['repo_config_dir']}"
                self.logger.error(error_msg)
                return []
                
            # Use config manager to discover configs if available
            if hasattr(self, 'config_manager') and hasattr(self.config_manager, 'discover_repo_configs'):
                repo_configs = await self._to_async(self.config_manager.discover_repo_configs)()
            else:
                # Fallback to manual discovery if config_manager is not available
                for root, _, files in os.walk(self.config['repo_config_dir']):
                    for file in files:
                        if file.endswith(('.yaml', '.yml', '.json', '.toml')):
                            rel_path = os.path.relpath(os.path.join(root, file), self.config['repo_config_dir'])
                            repo_configs.append(rel_path)
            
            self.logger.info(f"Found {len(repo_configs)} config files in repository")
            
            return repo_configs
            
        except Exception as e:
            error_msg = f"Error discovering repository configurations: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return []
    
    async def _to_async(self, func):
        """Convert a synchronous function to async if needed."""
        if hasattr(func, '__await__'):
            return await func
        return func
    
    @handle_ui_errors(operation="check_and_copy_missing_configs")
    async def check_and_copy_missing_configs(self) -> Dict[str, Any]:
        """Check and copy missing configs from repository to local config directory.
        
        Returns:
            Dict containing:
            - status: Overall success status
            - message: Summary message
            - copied: List of copied config files
            - missing: List of configs that couldn't be copied
            - total_repo: Total configs found in repository
            - total_local: Total configs in local directory after operation
            - errors: List of error messages if any
        """
        result: Dict[str, Any] = {
            'status': False,
            'message': '',
            'copied': [],
            'missing': [],
            'total_repo': 0,
            'total_local': 0,
            'errors': []
        }
        
        try:
            # Get repository configs
            repo_configs = await self.get_available_configs()
            result['total_repo'] = len(repo_configs)
            
            if not repo_configs:
                msg = "No configurations found in repository"
                self.logger.warning(msg)
                result.update({
                    'status': False,
                    'message': msg,
                    'missing': []
                })
                return result
            
            # Get local configs
            local_configs = []
            config_dir = Path(self.config['config_dir'])
            
            if config_dir.exists():
                local_configs = [f.name for f in config_dir.rglob('*.yaml')] + \
                              [f.name for f in config_dir.rglob('*.yml')] + \
                              [f.name for f in config_dir.rglob('*.json')] + \
                              [f.name for f in config_dir.rglob('*.toml')]
            
            result['total_local'] = len(local_configs)
            
            # Find missing configs
            missing_configs = [cfg for cfg in repo_configs 
                             if os.path.basename(cfg) not in local_configs]
            
            if not missing_configs:
                msg = "All configurations are up to date"
                self.logger.info(msg)
                
                result.update({
                    'status': True,
                    'message': msg,
                    'copied': [],
                    'missing': []
                })
                return result
            
            # Copy missing configs
            copied = []
            failed = []
            total_files = len(missing_configs)
            
            for idx, config in enumerate(missing_configs, 1):
                try:
                    src_path = os.path.join(self.config['repo_config_dir'], config)
                    dest_path = os.path.join(self.config['config_dir'], config)
                    
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Copy the file
                    import shutil
                    shutil.copy2(src_path, dest_path)
                    copied.append(config)
                    self.logger.info(f"Copied config: {config}")
                    
                except Exception as e:
                    error_msg = f"Failed to copy config {config}: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    failed.append(config)
                    result['errors'].append(error_msg)
            
            # Update result
            result.update({
                'status': len(failed) == 0,
                'copied': copied,
                'missing': failed,
                'total_local': len(local_configs) + len(copied)
            })
            
            # Set appropriate message
            if copied and not failed:
                msg = f"Successfully copied {len(copied)} configurations"
                result['message'] = msg
                if hasattr(self, 'progress_tracker'):
                    self.progress_tracker.set_progress(100, msg)
            elif copied and failed:
                msg = f"Copied {len(copied)} configurations, failed to copy {len(failed)}"
                result['message'] = msg
                # Progress tracking removed - handled by workflow manager
            else:
                msg = "Failed to copy any configurations"
                result['message'] = msg
                # Progress tracking removed - handled by workflow manager
            
            return result
            
        except Exception as e:
            error_msg = f"Error in check_and_copy_missing_configs: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Progress tracking removed - handled by workflow manager
            
            # Update result with error
            result.update({
                'status': False,
                'message': error_msg,
                'errors': result.get('errors', []) + [error_msg]
            })
            
            return result