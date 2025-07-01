"""
Environment Configuration Initializer Module.

This module provides the EnvConfigInitializer class which is responsible for
initializing and managing the environment configuration UI. It follows the
CommonInitializer pattern used throughout the application for consistency.

Key Features:
- Implements the CommonInitializer interface for consistent initialization
- Manages the lifecycle of environment configuration UI components
- Coordinates between EnvConfigHandler (configuration) and SetupHandler (workflow)
- Handles error states and provides user feedback
- Supports both programmatic and interactive usage patterns

Example:
    >>> initializer = EnvConfigInitializer()
    >>> ui = initializer.initialize()
    >>> display(ui)
"""

import os
import sys
from typing import Dict, Any, Optional, Type, TypeVar, cast

# Import CommonInitializer base class
from smartcash.ui.initializers.common_initializer import CommonInitializer

# Import consolidated logger
from smartcash.ui.utils.ui_logger import get_module_logger

# Import handlers
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.handlers.env_config_handler import EnvConfigHandler
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler as BaseConfigHandler

# Type aliases
T = TypeVar('T')

class EnvConfigInitializer(CommonInitializer):
    """Environment configuration UI initializer using CommonInitializer pattern.
    
    This class is responsible for initializing and managing the environment
    configuration UI. It follows the CommonInitializer pattern for consistency
    with other UI initializers in the application.
    
    The initialization process follows these steps:
    1. Initialize configuration handler
    2. Initialize EnvConfigHandler (configuration management)
    3. Initialize SetupHandler (workflow management)
    4. Set up UI components and event handlers
    5. Perform any post-initialization setup
    """
    
    def __init__(self, config_handler_class: Type[BaseConfigHandler] = ConfigHandler):
        """Initialize the environment configuration initializer.
        
        Args:
            config_handler_class: Optional ConfigHandler class (defaults to ConfigHandler)
        """
        # Initialize ui_components before calling parent's __init__
        self._ui_components = {}
        self.ui_components = self._ui_components
        super().__init__(module_name='env_config', config_handler_class=config_handler_class)
        self._env_config_handler = None
        self._setup_handler = None
    
    def _init_handlers(self, config: Dict[str, Any]) -> None:
        """Initialize the environment configuration and setup handlers.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Initializing environment configuration handlers")
        
        # Initialize EnvConfigHandler first (configuration management)
        self._env_config_handler = EnvConfigHandler(
            config_handler=self.config_handler,
            logger=self.logger
        )
        
        # Initialize SetupHandler with reference to EnvConfigHandler
        self._setup_handler = SetupHandler(
            config_handler=self.config_handler,
            env_config_handler=self._env_config_handler,
            logger=self.logger
        )
        
        # Store references in the handlers dictionary
        self._handlers = {
            'env_config': self._env_config_handler,
            'setup': self._setup_handler
        }
        
        self.logger.info("Environment configuration handlers initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for environment setup.
        
        Returns:
            Dictionary containing default configuration
            
        Raises:
            ImportError: If default config module is not found
        """
        self.logger.debug("Loading default configuration")
        
        try:
            # Default configuration for environment setup
            return {
                'version': '1.0.0',  # Add version for compatibility with tests
                'env_config': {
                    'env_name': 'smartcash_env',
                    'env_path': os.path.join(os.path.expanduser('~'), 'smartcash_envs'),
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
                },
                'ui': {'theme': 'dark', 'log_level': 'INFO'},
                'settings': {}  # Add empty settings for compatibility with tests
            }
            
        except Exception as e:
            self.logger.error(f"Gagal mendapatkan konfigurasi default: {str(e)}", exc_info=True)
            # Fallback to minimal config
            return {
                'version': '1.0.0',
                'env_config': {},
                'ui': {'theme': 'light', 'log_level': 'INFO'},
                'settings': {}
            }
    
    # Override _setup_handlers to use EnvConfigHandler orchestrator
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup event handlers with proper error handling
        
        Args:
            ui_components: Dictionary containing UI components
            config: Configuration to use
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components updated with handlers
            
        Raises:
            ValueError: If handler setup fails
        """
        try:
            self.logger.debug("Setting up event handlers")
            
            # Initialize the ConfigHandler first
            config_handler = ConfigHandler(
                module_name='env_config',
                parent_module='setup',
                persistence_enabled=True
            )
            
            # Initialize the EnvConfigHandler orchestrator with the config handler
            handler = EnvConfigHandler(
                config_handler=config_handler,
                ui_components=ui_components,
                config=config
            )
            
            if not handler:
                raise ValueError("Failed to initialize environment config handler")
                
            ui_components['env_config_handler'] = handler
            ui_components['config_handler'] = config_handler  # Store for later use
            self.logger.info("Environment config handler initialized successfully")
            
            # Set up event handlers
            if 'setup_button' in ui_components and hasattr(ui_components['setup_button'], 'on_click'):
                # Wrap the click handler to update status after sync
                async def wrapped_click(button):
                    try:
                        await handler.handle_setup_button_click(button)
                        # Update status panel after sync
                        if 'status_panel' in ui_components:
                            summary = handler.get_summary()
                            ui_components['status_panel'].value = self._format_summary(summary)
                    except Exception as e:
                        self.logger.error(f"Error in setup button click: {str(e)}", exc_info=True)
                
                ui_components['setup_button'].on_click(wrapped_click)
                self.logger.debug("Setup button click handler attached")
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to setup handlers: {str(e)}", exc_info=True)
            raise
            
    def _pre_initialize_checks(self, config: Dict[str, Any], **kwargs) -> None:
        """Perform pre-initialization checks for environment configuration.
        
        This method checks drive connectivity using EnvironmentManager before proceeding
        with the initialization process.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
            
        Raises:
            RuntimeError: If drive connectivity check fails
        """
        self.logger.debug("Checking drive connectivity before initialization")
        
        try:
            from smartcash.common.environment import get_environment_manager
            
            # Get environment manager instance
            env_manager = get_environment_manager()
            
            # Check if we're in Colab
            if not env_manager._in_colab:
                self.logger.warning("Not a Colab environment, skipping drive check")
                return
                
            # Refresh drive status
            drive_mounted = env_manager.refresh_drive_status()
            
            # Log drive path if mounted
            if drive_mounted and env_manager.drive_path:
                self.logger.info(f"Google Drive connected at: {env_manager.drive_path}")
            else:
                self.logger.warning("Google Drive not detected")
                
        except Exception as e:
            error_msg = f"Failed to check drive connectivity: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
    # Using CommonInitializer's _create_ui_components implementation
        
    def get_default_config(self) -> Dict[str, Any]:
        """Public method to get the default configuration."""
        return self._get_default_config()
        
    def get_ui_root(self) -> Any:
        """Get the root UI component.
        
        Returns:
            The root widget of the UI or None if not initialized
        """
        return self._ui_components.get('root') if hasattr(self, '_ui_components') else None
    
    # Removed _register_handlers() - handler registration is now handled in _setup_handlers()
            
    def _create_ui_components(self, config: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Create and return UI components as a dictionary.
        
        Args:
            config: Loaded configuration
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components with required keys:
            - 'ui': Main UI component
            - 'log_output': Log output widget
            - 'status_panel': Status panel widget
        """
        try:
            from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
            
            # Ensure we have a valid config
            if not config:
                config = self.config_handler.get_default_config()
                self.logger.debug("Using default configuration")
            
            # Ensure required sections exist
            if 'env_config' not in config:
                config['env_config'] = {}
                self.logger.debug("Added 'env_config' section to configuration")
            
            # Create UI components with immediate validation
            self.logger.info("Creating main UI components")
            ui_components = create_env_config_ui()
            
            if not isinstance(ui_components, dict):
                raise ValueError(f"UI components must be a dictionary, got: {type(ui_components)}")
            
            if not ui_components:
                raise ValueError("UI components cannot be empty")
            
            # Validate critical components exist
            required_components = ['ui', 'log_output', 'status_panel']
            missing = [comp for comp in required_components if comp not in ui_components]
            if missing:
                raise ValueError(f"Missing required UI components: {missing}")
            
            # Add module-specific metadata
            ui_components['config_handler'] = self.config_handler
            if 'env' in kwargs:
                ui_components['env'] = kwargs['env']
            
            self.logger.info("UI components initialized successfully")
            return ui_components
            
        except Exception as e:
            error_msg = f"Failed to create UI components: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise
    
    def _after_init_checks(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform post-initialization checks and updates.
        
        This method is called after all initialization is complete to perform
        any final checks or updates to the UI state, including syncing config
        templates if the drive is mounted and setup is complete.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            The updated ui_components dictionary
            
        Note:
            This method is called by the parent class's initialize() method
            after all other initialization is complete.
        """
        try:
            self.logger.info("Performing post-initialization checks...")
            
            # Perform initial status check
            self._perform_initial_status_check(ui_components)
            
            # Check if we should sync config templates
            if self._should_sync_config_templates():
                self.logger.info("Drive is mounted and setup is complete, syncing config templates...")
                self._update_status(ui_components, "Syncing config templates...", "info")
                self._sync_config_templates()
            
            # Update status to show initialization is complete
            self._update_status(ui_components, "Environment configuration UI ready", "success")
            
            self.logger.info("Post-initialization checks completed successfully")
            return ui_components
            
        except Exception as e:
            error_msg = f"Error during post-initialization checks: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._update_status(ui_components, f"Error: {str(e)}", "error")
            return ui_components
    
    def _should_sync_config_templates(self) -> bool:
        """Determine if config templates should be synced.
        
        Returns:
            bool: True if config templates should be synced, False otherwise
        """
        try:
            # Check if drive is mounted
            if not hasattr(self, '_env_manager') or not self._env_manager.drive_mounted:
                self.logger.debug("Skipping config template sync: Drive not mounted")
                return False
            
            # Check if setup is complete by looking for a marker file or config
            if not self._is_setup_complete():
                self.logger.debug("Skipping config template sync: Setup not complete")
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking if should sync config templates: {str(e)}")
            return False
    
    def _is_setup_complete(self) -> bool:
        """Check if the environment setup is complete.
        
        This checks for the existence of a marker file or configuration
        that indicates the initial setup has been completed.
        
        Returns:
            bool: True if setup is complete, False otherwise
        """
        try:
            # Check for a marker file in the config directory
            marker_file = Path(self.config_handler.config_dir) / '.setup_complete'
            return marker_file.exists()
            
        except Exception as e:
            self.logger.warning(f"Error checking setup completion: {str(e)}")
            return False
    
    def _sync_config_templates(self) -> None:
        """Sync configuration templates from repository to config directory.
        
        This method is called after initialization when the drive is mounted
        and setup is complete to ensure all required config templates are available.
        """
        try:
            self.logger.info("Starting config template synchronization...")
            
            # Get the config manager instance with auto_sync enabled
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager(auto_sync=True)
            
            # Sync all configs
            result = config_manager.sync_configs_to_drive(force_overwrite=False)
            
            if result.get('success', False):
                self.logger.info(
                    f"Successfully synced {result.get('synced_count', 0)} config templates. "
                    f"Skipped {result.get('skipped_count', 0)} up-to-date files."
                )
            else:
                self.logger.warning(
                    f"Config template sync completed with issues: {result.get('message', 'Unknown error')}"
                )
                
        except Exception as e:
            self.logger.error(f"Error during config template sync: {str(e)}", exc_info=True)

def initialize_env_config_ui(config: Dict[str, Any] = None, **kwargs) -> Any:
    """Initialize and return the environment configuration UI.
    
    This is the main entry point for the environment configuration UI.
    It creates an instance of EnvConfigInitializer and initializes it with the provided config.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments to pass to the initializer
        
    Returns:
        The initialized UI widget (success or error) from EnvConfigInitializer
    """
    # Get module logger for initialization
    logger = get_module_logger('smartcash.ui.setup.env_config')
    logger.debug("Initializing environment configuration UI")
    
    # Create and initialize the initializer
    initializer = EnvConfigInitializer()
    return initializer.initialize(config=config, **kwargs)