"""
File: smartcash/ui/setup/env_config/env_config_initializer.py

Environment Configuration Initializer Module.

This module provides the EnvConfigInitializer class which is responsible for
initializing and managing the environment configuration UI. It follows the
CommonInitializer pattern used throughout the application for consistency.

Key Features:
- Implements the CommonInitializer interface for consistent initialization
- Manages the lifecycle of environment configuration UI components
- Coordinates configuration and workflow using ConfigHandler and SetupHandler
- Handles error states and provides user feedback
- Supports both programmatic and interactive usage patterns

Example:
    >>> initializer = EnvConfigInitializer()
    >>> ui = initializer.initialize()
    >>> display(ui)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, cast

# Import CommonInitializer base class
from smartcash.ui.initializers.common_initializer import CommonInitializer

# Import consolidated logger
from smartcash.ui.utils.ui_logger import get_module_logger

# Import handlers
from smartcash.ui.setup.env_config.handlers.env_config_handler import EnvConfigHandler
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
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
    2. Initialize SetupHandler with ConfigHandler
    3. Set up UI components and event handlers
    4. Perform any post-initialization setup
    """
    
    def __init__(self, config_handler_class: Type[BaseConfigHandler] = ConfigHandler):
        """Initialize the environment configuration initializer.
        
        Args:
            config_handler_class: Optional ConfigHandler class (defaults to ConfigHandler)
            
        Raises:
            RuntimeError: If environment manager initialization fails
        """
        try:
            # Initialize ui_components before calling parent's __init__
            self._ui_components = {}
            self.ui_components = self._ui_components
            
            # Initialize environment manager
            from smartcash.common.environment import get_environment_manager
            self._env_manager = get_environment_manager()
            
            # Call parent initializer
            super().__init__(module_name='env_config', config_handler_class=config_handler_class)
            self._env_config_handler = None
            
        except Exception as e:
            error_msg = f"Failed to initialize environment manager: {str(e)}"
            raise RuntimeError(error_msg) from e
    
    def _init_handlers(self, config: Dict[str, Any]) -> None:
        """Initialize the environment configuration and setup handlers.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Initializing environment configuration handlers")
        
        # Initialize the main EnvConfigHandler which will manage other handlers
        self._env_config_handler = EnvConfigHandler(
            logger=self.logger
        )
        
        # Store references in the handlers dictionary
        self._handlers = {
            'env_config': self._env_config_handler,
            'setup': self._env_config_handler.setup_handler,
            'config': self._env_config_handler.config_handler
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
    
    # Override _setup_handlers to use ConfigHandler
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
            
            # Store the main handler in ui_components
            ui_components['env_config_handler'] = self._env_config_handler
            ui_components['config_handler'] = self._env_config_handler.config_handler
            ui_components['setup_handler'] = self._env_config_handler.setup_handler
            
            self.logger.info("Environment config handler initialized successfully")
            
            # Set up event handlers
            if 'setup_button' in ui_components and hasattr(ui_components['setup_button'], 'on_click'):
                # Directly use the handler's method for button clicks
                ui_components['setup_button'].on_click(self._env_config_handler.handle_setup_button_click)
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
            # Use the instance's environment manager
            if not hasattr(self, '_env_manager') or self._env_manager is None:
                raise RuntimeError("Environment manager not initialized")
            
            # Check if we're in Colab
            if not self._env_manager._in_colab:
                self.logger.warning("Not a Colab environment, skipping drive check")
                return
                
            # Refresh drive status
            drive_mounted = self._env_manager.refresh_drive_status()
            
            # Log drive path if mounted
            if drive_mounted and self._env_manager.drive_path:
                self.logger.info(f"Google Drive connected at: {self._env_manager.drive_path}")
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
        """
        try:
            self.logger.info("Performing post-initialization checks...")
            # Delegate status check to setup handler if available
            if hasattr(self, '_env_config_handler') and hasattr(self._env_config_handler, 'setup_handler'):
                setup_handler = self._env_config_handler.setup_handler
                
                # Perform initial status check using setup handler
                setup_handler._perform_initial_status_check(ui_components)
                
                # Check if we should sync config templates
                if setup_handler._should_sync_config_templates():
                    self.logger.info("Drive is mounted and setup is complete, syncing config templates...")
                    # Use the new sync method with UI updates enabled for the initializer
                    setup_handler.sync_config_templates(
                        force_overwrite=False,
                        update_ui=True,
                        ui_components=ui_components
                    )
                
                
            else:
                self.logger.warning("Setup handler not available for post-initialization checks")
            
            self.logger.info("Post-initialization checks completed successfully")
            return ui_components
            
        except Exception as e:
            error_msg = f"Error during post-initialization checks: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return ui_components


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