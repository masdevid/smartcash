"""
Environment Configuration Handler.

This module provides the EnvConfigHandler class which serves as the main orchestrator
for environment configuration, coordinating between different handlers and managing
the overall environment setup process.
"""

from typing import Dict, Any, Optional, List, Callable
import logging
from pathlib import Path

# Import core handlers
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.handlers.ui_handler import UIHandler
from smartcash.ui.core.handlers.config_handler import ConfigHandler as CoreConfigHandler

# Import module-specific handlers
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler, SetupSummary
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.constants import SetupStage


class EnvConfigHandler(UIHandler):
    """Main orchestrator for environment configuration.
    
    This handler coordinates between different environment configuration components:
    - SetupHandler: Manages the setup workflow
    - ConfigHandler: Manages configuration synchronization
    - Other environment-related handlers
    
    It provides a unified interface for the UI to interact with the environment
    configuration system.
    """
    
    def __init__(
        self,
        ui_components: Dict[str, Any],
        parent_module: str = 'setup',
        module_name: str = 'env_config',
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the environment configuration handler.
        
        Args:
            ui_components: Dictionary containing UI components
            parent_module: Parent module name (default: 'setup')
            module_name: Module name (default: 'env_config')
            logger: Optional logger instance
        """
        # Initialize the base handler
        super().__init__(
            ui_components=ui_components,
            logger=logger
        )
        
        # Store module information
        self.module_name = module_name
        self.parent_module = parent_module
        
        # Initialize state
        self._current_stage = SetupStage.INIT
        self._status = 'idle'
        self._progress = 0.0
        
        self.logger.info("Environment configuration handler initialized")
    
    def setup(self) -> None:
        """Set up the handler after initialization.
        
        This method initializes the config handler and setup handler.
        """
        # Initialize config handler if not already provided
        self._config_handler = self.ui_components.get('config_handler')
        if not self._config_handler:
            self._config_handler = ConfigHandler(
                ui_components=self.ui_components,
                logger=self.logger
            )
            self.ui_components['config_handler'] = self._config_handler
        
        # Initialize setup handler
        self._setup_handler = SetupHandler(
            ui_components=self.ui_components,
            config_handler=self._config_handler,
            logger=self.logger
        )
        self.ui_components['setup_handler'] = self._setup_handler
        
        # Set up event handlers for UI components
        self._setup_event_handlers()
        
        self.logger.debug("EnvConfigHandler setup completed")
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for UI components."""
        # Set up setup button click handler
        if 'setup_button' in self.ui_components:
            setup_button = self.ui_components['setup_button']
            if hasattr(setup_button, 'on_click'):
                setup_button.on_click(self.handle_setup_button_click)
    
    @property
    def config_handler(self) -> ConfigHandler:
        """Get the configuration handler instance."""
        return self._config_handler
    
    @property
    def setup_handler(self) -> SetupHandler:
        """Get the setup handler instance."""
        return self._setup_handler
    
    async def initialize_environment(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the environment with the given configuration.
        
        Args:
            config: Optional configuration overrides
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self._status = 'initializing'
            self._progress = 0.0
            self.update_status("Initializing environment...", "info")
            
            # Initialize configuration
            if config and hasattr(self._config_handler, 'set_config'):
                self._config_handler.set_config(config)
            
            # Start the setup process
            status_result = await self._setup_handler.start_setup()
        
            self._status = 'ready' if status_result else 'error'
            self._progress = 100.0 if status_result else 0.0
            
            message = "Environment initialized successfully" if status_result else "Failed to initialize environment"
            status_type = "success" if status_result else "error"
            self.update_status(message, status_type)
        
            return status_result
            
        except Exception as e:
            error_msg = f"Failed to initialize environment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._status = 'error'
            self.update_status(error_msg, "error")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the environment configuration.
        
        Returns:
            Dict containing status information
        """
        setup_summary = self._setup_handler.get_summary() if hasattr(self._setup_handler, 'get_summary') else None
        
        return {
            'status': self._status,
            'progress': self._progress,
            'current_stage': self._current_stage.value,
            'setup': dict(setup_summary) if setup_summary else {}
        }
    
    def handle_setup_button_click(self, button) -> None:
        """Handle setup button click event.
        
        Args:
            button: The button that was clicked
        """
        try:
            if self._status == 'ready':
                import asyncio
                asyncio.create_task(self.initialize_environment())
            else:
                self.logger.warning("Environment initialization already in progress")
                self.update_status("Environment initialization already in progress", "warning")
                
        except Exception as e:
            error_msg = f"Error handling setup button click: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._status = 'error'
            self.update_status(error_msg, "error")
