"""
Environment Configuration Handler.

This module provides the EnvConfigHandler class which serves as the main orchestrator
for environment configuration, coordinating between different handlers and managing
the overall environment setup process.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.handlers.base_env_handler import BaseEnvHandler
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler, SetupSummary
from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
from smartcash.ui.setup.env_config.constants import SetupStage


class EnvConfigHandler(BaseEnvHandler):
    """Main orchestrator for environment configuration.
    
    This handler coordinates between different environment configuration components:
    - SetupHandler: Manages the setup workflow
    - ConfigHandler: Manages configuration synchronization
    - Other environment-related handlers
    
    It provides a unified interface for the UI to interact with the environment
    configuration system.
    """
    
    def __init__(self, config_handler: Optional[ConfigHandler] = None, **kwargs):
        """Initialize the environment configuration handler.
        
        Args:
            config_handler: Optional ConfigHandler instance. If not provided, a new one will be created.
            **kwargs: Additional keyword arguments for BaseEnvHandler
        """
        super().__init__(
            module_name='env_config',
            parent_module='setup',
            **kwargs
        )
        
        # Initialize core handlers
        self._config_handler = config_handler or ConfigHandler(
            module_name='env_config',
            parent_module='setup',
            logger=self.logger
        )
        
        self._setup_handler = SetupHandler(
            config_handler=self._config_handler,
            logger=self.logger
        )
        
        self._current_stage = SetupStage.INIT
        self._status = 'idle'
        self._progress = 0.0
        
        self.logger.info("Environment configuration handler initialized")
    
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
            
            # Initialize configuration
            if config:
                self._config_handler.update_config(config)
            
            # Start the setup process
            success = await self._setup_handler.start_setup()
            
            self._status = 'ready' if success else 'error'
            self._progress = 100.0 if success else 0.0
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {str(e)}", exc_info=True)
            self._status = 'error'
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the environment configuration.
        
        Returns:
            Dict containing status information
        """
        setup_summary = self._setup_handler.get_summary()
        
        return {
            'status': self._status,
            'progress': self._progress,
            'current_stage': self._current_stage.value,
            'setup': dict(setup_summary) if setup_summary else {}
        }
    
    async def handle_setup_button_click(self, button) -> None:
        """Handle setup button click event.
        
        Args:
            button: The button that was clicked
        """
        try:
            if self._status == 'ready':
                await self.initialize_environment()
            else:
                self.logger.warning("Environment initialization already in progress")
                
        except Exception as e:
            self.logger.error(f"Error handling setup button click: {str(e)}", exc_info=True)
            self._status = 'error'
