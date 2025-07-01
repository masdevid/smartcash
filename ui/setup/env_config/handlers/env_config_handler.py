"""
Environment Configuration Handler Orchestrator.

This module provides the main orchestrator for environment configuration operations.
It serves as a central configuration service and coordinates between specialized
configuration handlers.
"""

from typing import Dict, Any, List, Optional, Type, TypeVar, cast
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin
from smartcash.ui.setup.env_config.constants import SetupStage

# Type variable for handler classes
T = TypeVar('T')

class ConfigState:
    """Represents the current state of the environment configuration."""
    
    def __init__(self):
        self.is_configured = False
        self.current_stage = SetupStage.INIT
        self.last_error = None
        self.config = {}
        self.handlers = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'is_configured': self.is_configured,
            'current_stage': self.current_stage.value,
            'last_error': str(self.last_error) if self.last_error else None,
            'handlers': {name: str(type(h)) for name, h in self.handlers.items()}
        }

class EnvConfigHandler(BaseHandler, BaseConfigMixin):
    """Central configuration service for environment setup.
    
    This class serves as the single source of truth for environment configuration.
    It manages the lifecycle of configuration handlers and provides a unified
    interface for accessing configuration state.
    
    Key Responsibilities:
    - Manages configuration state
    - Coordinates between configuration handlers
    - Provides configuration validation
    - Handles configuration persistence
    
    Note: This class does not handle workflow or UI logic - see SetupHandler for that.
    """
    
    # Default configuration for the handler
    DEFAULT_CONFIG = {
        'enable_logging': True,
        'log_level': 'INFO',
        'auto_initialize': True,
        'handlers': {
            'drive': {},
            'folder': {},
            'config': {},
            'status': {},
            'setup': {}
        },
        'validation': {
            'strict': True,
            'required_fields': []
        }
    }
    
    def __init__(self, config_handler=None, **kwargs):
        """Initialize the EnvConfigHandler with configuration.
        
        Args:
            config_handler: Instance of ConfigHandler for configuration
            **kwargs: Additional keyword arguments for BaseHandler
        """
        super().__init__(
            module_name='env_config',
            parent_module='setup',
            **kwargs
        )
        
        # Initialize BaseConfigMixin with the provided config handler
        BaseConfigMixin.__init__(self, config_handler=config_handler, **kwargs)
        
        # Initialize state
        self.state = ConfigState()
        self.handlers: Dict[str, BaseHandler] = {}
        
        # Setup logging
        self.logger.debug("Initializing EnvConfigHandler")
        
        # Initialize handlers if auto_initialize is True
        if self.get_config_value('auto_initialize', True):
            self.initialize_handlers()
    
    def initialize_handlers(self, **handler_kwargs) -> Dict[str, BaseHandler]:
        """Initialize all required configuration handlers.
        
        Args:
            **handler_kwargs: Additional keyword arguments to pass to handlers
            
        Returns:
            Dictionary of initialized handlers
            
        Raises:
            RuntimeError: If handler initialization fails
        """
        self.state.current_stage = SetupStage.INIT
        self.logger.info("Initializing configuration handlers")
        
        # Common kwargs for all handlers
        common_kwargs = {
            'config_handler': self.get_config_handler(),
            'logger': self.logger,
            **handler_kwargs
        }
        
        try:
            # Lazy import to avoid circular imports
            from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
            from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler
            from smartcash.ui.setup.env_config.handlers.config_handler import ConfigHandler
            from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
            from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
            
            # Initialize each handler
            self.handlers = {
                'drive': DriveHandler(**common_kwargs),
                'folder': FolderHandler(**common_kwargs),
                'config': ConfigHandler(**common_kwargs),
                'status': StatusChecker(**common_kwargs),
                'setup': SetupHandler(**common_kwargs)
            }
            
            # Initialize the setup handler's handlers
            if hasattr(self.handlers['setup'], 'initialize_handlers'):
                self.handlers['setup'].initialize_handlers(**common_kwargs)
            
            self.logger.info("All environment configuration handlers initialized")
            return self.handlers
            
        except Exception as e:
            error_msg = f"Error initializing environment configuration handlers: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def get_handler(self, handler_name: str) -> BaseHandler:
        """Get a handler by name.
        
        Args:
            handler_name: Name of the handler to retrieve
            
        Returns:
            The requested handler instance
            
        Raises:
            KeyError: If the handler is not found
        """
        if handler_name not in self.handlers:
            raise KeyError(f"Handler '{handler_name}' not found. Available handlers: {list(self.handlers.keys())}")
        return self.handlers[handler_name]
    
    async def run_setup(self, **kwargs) -> Dict[str, Any]:
        """Run the complete environment setup process.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the setup handler
            
        Returns:
            Dictionary with setup results
        """
        self.set_stage(SetupStage.SETUP, "Starting environment setup")
        
        try:
            # Ensure handlers are initialized
            if not self.handlers:
                await self.initialize_handlers()
            
            # Run the setup process
            setup_handler = self.get_handler('setup')
            result = await setup_handler.run_setup(**kwargs)
            
            self.set_stage(SetupStage.COMPLETE, "Environment setup completed")
            return result
            
        except Exception as e:
            error_msg = f"Error during environment setup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.set_stage(SetupStage.ERROR, error_msg)
            raise
    
    async def check_environment(self, **kwargs) -> Dict[str, Any]:
        """Check the environment status.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the status checker
            
        Returns:
            Dictionary with environment status information
        """
        self.set_stage(SetupStage.VERIFICATION, "Checking environment status")
        
        try:
            # Ensure handlers are initialized
            if not self.handlers:
                await self.initialize_handlers()
            
            # Run the status check
            status_checker = self.get_handler('status')
            result = await status_checker.check_environment(**kwargs)
            
            self.set_stage(SetupStage.VERIFICATION, "Environment check completed")
            return result
            
        except Exception as e:
            error_msg = f"Error checking environment: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.set_stage(SetupStage.ERROR, error_msg)
            raise
    
    async def mount_drive(self, **kwargs) -> Dict[str, Any]:
        """Mount Google Drive.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the drive handler
            
        Returns:
            Dictionary with mount results
        """
        self.set_stage(SetupStage.DRIVE_MOUNT, "Mounting Google Drive")
        
        try:
            # Ensure handlers are initialized
            if not self.handlers:
                await self.initialize_handlers()
            
            # Mount the drive
            drive_handler = self.get_handler('drive')
            result = await drive_handler.mount_drive(**kwargs)
            
            self.set_stage(SetupStage.DRIVE_MOUNT, "Google Drive mount completed")
            return result
            
        except Exception as e:
            error_msg = f"Error mounting Google Drive: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.set_stage(SetupStage.ERROR, error_msg)
            raise
    
    async def create_folders(self, **kwargs) -> Dict[str, Any]:
        """Create required folders and symlinks.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the folder handler
            
        Returns:
            Dictionary with folder creation results
        """
        self.set_stage(SetupStage.FOLDER_SETUP, "Creating required folders")
        
        try:
            # Ensure handlers are initialized
            if not self.handlers:
                await self.initialize_handlers()
            
            # Create folders
            folder_handler = self.get_handler('folder')
            result = await folder_handler.create_required_folders(**kwargs)
            
            self.set_stage(SetupStage.FOLDER_SETUP, "Folder creation completed")
            return result
            
        except Exception as e:
            error_msg = f"Error creating folders: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.set_stage(SetupStage.ERROR, error_msg)
            raise
    
    async def sync_configs(self, **kwargs) -> Dict[str, Any]:
        """Synchronize configurations.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the config handler
            
        Returns:
            Dictionary with sync results
        """
        self.set_stage(SetupStage.CONFIG_SYNC, "Synchronizing configurations")
        
        try:
            # Ensure handlers are initialized
            if not self.handlers:
                await self.initialize_handlers()
            
            # Sync configs
            config_handler = self.get_handler('config')
            result = await config_handler.sync_configurations(**kwargs)
            
            self.set_stage(SetupStage.CONFIG_SYNC, "Configuration sync completed")
            return result
            
        except Exception as e:
            error_msg = f"Error synchronizing configurations: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.set_stage(SetupStage.ERROR, error_msg)
            raise
    
    def get_setup_summary(self) -> Dict[str, Any]:
        """Get the summary of the last setup run.
        
        Returns:
            Dictionary with setup summary, or None if no setup has been run
        """
        if not self.handlers or 'setup' not in self.handlers:
            return {}
        
        setup_handler = self.handlers['setup']
        if hasattr(setup_handler, 'get_last_summary'):
            return setup_handler.get_last_summary() or {}
        
        return {}
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment.
        
        Returns:
            Dictionary with environment information
        """
        if not self.handlers or 'status' not in self.handlers:
            return {}
        
        status_checker = self.handlers['status']
        if hasattr(status_checker, 'get_last_check_result'):
            result = status_checker.get_last_check_result()
            if result and 'env_info' in result:
                return result['env_info']
        
        return {}
