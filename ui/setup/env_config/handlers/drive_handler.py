"""
Google Drive mounting and management handler.

This module provides the DriveHandler class which manages Google Drive mounting
and related operations with proper error handling and status updates.
"""

import os
from typing import Dict, Any, Optional, TypedDict, Union, cast, Tuple
from pathlib import Path

from smartcash.common.environment import get_environment_manager
from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.handlers.base_config_mixin import BaseConfigMixin
from smartcash.ui.setup.env_config.constants import SetupStage

class MountResult(TypedDict, total=False):
    """Type definition for drive mount operation results."""
    status: bool  # True for success, False for failure
    mount_path: str
    already_mounted: bool
    cancelled: bool
    reason: str
    message: str
    error: str
    user_cancelled: bool

class DriveHandler(BaseHandler, BaseConfigMixin):
    """Handler for Google Drive operations.
    
    This handler manages Google Drive mounting and operations in the environment,
    using the environment manager for drive-related operations.
    """
    
    # Default configuration for the handler
    DEFAULT_CONFIG = {
        'auto_mount': True,
        'mount_path': '/content/drive',
        'force_remount': False,
        'timeout_seconds': 30,
        'use_metadata_server': True
    }
    
    def __init__(self, config_handler=None, **kwargs):
        """Initialize the DriveHandler with configuration.
        
        Args:
            config_handler: Instance of ConfigHandler for configuration
            **kwargs: Additional keyword arguments for BaseHandler
        """
        # Initialize BaseHandler first
        super().__init__(
            module_name='drive',
            parent_module='env_config',
            **kwargs
        )
        
        # Then initialize BaseConfigMixin
        BaseConfigMixin.__init__(self, config_handler=config_handler, **kwargs)
        
        # Get environment manager
        self.env_manager = get_environment_manager(logger=self.logger)
        
        # Initialize last mount result
        self._last_mount_result = None
        
        self.logger.debug("DriveHandler initialized with configuration")
        self.logger.debug(f"Auto-mount: {self.get_config_value('auto_mount')}")
        self.logger.debug(f"Mount path: {self.get_config_value('mount_path')}")
        
        # Initialize from config
        self.auto_mount = self.get_config_value('auto_mount', True)
        self.max_retries = self.get_config_value('max_retries', 3)
        self.retry_delay = self.get_config_value('retry_delay', 2)
        
        # Use environment manager's drive path
        self.mount_path = str(self.env_manager.drive_path) if self.env_manager.drive_path else '/content/drive'
        
        self._last_mount_result: Optional[MountResult] = None
        
        self.logger.debug(f"Initialized DriveHandler with mount_path={self.mount_path}")
    
    async def mount_drive(self, force_remount: bool = False) -> MountResult:
        """Mount Google Drive with proper error handling and status updates.
        
        Args:
            force_remount: Whether to force remount even if already mounted
            
        Returns:
            MountResult dictionary with operation status and details
        """
        self.set_stage(SetupStage.DRIVE_MOUNT, "Mounting Google Drive")
        
        # Check if already mounted
        if not force_remount and self.env_manager.is_drive_mounted:
            self.logger.info("Google Drive is already mounted")
            self._last_mount_result = {
                'status': True,
                'mount_path': str(self.env_manager.drive_path),
                'already_mounted': True,
                'cancelled': False,
                'reason': 'already_mounted',
                'message': 'Google Drive is already mounted'
            }
            return self._last_mount_result
        
        try:
            # Use environment manager to mount drive
            status_result, message = self.env_manager.mount_drive()
            
            # Update mount path from environment manager
            self.mount_path = str(self.env_manager.drive_path) if self.env_manager.drive_path else self.mount_path
            
            if status_result:
                self._last_mount_result = {
                    'status': True,
                    'mount_path': self.mount_path,
                    'already_mounted': False,
                    'cancelled': False,
                    'reason': 'mounted',
                    'message': message
                }
                self.logger.info("Google Drive mounted successfully")
            else:
                self._last_mount_result = {
                    'status': False,
                    'mount_path': self.mount_path,
                    'already_mounted': False,
                    'cancelled': False,
                    'reason': 'mount_failed',
                    'message': message,
                    'error': message
                }
                self.logger.error(f"Failed to mount Google Drive: {message}")
            
        except Exception as e:
            error_msg = f"Failed to mount Google Drive: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._last_mount_result = {
                'status': False,
                'mount_path': self.mount_path,
                'already_mounted': False,
                'cancelled': False,
                'reason': 'mount_failed',
                'message': error_msg,
                'error': str(e)
            }
        
        return self._last_mount_result
    
    def _is_drive_mounted(self) -> bool:
        """Check if Google Drive is currently mounted using environment manager.
        
        Returns:
            bool: True if drive is mounted, False otherwise
        """
        # Use environment manager's drive status
        return self.env_manager.is_drive_mounted
    
    def get_last_mount_result(self) -> Optional[MountResult]:
        """Get the result of the last mount operation.
        
        Returns:
            MountResult dictionary with the last operation's status and details,
            or None if no mount operation has been performed yet.
        """
        return self._last_mount_result
