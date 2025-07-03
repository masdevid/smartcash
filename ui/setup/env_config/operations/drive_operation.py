"""
File: smartcash/ui/setup/env_config/handlers/operations/drive_operation.py

Drive Operation Handler untuk stage-based setup.
"""

from typing import Dict, Any, Optional, Callable
from pathlib import Path

from smartcash.ui.core.handlers.operation_handler import OperationHandler, OperationResult
from smartcash.common.environment import get_environment_manager


class DriveOperation(OperationHandler):
    """Handler untuk drive mounting operations."""
    
    def __init__(self):
        """Initialize drive operation handler."""
        super().__init__(
            module_name='drive_operation',
            parent_module='setup.env_config'
        )
        
        self.env_manager = get_environment_manager()
        self._is_initialized = False
        
        # Initialize the handler
        self.initialize()
        
    def mount_drive(self) -> Dict[str, Any]:
        """Mount Google Drive.
        
        Returns:
            Dictionary berisi hasil mounting
        """
        try:
            self.logger.info("💾 Mounting Google Drive...")
            
            # Check if already mounted
            if self.env_manager.is_drive_mounted():
                mount_path = self.env_manager.get_drive_path()
                return {
                    'status': True,
                    'message': f'Drive sudah mounted di {mount_path}',
                    'mount_path': str(mount_path),
                    'already_mounted': True
                }
            
            # Attempt to mount
            result = self.env_manager.mount_drive()
            
            if result.get('status', False):
                return {
                    'status': True,
                    'message': 'Drive berhasil di-mount',
                    'mount_path': result.get('mount_path', ''),
                    'already_mounted': False
                }
            else:
                return {
                    'status': False,
                    'message': f'Gagal mount drive: {result.get("message", "Unknown error")}',
                    'error': result.get('error', 'Unknown error')
                }
                
        except Exception as e:
            error_msg = f"Drive mount operation failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e)
            }
    
    def verify_drive_mount(self) -> Dict[str, Any]:
        """Verify drive mount status.
        
        Returns:
            Dictionary berisi verification results
        """
        try:
            is_mounted = self.env_manager.is_drive_mounted()
            
            if is_mounted:
                mount_path = self.env_manager.get_drive_path()
                return {
                    'status': True,
                    'message': 'Drive mount verified',
                    'mount_path': str(mount_path),
                    'is_mounted': True
                }
            else:
                return {
                    'status': False,
                    'message': 'Drive not mounted',
                    'is_mounted': False
                }
                
        except Exception as e:
            error_msg = f"Drive verification failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e),
                'is_mounted': False
            }
    
    def get_drive_info(self) -> Dict[str, Any]:
        """Get drive information.
        
        Returns:
            Dictionary berisi drive info
        """
        try:
            info = {
                'is_mounted': self.env_manager.is_drive_mounted(),
                'mount_path': '',
                'available_space': 0,
                'total_space': 0
            }
            
            if info['is_mounted']:
                mount_path = self.env_manager.get_drive_path()
                info['mount_path'] = str(mount_path)
                
                # Get space info jika mounted
                if mount_path.exists():
                    import shutil
                    total, used, free = shutil.disk_usage(mount_path)
                    info['total_space'] = total
                    info['available_space'] = free
            
            return {
                'status': True,
                'info': info
            }
            
        except Exception as e:
            error_msg = f"Failed to get drive info: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'message': error_msg,
                'error': str(e)
            }
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler.
        
        Returns:
            Dictionary mapping operation names to their handler methods
        """
        return {
            'mount_drive': self.mount_drive,
            'verify_drive_mount': self.verify_drive_mount,
            'get_drive_info': self.get_drive_info
        }
    
    def initialize(self) -> bool:
        """Initialize the operation handler.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._is_initialized:
            return True
            
        try:
            self.logger.info("Initializing drive operation handler...")
            # Perform any necessary initialization here
            self._is_initialized = True
            self.logger.info("✅ Drive operation handler initialized")
            return True
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize drive operation handler: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self._is_initialized = False
            return False