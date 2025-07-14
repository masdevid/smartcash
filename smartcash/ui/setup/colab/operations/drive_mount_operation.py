"""
File: smartcash/ui/setup/colab/operations/drive_mount_operation.py
Description: Mount Google Drive with verification using EnvironmentManager
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment


class DriveMountOperation(OperationHandler):
    """Mount Google Drive with verification."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize drive mount operation.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='drive_mount_operation',
            parent_module='colab',
            **kwargs
        )
        self.config = config
    
    def initialize(self) -> None:
        """Initialize the drive mount operation."""
        self.logger.info("🚀 Initializing drive mount operation")
        # No specific initialization needed for drive mount operation
        self.logger.info("✅ Drive mount operation initialization complete")
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'mount_drive': self.execute_mount_drive
        }
    
    def execute_mount_drive(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Mount Google Drive with verification using EnvironmentManager.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation results
        """
        try:
            if progress_callback:
                progress_callback(10, "🔍 Mengecek status environment...")
            
            # Use standardized environment manager
            env_manager = get_environment_manager(logger=self.logger)
            
            if not env_manager.is_colab:
                system_info = env_manager.get_system_info()
                error_msg = (
                    f"Mount Google Drive hanya tersedia di lingkungan Colab. "
                    f"Environment saat ini: {system_info.get('environment', 'Unknown')}"
                )
                return {
                    'success': False,
                    'error': error_msg,
                    'environment_info': system_info
                }
            
            if progress_callback:
                progress_callback(30, "🔍 Mengecek status mount Drive...")
            
            # Check if already mounted using EnvironmentManager
            if env_manager.is_drive_mounted:
                self.log("Google Drive sudah terpasang", 'info')
                if progress_callback:
                    progress_callback(100, "✅ Google Drive sudah terpasang")
                
                drive_path = env_manager.drive_path
                # Get updated paths
                paths = get_paths_for_environment(is_colab=True, is_drive_mounted=True)
                
                return {
                    'success': True,
                    'already_mounted': True,
                    'path': str(drive_path) if drive_path else '/content/drive',
                    'paths': paths,
                    'message': 'Google Drive sudah terpasang sebelumnya'
                }
            
            if progress_callback:
                progress_callback(50, "📁 Memasang Google Drive...")
            
            # Use EnvironmentManager to mount drive
            success, message = env_manager.mount_drive()
            
            if progress_callback:
                progress_callback(90, "🔍 Memverifikasi mount...")
            
            if success:
                # Get updated paths after successful mount
                paths = get_paths_for_environment(is_colab=True, is_drive_mounted=True)
                
                # Test write access if drive path is available
                write_access = False
                if env_manager.drive_path:
                    write_access = self._test_write_access(str(env_manager.drive_path))
                
                if progress_callback:
                    progress_callback(100, "✅ Google Drive berhasil dipasang")
                
                self.log("Google Drive berhasil dipasang dan diverifikasi", 'success')
                
                return {
                    'success': True,
                    'path': str(env_manager.drive_path) if env_manager.drive_path else '/content/drive',
                    'paths': paths,
                    'write_access': write_access,
                    'message': message
                }
            else:
                return {
                    'success': False,
                    'error': message
                }
                
        except Exception as e:
            self.log(f"Drive mount operation failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Drive mount operation failed: {str(e)}'
            }
    
    def _test_write_access(self, drive_path: str) -> bool:
        """Test write access to mounted drive.
        
        Args:
            drive_path: Path to mounted drive
            
        Returns:
            True if write access is available, False otherwise
        """
        try:
            test_file = os.path.join(drive_path, '.smartcash_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            self.log("Akses tulis Drive diverifikasi", 'info')
            return True
        except Exception as e:
            self.log(f"Akses tulis Drive terbatas: {e}", 'warning')
            return False