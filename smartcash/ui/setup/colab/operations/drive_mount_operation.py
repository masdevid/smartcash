"""
File: smartcash/ui/setup/colab/operations/drive_mount_operation.py
Description: Mount Google Drive with verification using EnvironmentManager
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment


class DriveMountOperation(BaseColabOperation):
    """Mount Google Drive with verification."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize drive mount operation.
        
        Args:
            operation_name: Name of the operation
            config: Configuration dictionary
            operation_container: Optional operation container for UI integration
            **kwargs: Additional arguments
        """
        super().__init__(operation_name, config, operation_container, **kwargs)
    
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
        def execute_operation():
            progress_steps = self.get_progress_steps('mount')
            
            # Step 1: Check environment status
            self.update_progress_safe(
                progress_callback, 
                progress_steps[0]['progress'], 
                progress_steps[0]['message'],
                progress_steps[0].get('phase_progress', 0)
            )
            
            # Simulate environment check
            env_check_steps = [
                ("Menginisialisasi manajer lingkungan...", 30),
                ("Memeriksa tipe lingkungan...", 60),
                ("Menyiapkan akses...", 90),
                ("Pemeriksaan lingkungan selesai", 100)
            ]
            
            for msg, phase_pct in env_check_steps:
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[0]['progress'] + (progress_steps[1]['progress'] - progress_steps[0]['progress']) * (phase_pct / 100)),
                    msg,
                    int(progress_steps[0].get('phase_progress', 0) + (progress_steps[1].get('phase_progress', 0) - progress_steps[0].get('phase_progress', 0)) * (phase_pct / 100))
                )
            
            # Use standardized environment manager
            env_manager = get_environment_manager(logger=self.logger)
            
            if not env_manager.is_colab:
                system_info = env_manager.get_system_info()
                error_msg = (
                    f"Mount Google Drive hanya tersedia di lingkungan Colab. "
                    f"Environment saat ini: {system_info.get('environment', 'Unknown')}"
                )
                return self.create_error_result(error_msg, environment_info=system_info)
            
            # Step 2: Check Drive mount status
            self.update_progress_safe(
                progress_callback, 
                progress_steps[1]['progress'], 
                progress_steps[1]['message'],
                progress_steps[1].get('phase_progress', 0)
            )
            
            # Simulate mount status check
            mount_check_steps = [
                ("Memeriksa status mount Drive...", 40),
                ("Memverifikasi akses...", 80),
                ("Pemeriksaan status selesai", 100)
            ]
            
            for msg, phase_pct in mount_check_steps:
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[1]['progress'] + (progress_steps[2]['progress'] - progress_steps[1]['progress']) * (phase_pct / 100)),
                    msg,
                    int(progress_steps[1].get('phase_progress', 0) + (progress_steps[2].get('phase_progress', 0) - progress_steps[1].get('phase_progress', 0)) * (phase_pct / 100))
                )
            
            # Check if already mounted using EnvironmentManager
            if env_manager.is_drive_mounted:
                self.log("Google Drive sudah terpasang", 'info')
                
                drive_path = env_manager.drive_path
                # Get updated paths
                paths = get_paths_for_environment(is_colab=True, is_drive_mounted=True)
                
                return self.create_success_result(
                    'Google Drive sudah terpasang sebelumnya',
                    already_mounted=True,
                    path=str(drive_path) if drive_path else '/content/drive',
                    paths=paths
                )
            
            # Step 3: Mount Google Drive
            self.update_progress_safe(
                progress_callback, 
                progress_steps[2]['progress'], 
                progress_steps[2]['message'],
                progress_steps[2].get('phase_progress', 0)
            )
            
            # Simulate mounting process
            mount_steps = [
                ("Menyiapkan autentikasi...", 20),
                ("Menghubungkan ke Google Drive...", 50),
                ("Memproses izin...", 80),
                ("Menyelesaikan koneksi...", 100)
            ]
            
            for msg, phase_pct in mount_steps:
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[2]['progress'] + (progress_steps[3]['progress'] - progress_steps[2]['progress']) * (phase_pct / 100)),
                    msg,
                    int(progress_steps[2].get('phase_progress', 0) + (progress_steps[3].get('phase_progress', 0) - progress_steps[2].get('phase_progress', 0)) * (phase_pct / 100))
                )
            
            # Use EnvironmentManager to mount drive
            success, message = env_manager.mount_drive()
            
            # Step 4: Verify mount
            self.update_progress_safe(
                progress_callback, 
                progress_steps[3]['progress'], 
                progress_steps[3]['message'],
                progress_steps[3].get('phase_progress', 0)
            )
            
            if success:
                # Get updated paths after successful mount
                paths = get_paths_for_environment(is_colab=True, is_drive_mounted=True)
                
                # Test write access if drive path is available
                write_access = False
                if env_manager.drive_path:
                    write_access = self.test_write_access(str(env_manager.drive_path))
                
                # Step 5: Complete
                self.update_progress_safe(
                    progress_callback, 
                    progress_steps[4]['progress'], 
                    progress_steps[4]['message'],
                    progress_steps[4].get('phase_progress', 0)
                )
                
                self.log("Google Drive berhasil dipasang dan diverifikasi", 'success')
                
                return self.create_success_result(
                    message,
                    path=str(env_manager.drive_path) if env_manager.drive_path else '/content/drive',
                    paths=paths,
                    write_access=write_access
                )
            else:
                return self.create_error_result(message)
                
        return self.execute_with_error_handling(execute_operation)
    
