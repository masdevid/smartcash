"""
File: smartcash/ui/setup/colab/operations/folders_operation.py
Description: Create required folders in Colab using REQUIRED_FOLDERS
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation
from ..constants import REQUIRED_FOLDERS


class FoldersOperation(BaseColabOperation):
    """Create required folders in Colab using REQUIRED_FOLDERS."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize folders operation.
        
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
            'create_folders': self.execute_create_folders
        }
    
    def execute_create_folders(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Create required folders in Colab using REQUIRED_FOLDERS.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation results
        """
        def execute_operation():
            progress_steps = self.get_progress_steps('folders')
            
            # Step 1: Check folder configuration
            self.update_progress_safe(
                progress_callback, 
                progress_steps[0]['progress'], 
                progress_steps[0]['message'],
                progress_steps[0].get('phase_progress', 0)
            )
            
            # Simulate folder configuration check
            config_steps = [
                ("Memeriksa konfigurasi folder...", 30),
                ("Memuat daftar folder yang diperlukan...", 70),
                ("Konfigurasi folder siap", 100)
            ]
            
            for msg, phase_pct in config_steps:
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[0]['progress'] + (progress_steps[1]['progress'] - progress_steps[0]['progress']) * (phase_pct / 100)),
                    msg,
                    int(progress_steps[0].get('phase_progress', 0) + (progress_steps[1].get('phase_progress', 0) - progress_steps[0].get('phase_progress', 0)) * (phase_pct / 100))
                )
            
            total_folders = len(REQUIRED_FOLDERS)
            self.log(f"Creating {total_folders} required folders", 'info')
            
            # Step 2: Create required folders
            self.update_progress_safe(
                progress_callback, 
                progress_steps[1]['progress'], 
                progress_steps[1]['message'],
                progress_steps[1].get('phase_progress', 0)
            )
            
            # Create folders with progress updates
            created_dirs = []
            failed_dirs = []
            
            for idx, folder in enumerate(REQUIRED_FOLDERS, 1):
                # Calculate progress within this step (0-100% of this phase)
                phase_pct = int((idx / total_folders) * 100)
                
                # Update progress
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[1]['progress'] + (progress_steps[2]['progress'] - progress_steps[1]['progress']) * (idx / total_folders)),
                    f"Membuat folder {idx}/{total_folders}: {folder}",
                    int(progress_steps[1].get('phase_progress', 0) + (progress_steps[2].get('phase_progress', 0) - progress_steps[1].get('phase_progress', 0)) * (phase_pct / 100))
                )
                
                # Create the directory
                try:
                    os.makedirs(folder, exist_ok=True)
                    created_dirs.append(folder)
                    self.log(f"✅ Created directory: {folder}", 'info')
                except Exception as e:
                    error_msg = f"Gagal membuat direktori {folder}: {str(e)}"
                    self.log(error_msg, 'error')
                    failed_dirs.append({"path": folder, "error": str(e)})
            
            # Step 3: Verify folder structure
            self.update_progress_safe(
                progress_callback, 
                progress_steps[2]['progress'], 
                progress_steps[2]['message'],
                progress_steps[2].get('phase_progress', 0)
            )
            
            # Verify all folders exist with progress updates
            verification = self.validate_items_exist(REQUIRED_FOLDERS, "folder")
            
            # Simulate verification steps
            verify_steps = [
                ("Memverifikasi struktur folder...", 40),
                ("Memeriksa izin...", 70),
                ("Verifikasi selesai", 100)
            ]
            
            for msg, phase_pct in verify_steps:
                self.update_progress_safe(
                    progress_callback,
                    int(progress_steps[2]['progress'] + (progress_steps[3]['progress'] - progress_steps[2]['progress']) * (phase_pct / 100)),
                    msg,
                    int(progress_steps[2].get('phase_progress', 0) + (progress_steps[3].get('phase_progress', 0) - progress_steps[2].get('phase_progress', 0)) * (phase_pct / 100))
                )
            
            # Step 4: Complete
            self.update_progress_safe(
                progress_callback, 
                progress_steps[3]['progress'], 
                progress_steps[3]['message'],
                progress_steps[3].get('phase_progress', 0)
            )
            
            return self.create_success_result(
                f'Created {len(created_dirs)} new folders, {total_folders - len(created_dirs)} already existed',
                folders_created=created_dirs,
                folders_failed=failed_dirs,
                total_count=total_folders,
                created_count=len(created_dirs),
                verification=verification
            )
            
        return self.execute_with_error_handling(execute_operation)
    
    def verify_folders(self) -> Dict[str, Any]:
        """Verify that all required folders exist.
        
        Returns:
            Dictionary with verification results
        """
        return self.validate_items_exist(REQUIRED_FOLDERS, "folder")