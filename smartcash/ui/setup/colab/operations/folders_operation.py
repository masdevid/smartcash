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
            
            total_folders = len(REQUIRED_FOLDERS)
            self.log(f"Creating {total_folders} required folders", 'info')
            
            # Step 2: Create required folders
            self.update_progress_safe(
                progress_callback, 
                progress_steps[1]['progress'], 
                progress_steps[1]['message'],
                progress_steps[1].get('phase_progress', 0)
            )
            
            created_dirs, failed_dirs = self.create_directories_batch(REQUIRED_FOLDERS)
            
            # Step 3: Verify folder structure
            self.update_progress_safe(
                progress_callback, 
                progress_steps[2]['progress'], 
                progress_steps[2]['message'],
                progress_steps[2].get('phase_progress', 0)
            )
            
            # Verify all folders exist
            verification = self.validate_items_exist(REQUIRED_FOLDERS, "folder")
            
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