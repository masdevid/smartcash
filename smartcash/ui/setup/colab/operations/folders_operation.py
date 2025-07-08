"""
File: smartcash/ui/setup/colab/operations/folders_operation.py
Description: Create required folders in Colab using REQUIRED_FOLDERS
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import REQUIRED_FOLDERS


class FoldersOperation(OperationHandler):
    """Create required folders in Colab using REQUIRED_FOLDERS."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize folders operation.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='folders_operation',
            parent_module='colab',
            **kwargs
        )
        self.config = config
    
    def initialize(self) -> None:
        """Initialize the folders operation."""
        self.logger.info("🚀 Initializing folders operation")
        # No specific initialization needed for folders operation
        self.logger.info("✅ Folders operation initialization complete")
    
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
        try:
            if progress_callback:
                progress_callback(5, "📋 Planning folder structure...")
            
            folders_created = []
            folders_failed = []
            total_folders = len(REQUIRED_FOLDERS)
            
            self.log(f"Creating {total_folders} required folders", 'info')
            
            for i, folder_path in enumerate(REQUIRED_FOLDERS):
                current_progress = 5 + ((i + 1) / total_folders) * 90  # 5% to 95%
                folder_name = os.path.basename(folder_path)
                
                try:
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path, exist_ok=True)
                        folders_created.append(folder_path)
                        self.log(f"✅ Created folder: {folder_path}", 'info')
                        
                        if progress_callback:
                            progress_callback(current_progress, f"✅ Created: {folder_name}")
                    else:
                        self.log(f"ℹ️ Folder exists: {folder_path}", 'debug')
                        
                        if progress_callback:
                            progress_callback(current_progress, f"ℹ️ Exists: {folder_name}")
                    
                except Exception as e:
                    folders_failed.append({
                        'folder': folder_path,
                        'error': str(e)
                    })
                    self.log(f"❌ Failed to create folder: {folder_path}: {str(e)}", 'error')
                    
                    if progress_callback:
                        progress_callback(current_progress, f"❌ Failed: {folder_name}")
            
            if progress_callback:
                progress_callback(100, f"✅ Processed {total_folders} folders")
            
            success = len(folders_failed) == 0
            
            return {
                'success': success,
                'folders_created': folders_created,
                'folders_failed': folders_failed,
                'total_count': total_folders,
                'created_count': len(folders_created),
                'message': f'Created {len(folders_created)} new folders, {total_folders - len(folders_created)} already existed'
            }
            
        except Exception as e:
            self.log(f"Folder creation failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Directory creation failed: {str(e)}'
            }
    
    def verify_folders(self) -> Dict[str, Any]:
        """Verify that all required folders exist.
        
        Returns:
            Dictionary with verification results
        """
        missing_folders = []
        existing_folders = []
        
        for folder_path in REQUIRED_FOLDERS:
            if os.path.exists(folder_path):
                existing_folders.append(folder_path)
            else:
                missing_folders.append(folder_path)
        
        return {
            'all_exist': len(missing_folders) == 0,
            'existing_folders': existing_folders,
            'missing_folders': missing_folders,
            'total_count': len(REQUIRED_FOLDERS),
            'existing_count': len(existing_folders),
            'missing_count': len(missing_folders)
        }