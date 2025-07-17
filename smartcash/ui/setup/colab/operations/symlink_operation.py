"""
File: smartcash/ui/setup/colab/operations/symlink_operation.py
Description: Create symbolic links using SYMLINK_MAP with detailed progress
"""

import os
import shutil
from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation
from ..constants import SYMLINK_MAP, SOURCE_DIRECTORIES


class SymlinkOperation(BaseColabOperation):
    """Create symbolic links using SYMLINK_MAP with detailed progress."""
    
    def __init__(self, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize symlink operation.
        
        Args:
            config: Configuration dictionary
            operation_container: Optional operation container for UI integration
            **kwargs: Additional arguments
        """
        super().__init__('symlink_operation', config, operation_container, **kwargs)
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'create_symlinks': self.execute_create_symlinks
        }
    
    def execute_create_symlinks(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Create symbolic links using SYMLINK_MAP with detailed progress.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation results
        """
        def execute_operation():
            progress_steps = self.get_progress_steps('symlink')
            
            # Step 1: Check symlink configuration
            self.update_progress_safe(progress_callback, progress_steps[0]['progress'], progress_steps[0]['message'])
            
            env_config = self.config.get('environment', {})
            
            if env_config.get('type') != 'colab':
                return self.create_error_result('Symbolic links are only created in Colab environment')
            
            # Check if Drive is mounted
            if not os.path.exists('/content/drive/MyDrive'):
                return self.create_error_result('Google Drive must be mounted before creating symlinks')
            
            # Check source directories exist
            missing_sources = self._check_source_directories()
            if missing_sources:
                self.log(f"Creating {len(missing_sources)} missing source directories", 'info')
                self._create_missing_directories(missing_sources)
            
            # Step 2: Create symlinks
            self.update_progress_safe(progress_callback, progress_steps[1]['progress'], progress_steps[1]['message'])
            
            # Create symlinks using SYMLINK_MAP
            symlinks_created = []
            symlinks_failed = []
            total_symlinks = len(SYMLINK_MAP)
            
            for source, target in SYMLINK_MAP.items():
                try:
                    # Remove target if it exists and is not a symlink
                    if os.path.exists(target) and not os.path.islink(target):
                        if os.path.isdir(target):
                            shutil.rmtree(target)
                        else:
                            os.remove(target)
                        self.log(f"Removed existing target: {target}", 'info')
                    
                    # Remove existing symlink if it exists
                    elif os.path.islink(target):
                        os.unlink(target)
                        self.log(f"Removed existing symlink: {target}", 'info')
                    
                    # Create parent directory for target if needed
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    
                    # Create symlink
                    os.symlink(source, target)
                    symlinks_created.append({
                        'source': source,
                        'target': target,
                        'verified': os.path.islink(target) and os.path.exists(target)
                    })
                    
                    self.log(f"✅ Symlink created: {source} → {target}", 'info')
                    
                except Exception as e:
                    symlinks_failed.append({
                        'source': source,
                        'target': target,
                        'error': str(e)
                    })
                    self.log(f"❌ Symlink failed: {source} → {target}: {str(e)}", 'error')
            
            # Step 3: Verify symlinks
            self.update_progress_safe(progress_callback, progress_steps[2]['progress'], progress_steps[2]['message'])
            
            # Final verification using base class method
            verification = self.verify_symlinks_batch(SYMLINK_MAP)
            
            # Step 4: Complete
            self.update_progress_safe(progress_callback, progress_steps[3]['progress'], progress_steps[3]['message'])
            
            verified_count = sum(1 for sl in symlinks_created if sl['verified'])
            success = len(symlinks_failed) == 0
            
            return self.create_success_result(
                f'Created {verified_count}/{total_symlinks} symbolic links',
                symlinks_created=symlinks_created,
                symlinks_failed=symlinks_failed,
                verified_count=verified_count,
                total_count=total_symlinks,
                verification=verification
            )
            
        return self.execute_with_error_handling(execute_operation)
    
    def _check_source_directories(self) -> list:
        """Check if source directories exist and return missing ones.
        
        Returns:
            List of missing source directories
        """
        missing_sources = []
        for source_dir in SOURCE_DIRECTORIES:
            if not os.path.exists(source_dir):
                missing_sources.append(source_dir)
                self.log(f"Missing source directory: {source_dir}", 'warning')
        return missing_sources
    
    def _create_missing_directories(self, missing_dirs: list) -> None:
        """Create missing source directories.
        
        Args:
            missing_dirs: List of missing directories to create
        """
        created_dirs, failed_dirs = self.create_directories_batch(missing_dirs)
        if failed_dirs:
            self.log(f"Failed to create {len(failed_dirs)} directories", 'warning')