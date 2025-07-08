"""
File: smartcash/ui/setup/colab/operations/symlink_operation.py
Description: Create symbolic links using SYMLINK_MAP with detailed progress
"""

import os
import shutil
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import SYMLINK_MAP, SOURCE_DIRECTORIES


class SymlinkOperation(OperationHandler):
    """Create symbolic links using SYMLINK_MAP with detailed progress."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize symlink operation.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='symlink_operation',
            parent_module='colab',
            **kwargs
        )
        self.config = config
    
    def initialize(self) -> None:
        """Initialize the symlink operation."""
        self.logger.info("🚀 Initializing symlink operation")
        # No specific initialization needed for symlink operation
        self.logger.info("✅ Symlink operation initialization complete")
    
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
        try:
            env_config = self.config.get('environment', {})
            
            if env_config.get('type') != 'colab':
                return {
                    'success': False,
                    'error': 'Symbolic links are only created in Colab environment'
                }
            
            if progress_callback:
                progress_callback(5, "🔍 Checking Drive mount...")
            
            # Check if Drive is mounted
            if not os.path.exists('/content/drive/MyDrive'):
                return {
                    'success': False,
                    'error': 'Google Drive must be mounted before creating symlinks'
                }
            
            if progress_callback:
                progress_callback(10, "📋 Checking source directories...")
            
            # Check source directories exist
            missing_sources = self._check_source_directories()
            if missing_sources:
                self.log(f"Creating {len(missing_sources)} missing source directories", 'info')
                self._create_missing_directories(missing_sources)
            
            if progress_callback:
                progress_callback(25, f"📁 Creating {len(SYMLINK_MAP)} symbolic links...")
            
            # Create symlinks using SYMLINK_MAP
            symlinks_created = []
            symlinks_failed = []
            total_symlinks = len(SYMLINK_MAP)
            
            for i, (source, target) in enumerate(SYMLINK_MAP.items()):
                current_progress = 25 + ((i + 1) / total_symlinks) * 65  # 25% to 90%
                
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
                    
                    if progress_callback:
                        progress_callback(current_progress, f"✅ Created: {os.path.basename(target)}")
                    
                except Exception as e:
                    symlinks_failed.append({
                        'source': source,
                        'target': target,
                        'error': str(e)
                    })
                    self.log(f"❌ Symlink failed: {source} → {target}: {str(e)}", 'error')
                    
                    if progress_callback:
                        progress_callback(current_progress, f"❌ Failed: {os.path.basename(target)}")
            
            if progress_callback:
                progress_callback(95, "🔍 Verifying symlinks...")
            
            # Final verification
            verified_count = sum(1 for sl in symlinks_created if sl['verified'])
            
            if progress_callback:
                progress_callback(100, f"✅ Created {verified_count}/{total_symlinks} symlinks")
            
            success = len(symlinks_failed) == 0
            
            return {
                'success': success,
                'symlinks_created': symlinks_created,
                'symlinks_failed': symlinks_failed,
                'verified_count': verified_count,
                'total_count': total_symlinks,
                'message': f'Created {verified_count}/{total_symlinks} symbolic links'
            }
            
        except Exception as e:
            self.log(f"Symlink operation failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Symlink creation failed: {str(e)}'
            }
    
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
        for source_dir in missing_dirs:
            try:
                os.makedirs(source_dir, exist_ok=True)
                self.log(f"Created: {source_dir}", 'info')
            except Exception as e:
                self.log(f"Failed to create {source_dir}: {str(e)}", 'error')