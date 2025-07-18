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
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize symlink operation.
        
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
            self.update_progress_safe(
                progress_callback, 
                progress_steps[0]['progress'], 
                progress_steps[0]['message'],
                progress_steps[0].get('phase_progress', 0)
            )
            
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
            self.update_progress_safe(
                progress_callback, 
                progress_steps[1]['progress'], 
                progress_steps[1]['message'],
                progress_steps[1].get('phase_progress', 0)
            )
            
            # Create symlinks using SYMLINK_MAP
            symlinks_created = []
            symlinks_failed = []
            total_symlinks = len(SYMLINK_MAP)
            
            for idx, (source, target) in enumerate(SYMLINK_MAP.items(), 1):
                try:
                    # Calculate phase progress within the current step (50-75% of overall progress)
                    phase_progress = int(progress_steps[1]['phase_progress'] + 
                                      (progress_steps[2]['phase_progress'] - progress_steps[1]['phase_progress']) * 
                                      (idx / total_symlinks))
                    
                    # Update progress for current symlink
                    self.update_progress_safe(
                        progress_callback,
                        progress_steps[1]['progress'] + 
                        int((progress_steps[2]['progress'] - progress_steps[1]['progress']) * (idx / total_symlinks)),
                        f"🔗 Creating symlink {idx}/{total_symlinks}: {os.path.basename(target)}",
                        phase_progress
                    )
                    
                    # Handle existing target with backup and content preservation
                    backup_info = self._handle_existing_target(target)
                    
                    # Create parent directory for target if needed
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    
                    # Create symlink
                    os.symlink(source, target)
                    
                    # Copy backed up content to symlinked folder if backup exists
                    if backup_info['backup_created']:
                        self._restore_content_to_symlink(backup_info['backup_path'], target)
                    
                    symlinks_created.append({
                        'source': source,
                        'target': target,
                        'verified': os.path.islink(target) and os.path.exists(target),
                        'backup_info': backup_info
                    })
                    
                    self.log(f"✅ Symlink created: {source} → {target}", 'info')
                    if backup_info['backup_created']:
                        self.log(f"📦 Content restored from backup: {backup_info['files_copied']} files copied", 'info')
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Failed to create symlink {source} → {target}: {str(e)}"
                    self.log(error_msg, 'error')
                    self.log(traceback.format_exc(), 'debug')
                    
                    symlinks_failed.append({
                        'source': source,
                        'target': target,
                        'error': str(e)
                    })
            
            # Step 3: Verify symlinks
            self.update_progress_safe(
                progress_callback, 
                progress_steps[2]['progress'], 
                progress_steps[2]['message'],
                progress_steps[2].get('phase_progress', 0)
            )
            
            # Verify all created symlinks
            verification = self.verify_symlinks_batch(SYMLINK_MAP)
            
            # Update verification status for created symlinks
            for symlink in symlinks_created:
                symlink['verified'] = verification.get('symlink_status', {}).get(symlink['target'], {}).get('valid', False)
            
            # Step 4: Complete
            self.update_progress_safe(
                progress_callback, 
                progress_steps[3]['progress'], 
                progress_steps[3]['message'],
                progress_steps[3].get('phase_progress', 0)
            )
            
            # Prepare result
            verified_count = sum(1 for s in symlinks_created if s.get('verified', False))
            backups_created = sum(1 for s in symlinks_created if s.get('backup_info', {}).get('backup_created', False))
            files_restored = sum(s.get('backup_info', {}).get('files_copied', 0) for s in symlinks_created)
            
            result = {
                'success': len(symlinks_failed) == 0,
                'message': f"Created {len(symlinks_created)}/{total_symlinks} symbolic links " \
                         f"(backed up {backups_created} existing folders, " \
                         f"restored {files_restored} files)",
                'symlinks_created': symlinks_created,
                'symlinks_failed': symlinks_failed,
                'verified_count': verified_count,
                'total_count': len(symlinks_created),
                'backups_created': backups_created,
                'files_restored': files_restored,
                'verification': verification
            }
            
            return result
            
        return self.execute_with_error_handling(execute_operation)
    
    def _handle_existing_target(self, target: str) -> Dict[str, Any]:
        """Handle existing target by backing it up if it's a directory.
        
        Args:
            target: Target path for the symlink
            
        Returns:
            Dictionary with backup information
        """
        backup_info = {
            'backup_created': False,
            'backup_path': None,
            'target_type': None,
            'files_copied': 0
        }
        
        if not os.path.exists(target):
            return backup_info
        
        # If it's already a symlink, just remove it
        if os.path.islink(target):
            os.unlink(target)
            self.log(f"🔗 Removed existing symlink: {target}", 'info')
            backup_info['target_type'] = 'symlink'
            return backup_info
        
        # If it's a directory, back it up and remove the original
        if os.path.isdir(target):
            backup_info['target_type'] = 'directory'
            backup_info['backup_path'] = self._create_backup_directory(target)
            backup_info['backup_created'] = True
            self.log(f"📁 Backed up existing directory: {target} → {backup_info['backup_path']}", 'info')
            # Ensure the target directory is removed after backup
            if os.path.exists(target):
                shutil.rmtree(target)
        
        # If it's a file, back it up
        elif os.path.isfile(target):
            backup_info['target_type'] = 'file'
            backup_info['backup_path'] = self._create_backup_file(target)
            backup_info['backup_created'] = True
            self.log(f"📄 Backed up existing file: {target} → {backup_info['backup_path']}", 'info')
        
        return backup_info
    
    def _create_backup_directory(self, target: str) -> str:
        """Create backup of existing directory.
        
        Args:
            target: Path to directory to backup
            
        Returns:
            Path to backup directory
        """
        import tempfile
        import time
        
        # Create temp directory for backup
        temp_dir = tempfile.mkdtemp(prefix='smartcash_backup_')
        backup_name = f"{os.path.basename(target)}_backup_{int(time.time())}"
        backup_path = os.path.join(temp_dir, backup_name)
        
        # Move the directory to backup location
        shutil.move(target, backup_path)
        
        return backup_path
    
    def _create_backup_file(self, target: str) -> str:
        """Create backup of existing file.
        
        Args:
            target: Path to file to backup
            
        Returns:
            Path to backup file
        """
        import tempfile
        import time
        
        # Create temp directory for backup
        temp_dir = tempfile.mkdtemp(prefix='smartcash_backup_')
        backup_name = f"{os.path.basename(target)}_backup_{int(time.time())}"
        backup_path = os.path.join(temp_dir, backup_name)
        
        # Copy the file to backup location
        shutil.copy2(target, backup_path)
        
        # Remove original file
        os.remove(target)
        
        return backup_path
    
    def _restore_content_to_symlink(self, backup_path: str, target: str) -> None:
        """Restore content from backup to symlinked directory.
        
        Args:
            backup_path: Path to backup directory/file
            target: Path to symlinked directory
        """
        try:
            if not os.path.exists(backup_path):
                return
            
            # If backup is a directory, copy its contents
            if os.path.isdir(backup_path):
                self._copy_directory_contents(backup_path, target)
            
            # If backup is a file, copy it to the symlinked directory
            elif os.path.isfile(backup_path):
                if os.path.isdir(target):
                    dest_file = os.path.join(target, os.path.basename(backup_path).replace('_backup_', '_'))
                    shutil.copy2(backup_path, dest_file)
                    self.log(f"📄 Restored file: {dest_file}", 'info')
            
            # Clean up backup after successful restore
            if os.path.isdir(backup_path):
                shutil.rmtree(backup_path)
            elif os.path.isfile(backup_path):
                os.remove(backup_path)
            
        except Exception as e:
            self.log(f"⚠️ Failed to restore content from backup: {str(e)}", 'warning')
    
    def _copy_directory_contents(self, source_dir: str, dest_dir: str) -> int:
        """Copy contents of source directory to destination directory.
        
        Args:
            source_dir: Source directory path
            dest_dir: Destination directory path
            
        Returns:
            Number of files copied
        """
        files_copied = 0
        
        try:
            if not os.path.exists(dest_dir):
                self.log(f"⚠️ Destination directory does not exist: {dest_dir}", 'warning')
                return 0
            
            for root, _, files in os.walk(source_dir):
                # Calculate relative path from source root
                rel_path = os.path.relpath(root, source_dir)
                
                # Create corresponding directory in destination
                if rel_path != '.':
                    dest_subdir = os.path.join(dest_dir, rel_path)
                    os.makedirs(dest_subdir, exist_ok=True)
                
                # Copy files
                for file in files:
                    src_file = os.path.join(root, file)
                    if rel_path == '.':
                        dest_file = os.path.join(dest_dir, file)
                    else:
                        dest_file = os.path.join(dest_dir, rel_path, file)
                    
                    try:
                        shutil.copy2(src_file, dest_file)
                        files_copied += 1
                        self.log(f"📄 Copied: {file} → {os.path.relpath(dest_file, dest_dir)}", 'info')
                    except Exception as e:
                        self.log(f"⚠️ Failed to copy {file}: {str(e)}", 'warning')
            
            self.log(f"✅ Successfully copied {files_copied} files from backup", 'info')
            
        except Exception as e:
            self.log(f"⚠️ Error copying directory contents: {str(e)}", 'warning')
        
        return files_copied
    
    def _cleanup_backup_files(self, symlinks_created: list, symlinks_failed: list) -> None:  # noqa: ARG002
        """Clean up temporary backup files that weren't properly restored.
        
        Args:
            symlinks_created: List of successfully created symlinks
            symlinks_failed: List of failed symlinks
        """
        try:
            # Clean up backups from failed operations
            for failed in symlinks_failed:
                if 'backup_info' in failed and failed['backup_info']['backup_created']:
                    backup_path = failed['backup_info']['backup_path']
                    if backup_path and os.path.exists(backup_path):
                        try:
                            if os.path.isdir(backup_path):
                                shutil.rmtree(backup_path)
                            else:
                                os.remove(backup_path)
                            self.log(f"🧹 Cleaned up backup: {backup_path}", 'info')
                        except Exception as e:
                            self.log(f"⚠️ Failed to cleanup backup {backup_path}: {str(e)}", 'warning')
            
            # Clean up any remaining temporary directories
            import tempfile
            temp_dir = tempfile.gettempdir()
            for item in os.listdir(temp_dir):
                if item.startswith('smartcash_backup_'):
                    backup_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isdir(backup_path):
                            shutil.rmtree(backup_path)
                        else:
                            os.remove(backup_path)
                        self.log(f"🧹 Cleaned up orphaned backup: {backup_path}", 'info')
                    except Exception as e:
                        self.log(f"⚠️ Failed to cleanup orphaned backup {backup_path}: {str(e)}", 'warning')
                        
        except Exception as e:
            self.log(f"⚠️ Error during backup cleanup: {str(e)}", 'warning')
    
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
        _, failed_dirs = self.create_directories_batch(missing_dirs)
        if failed_dirs:
            self.log(f"Failed to create {len(failed_dirs)} directories", 'warning')