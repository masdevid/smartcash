"""
File: smartcash/ui/setup/env_config/handlers/folder_handler.py
Deskripsi: Handler untuk membuat folder dan symlink yang diperlukan
"""

import os
import shutil
import errno
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from smartcash.ui.setup.env_config.constants import REQUIRED_FOLDERS, SYMLINK_MAP, SOURCE_DIRECTORIES

class FolderHandler:
    """📁 Handler untuk folder dan symlink management"""
    
    def __init__(self, logger=None):
        """Initialize FolderHandler with optional logger"""
        self.logger = logger or self._create_dummy_logger()
        
    def _create_dummy_logger(self):
        """📝 Create dummy logger fallback"""
        class DummyLogger:
            def debug(self, msg): print(f"🔍 {msg}")
            def info(self, msg): print(f"ℹ️ {msg}")
            def success(self, msg): print(f"✅ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
            def error(self, msg): print(f"❌ {msg}")
        return DummyLogger()
        
    def create_required_folders(self) -> Dict[str, Any]:
        """Ἵ7️ Create all required folders and symlinks with optimized performance
        
        Note: Creates symlinks first to prevent directory creation conflicts
        """
        result = {
            'created_count': 0,
            'symlinks_count': 0,
            'backups_count': 0,
            'folders_created': [],
            'symlinks_created': [],
            'backups_created': [],
            'source_dirs_created': [],
            'errors': []
        }
        
        try:
            # 1. First create source directories in Google Drive
            result['source_dirs_created'] = self._create_source_directories()
            
            # 2. Create symlinks before creating local directories to prevent conflicts
            result['symlinks_created'], result['backups_created'] = self._create_symlinks()
            result['symlinks_count'] = len(result['symlinks_created'])
            result['backups_count'] = len(result.get('backups_created', []))
            
            # 3. Finally, create any remaining required local directories
            # Filter out directories that are symlink targets to avoid conflicts
            symlink_targets = set()
            for target in SYMLINK_MAP.values():
                symlink_targets.add(os.path.normpath(target))
            
            # Only create required folders that aren't symlink targets
            required_folders = [
                f for f in REQUIRED_FOLDERS 
                if os.path.normpath(f) not in symlink_targets
            ]
            
            # Temporarily replace REQUIRED_FOLDERS to avoid modifying the constant
            original_required_folders = REQUIRED_FOLDERS.copy()
            try:
                # Use a local copy of the constant
                import sys
                current_module = sys.modules[__name__]
                setattr(current_module, 'REQUIRED_FOLDERS', required_folders)
                
                # Now create the directories
                result['folders_created'] = self._create_directories()
                result['created_count'] = len(result['folders_created'])
            finally:
                # Restore the original constant
                setattr(current_module, 'REQUIRED_FOLDERS', original_required_folders)
            
        except Exception as e:
            error_msg = f"Error in create_required_folders: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result['errors'].append(error_msg)
        
        return result
    
    def _create_source_directories(self) -> List[str]:
        """📂 Create source directories in Google Drive with optimized performance"""
        created = []
        
        for folder_path in SOURCE_DIRECTORIES:
            try:
                # Skip if directory already exists
                if os.path.exists(folder_path):
                    continue
                    
                os.makedirs(folder_path, exist_ok=True)
                created.append(folder_path)
                
            except Exception as e:
                self.logger.error(f"Failed to create source directory {folder_path}: {e}")
                
        if created:
            self.logger.success(f"Created {len(created)} source directories")
            
        return created
        
    def _create_directories(self) -> List[str]:
        """📂 Create required local directories with optimized performance"""
        created = []
        
        for folder_path in REQUIRED_FOLDERS:
            try:
                # Skip if directory already exists
                if os.path.exists(folder_path):
                    continue
                    
                os.makedirs(folder_path, exist_ok=True)
                created.append(folder_path)
                
            except Exception as e:
                self.logger.error(f"Failed to create directory {folder_path}: {e}")
                
        if created:
            self.logger.success(f"Created {len(created)} required directories")
            
        return created
    
    def _create_temp_backup(self, path: str, temp_dir: str) -> Optional[Tuple[str, str]]:
        """Create a temporary backup of the given path.
        
        Args:
            path: Path to the file/directory to back up
            temp_dir: Temporary directory to store the backup
            
        Returns:
            Tuple of (temp_backup_path, final_backup_name) if successful, None otherwise
        """
        try:
            # Create temp directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate backup name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{os.path.basename(path)}_backup_{timestamp}"
            temp_backup_path = os.path.join(temp_dir, backup_name)
            
            # Create the backup in temp location
            if os.path.isfile(path):
                # Backup file
                shutil.copy2(path, temp_backup_path)
            elif os.path.isdir(path):
                # Backup directory
                if os.path.islink(path):
                    # Handle symlink to directory
                    link_target = os.readlink(path)
                    os.symlink(link_target, temp_backup_path)
                else:
                    shutil.copytree(path, temp_backup_path, symlinks=True)
            
            self.logger.debug(f"Created temporary backup at: {temp_backup_path}")
            return temp_backup_path, backup_name
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary backup of {path}: {e}")
            return None
            
    def _move_to_final_backup(self, temp_backup_path: str, final_backup_dir: str, backup_name: str) -> Optional[str]:
        """Move a backup from temp location to final backup directory.
        
        Args:
            temp_backup_path: Path to the temporary backup
            final_backup_dir: Final backup directory
            backup_name: Name of the backup file/directory
            
        Returns:
            Final backup path if successful, None otherwise
        """
        try:
            # Create final backup directory if it doesn't exist
            os.makedirs(final_backup_dir, exist_ok=True)
            
            final_backup_path = os.path.join(final_backup_dir, backup_name)
            
            # Handle case where final backup already exists (should be rare due to timestamps)
            if os.path.exists(final_backup_path):
                # Add a random suffix to avoid conflicts
                import uuid
                base, ext = os.path.splitext(backup_name)
                backup_name = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
                final_backup_path = os.path.join(final_backup_dir, backup_name)
            
            # Move the backup to final location
            shutil.move(temp_backup_path, final_backup_path)
            self.logger.info(f"Moved backup to final location: {final_backup_path}")
            return final_backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to move backup to final location: {e}")
            return None

    def _create_symlinks(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """🔗 Create required symlinks with optimized performance and backup handling
        
        Returns:
            Tuple of (created_symlinks, backup_info) where:
            - created_symlinks: List of created symlink paths
            - backup_info: List of tuples (original_path, final_backup_path) for successful backups
        """
        import tempfile
        import shutil
        
        created = []
        backup_info = []
        final_backup_dir = os.path.join(os.path.expanduser('~'), 'data', 'backup')
        
        # Ensure final backup directory exists
        os.makedirs(final_backup_dir, exist_ok=True)
        
        for source, target in SYMLINK_MAP.items():
            try:
                # Skip if source doesn't exist
                if not os.path.exists(source):
                    self.logger.warning(f"Source directory {source} does not exist, skipping")
                    continue
                    
                # If target exists and is not a symlink
                if os.path.exists(target) and not os.path.islink(target):
                    # If target is a directory and empty, remove it
                    if os.path.isdir(target) and not os.listdir(target):
                        try:
                            os.rmdir(target)
                            self.logger.info(f"Removed empty directory: {target}")
                        except OSError as e:
                            self.logger.error(f"Failed to remove empty directory {target}: {e}")
                            continue
                    # If target is a directory with content, move it to backup
                    elif os.path.isdir(target):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_name = f"{os.path.basename(target)}_backup_{timestamp}"
                        final_backup_path = os.path.join(final_backup_dir, backup_name)
                        
                        try:
                            shutil.move(target, final_backup_path)
                            self.logger.info(f"Moved existing directory {target} to backup at {final_backup_path}")
                            backup_info.append((target, final_backup_path))
                        except Exception as e:
                            self.logger.error(f"Failed to backup {target} to {final_backup_path}: {e}")
                            continue
                
                # Create parent directory if it doesn't exist
                os.makedirs(os.path.dirname(target), exist_ok=True)
                
                # Remove existing symlink if it exists and points to the wrong location
                if os.path.islink(target):
                    try:
                        current_target = os.path.realpath(target)
                        expected_target = os.path.realpath(source)
                        if current_target == expected_target:
                            self.logger.info(f"Symlink already exists and points to correct location: {target}")
                            created.append(target)
                            continue
                        else:
                            os.remove(target)
                            self.logger.info(f"Removed incorrect symlink: {target} (was pointing to {current_target})")
                    except OSError as e:
                        self.logger.error(f"Failed to verify/remove existing symlink {target}: {e}")
                        continue
                
                # Create the symlink
                try:
                    os.symlink(source, target)
                    created.append(target)
                    self.logger.success(f"Created symlink: {target} -> {source}")
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        self.logger.warning(f"Symlink {target} already exists but couldn't be verified")
                    else:
                        self.logger.error(f"Failed to create symlink {target} -> {source}: {e}")
                        
            except Exception as e:
                self.logger.error(f"Error processing symlink {source} -> {target}: {e}")
                continue
                
        if created:
            self.logger.success(f"Created {len(created)} symlinks")
            
        if backup_info:
            backup_summary = "\n".join(f"- {src} -> {dst}" for src, dst in backup_info)
            self.logger.info(f"Successfully created {len(backup_info)} backups in {final_backup_dir}:\n{backup_summary}")
                
        return created, backup_info