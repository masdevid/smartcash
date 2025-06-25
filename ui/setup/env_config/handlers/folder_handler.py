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
        """🏗️ Create all required folders and symlinks with optimized performance"""
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
            # Batch create source directories
            result['source_dirs_created'] = self._create_source_directories()
            
            # Batch create local directories
            result['folders_created'] = self._create_directories()
            result['created_count'] = len(result['folders_created'])
            
            # Create symlinks with optimized checks and backup handling
            result['symlinks_created'], result['backups_created'] = self._create_symlinks()
            result['symlinks_count'] = len(result['symlinks_created'])
            result['backups_count'] = len(result.get('backups_created', []))
            
        except Exception as e:
            error_msg = f"Error in create_required_folders: {str(e)}"
            self.logger.error(error_msg)
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
        
        # Create a temporary directory for initial backups
        with tempfile.TemporaryDirectory(prefix='smartcash_backup_') as temp_backup_dir:
            temp_backups = []
            
            # First pass: Create temporary backups and remove originals
            for source, target in SYMLINK_MAP.items():
                try:
                    # Skip if symlink already exists and points to the correct location
                    if (os.path.islink(target) and 
                        os.path.realpath(target) == os.path.realpath(source)):
                        continue
                    
                    # Create parent directory for target if it doesn't exist
                    target_parent = os.path.dirname(target)
                    if target_parent and not os.path.exists(target_parent):
                        os.makedirs(target_parent, exist_ok=True)
                    
                    # Handle existing target
                    if os.path.lexists(target):
                        try:
                            # Create temporary backup before modifying existing files/directories
                            temp_backup = self._create_temp_backup(target, temp_backup_dir)
                            if not temp_backup:
                                raise Exception("Failed to create temporary backup")
                                
                            temp_backup_path, backup_name = temp_backup
                            temp_backups.append((target, temp_backup_path, backup_name))
                            
                            # Remove existing target
                            if os.path.isdir(target) and not os.path.islink(target):
                                # For directories, only remove if empty
                                try:
                                    os.rmdir(target)
                                except OSError as e:
                                    if e.errno == errno.ENOTEMPTY:
                                        self.logger.warning(
                                            f"Skipping non-empty directory (backup created): {target}"
                                        )
                                        continue
                                    raise
                            else:
                                # For files and symlinks
                                os.remove(target)
                                
                        except Exception as e:
                            self.logger.error(f"Failed to handle existing {target}: {e}")
                            continue
                    
                    # Create the symlink
                    os.symlink(source, target, target_is_directory=os.path.isdir(source))
                    created.append(f"{source} -> {target}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to create symlink {source} -> {target}: {e}")
            
            # If we got here, all symlinks were created successfully
            # Now move temp backups to final location
            for target, temp_backup_path, backup_name in temp_backups:
                final_backup_path = self._move_to_final_backup(
                    temp_backup_path, final_backup_dir, backup_name
                )
                if final_backup_path:
                    backup_info.append((target, final_backup_path))
        
        if created:
            self.logger.success(f"Created {len(created)} symlinks")
            
        if backup_info:
            backup_summary = "\n".join(f"- {src} -> {dst}" for src, dst in backup_info)
            self.logger.info(f"Successfully created {len(backup_info)} backups in {final_backup_dir}:\n{backup_summary}")
                
        return created, backup_info