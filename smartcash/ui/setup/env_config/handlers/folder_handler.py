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
    
    def _create_backup(self, path: str, backup_dir: str) -> Optional[str]:
        """Create a backup of the given path in the backup directory.
        
        Args:
            path: Path to the file/directory to back up
            backup_dir: Directory to store the backup
            
        Returns:
            Path to the backup if successful, None otherwise
        """
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{os.path.basename(path)}_backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            if os.path.isfile(path):
                # Backup file
                shutil.copy2(path, backup_path)
            elif os.path.isdir(path):
                # Backup directory
                if os.path.islink(path):
                    # Handle symlink to directory
                    link_target = os.readlink(path)
                    os.symlink(link_target, backup_path)
                else:
                    shutil.copytree(path, backup_path, symlinks=True)
            
            self.logger.info(f"Created backup at: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup of {path}: {e}")
            return None

    def _create_symlinks(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """🔗 Create required symlinks with optimized performance and backup handling
        
        Returns:
            Tuple of (created_symlinks, backup_info) where:
            - created_symlinks: List of created symlink paths
            - backup_info: List of tuples (original_path, backup_path) for successful backups
        """
        created = []
        backup_info = []
        backup_dir = os.path.join(os.path.expanduser('~'), 'data', 'backup')
        
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
                        # Create backup before modifying existing files/directories
                        backup_path = self._create_backup(target, backup_dir)
                        if backup_path:
                            backup_info.append((target, backup_path))
                        
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
        
        if created:
            self.logger.success(f"Created {len(created)} symlinks")
            
        if backup_info:
            backup_summary = "\n".join(f"- {src} -> {dst}" for src, dst in backup_info)
            self.logger.info(f"Created {len(backup_info)} backups:\n{backup_summary}")
                
        return created, backup_info