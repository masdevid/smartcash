"""
File: smartcash/ui/setup/env_config/handlers/folder_handler.py
Deskripsi: Handler untuk membuat folder dan symlink yang diperlukan
"""

import os
from typing import Dict, Any, List, Optional
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
        """🏗️ Create all required folders and symlinks"""
        result = {
            'created_count': 0,
            'symlinks_count': 0,
            'folders_created': [],
            'symlinks_created': [],
            'source_dirs_created': [],
            'errors': []
        }
        
        try:
            # Create source directories in Google Drive first
            source_dirs_created = self._create_source_directories()
            result['source_dirs_created'] = source_dirs_created
            
            # Create local directories
            folders_created = self._create_directories()
            result['created_count'] = len(folders_created)
            result['folders_created'] = folders_created
            
            # Create symlinks
            symlinks_created = self._create_symlinks()
            result['symlinks_count'] = len(symlinks_created)
            result['symlinks_created'] = symlinks_created
            
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    def _create_source_directories(self) -> List[str]:
        """📂 Buat direktori sumber di Google Drive"""
        created = []
        
        for folder_path in SOURCE_DIRECTORIES:
            try:
                os.makedirs(folder_path, exist_ok=True)
                self.logger.success(f"Created source directory: {folder_path}")
                created.append(folder_path)
            except Exception as e:
                self.logger.error(f"Failed to create source directory {folder_path}: {e}")
                
        return created
        
    def _create_directories(self) -> List[str]:
        """📂 Buat direktori lokal yang diperlukan"""
        created = []
        
        for folder_path in REQUIRED_FOLDERS:
            try:
                os.makedirs(folder_path, exist_ok=True)
                self.logger.success(f"Created directory: {folder_path}")
                created.append(folder_path)
            except Exception as e:
                self.logger.error(f"Failed to create directory {folder_path}: {e}")
                
        return created
    
    def _create_symlinks(self) -> List[str]:
        """🔗 Buat symlink yang diperlukan"""
        created = []
        
        for source, target in SYMLINK_MAP.items():
            try:
                # Create parent directory for target if it doesn't exist
                target_parent = os.path.dirname(target)
                if not os.path.exists(target_parent):
                    os.makedirs(target_parent, exist_ok=True)
                
                # Check if target exists and handle it
                if os.path.exists(target) or os.path.islink(target):
                    if os.path.islink(target):
                        os.remove(target)
                        self.logger.debug(f"Removed existing symlink: {target}")
                    elif os.path.isfile(target):
                        os.remove(target)
                        self.logger.debug(f"Removed existing file: {target}")
                    elif os.path.isdir(target):
                        # Skip if target is a non-empty directory
                        if os.listdir(target):
                            self.logger.warning(f"Skipping non-empty directory: {target}")
                            continue
                        os.rmdir(target)
                        self.logger.debug(f"Removed empty directory: {target}")
                
                # Create symlink if it doesn't exist
                if not os.path.lexists(target):
                    os.symlink(source, target)
                    self.logger.success(f"Created symlink: {source} -> {target}")
                    created.append(f"{source} -> {target}")
                else:
                    self.logger.info(f"Symlink already exists: {source} -> {target}")
                
            except Exception as e:
                error_msg = f"Failed to create symlink {source} -> {target}: {str(e)}"
                self.logger.error(error_msg)
                raise  # Re-raise to handle in the calling function
                
        return created