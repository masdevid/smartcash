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
        """🏗️ Create all required folders and symlinks with optimized performance"""
        result = {
            'created_count': 0,
            'symlinks_count': 0,
            'folders_created': [],
            'symlinks_created': [],
            'source_dirs_created': [],
            'errors': []
        }
        
        try:
            # Batch create source directories
            result['source_dirs_created'] = self._create_source_directories()
            
            # Batch create local directories
            result['folders_created'] = self._create_directories()
            result['created_count'] = len(result['folders_created'])
            
            # Create symlinks with optimized checks
            result['symlinks_created'] = self._create_symlinks()
            result['symlinks_count'] = len(result['symlinks_created'])
            
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
    
    def _create_symlinks(self) -> List[str]:
        """🔗 Create required symlinks with optimized performance"""
        created = []
        
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
                
                # Remove existing target if it exists
                if os.path.lexists(target):
                    try:
                        if os.path.isdir(target) and not os.path.islink(target):
                            # Skip non-empty directories
                            if os.listdir(target):
                                self.logger.warning(f"Skipping non-empty directory: {target}")
                                continue
                            os.rmdir(target)
                        else:
                            os.remove(target)
                    except Exception as e:
                        self.logger.error(f"Failed to remove existing {target}: {e}")
                        continue
                
                # Create the symlink
                os.symlink(source, target)
                created.append(f"{source} -> {target}")
                
            except Exception as e:
                self.logger.error(f"Failed to create symlink {source} -> {target}: {e}")
        
        if created:
            self.logger.success(f"Created {len(created)} symlinks")
                
        return created