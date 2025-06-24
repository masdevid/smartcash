"""
File: smartcash/ui/setup/env_config/handlers/folder_handler.py
Deskripsi: Handler untuk membuat folder dan symlink yang diperlukan
"""

import os
from typing import Dict, Any, List
from smartcash.ui.setup.env_config.constants import REQUIRED_FOLDERS, SYMLINK_MAP

class FolderHandler:
    """📁 Handler untuk folder dan symlink management"""
    
    def create_required_folders(self) -> Dict[str, Any]:
        """🏗️ Buat semua folder dan symlink yang diperlukan"""
        result = {
            'created_count': 0,
            'symlinks_count': 0,
            'folders_created': [],
            'symlinks_created': [],
            'errors': []
        }
        
        try:
            # Create directories
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
    
    def _create_directories(self) -> List[str]:
        """📂 Buat direktori yang diperlukan"""
        created = []
        
        for folder_path in REQUIRED_FOLDERS:
            try:
                os.makedirs(folder_path, exist_ok=True)
                created.append(folder_path)
            except Exception as e:
                print(f"⚠️ Failed to create {folder_path}: {e}")
                
        return created
    
    def _create_symlinks(self) -> List[str]:
        """🔗 Buat symlink yang diperlukan"""
        created = []
        
        for source, target in SYMLINK_MAP.items():
            try:
                if os.path.exists(source) and not os.path.exists(target):
                    os.symlink(source, target)
                    created.append(f"{source} -> {target}")
            except Exception as e:
                print(f"⚠️ Failed to create symlink {source} -> {target}: {e}")
                
        return created