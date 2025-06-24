# File: smartcash/ui/setup/env_config/handlers/folder_handler.py
# Deskripsi: Handler untuk folder creation dan management

from pathlib import Path
from typing import Dict, bool, List
from smartcash.common.constants.paths import get_paths_for_environment

class FolderHandler:
    """📁 Handler untuk folder operations"""
    
    # Struktur folder data yang diperlukan
    DATA_FOLDERS = [
        'data', 'data/pretrained', 'data/train', 'data/train/images', 'data/train/labels',
        'data/valid', 'data/valid/images', 'data/valid/labels', 'data/test', 'data/test/images', 
        'data/test/labels', 'data/download', 'data/backup', 'data/augmented', 'data/augmented/images',
        'data/augmented/labels', 'data/preprocessed', 'data/preprocessed/images', 'data/preprocessed/labels',
        'data/invalid', 'data/visualizations'
    ]
    
    SYSTEM_FOLDERS = ['configs', 'logs', 'models', 'exports', 'output', 'runs']
    
    def create_folder_structures(self, logger=None):
        """Buat struktur folder lengkap di Drive dan lokal"""
        paths = get_paths_for_environment(is_colab=True, is_drive_mounted=True)
        
        # Create folders di Drive
        drive_base = "/content/drive/MyDrive/SmartCash"
        if logger:
            logger.info("📁 Membuat folder di Google Drive...")
        drive_results = self._create_folder_structure(drive_base, logger)
        
        # Create folders di lokal
        local_base = "/content"
        if logger:
            logger.info("📁 Membuat folder di storage lokal...")
        local_results = self._create_folder_structure(local_base, logger)
        
        # Log summary
        if logger:
            total_folders = len(drive_results) + len(local_results)
            success_count = sum(drive_results.values()) + sum(local_results.values())
            logger.success(f"📁 Berhasil membuat {success_count}/{total_folders} folder")
    
    def _create_folder_structure(self, base_path: str, logger=None) -> Dict[str, bool]:
        """Buat struktur folder lengkap di path tertentu"""
        results = {}
        all_folders = self.DATA_FOLDERS + self.SYSTEM_FOLDERS
        
        for folder in all_folders:
            folder_path = Path(base_path) / folder
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                results[folder] = True
                if logger:
                    logger.info(f"📁 Folder dibuat: {folder}")
            except Exception as e:
                results[folder] = False
                if logger:
                    logger.error(f"❌ Gagal membuat folder {folder}: {str(e)}")
        
        return results