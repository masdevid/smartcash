"""
File: smartcash/ui/setup/env_config/handlers/directory_manager.py
Deskripsi: Handler untuk manajemen direktori - SRP untuk directory operations
"""

import os
import shutil
from typing import Dict, Any, List
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.common.environment import get_environment_manager

class DirectoryManager:
    """Handler untuk manajemen direktori - fokus hanya pada directory operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi manager dengan UI components
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "directory_manager")
        self.env_manager = get_environment_manager()
    
    def create_local_directories(self) -> bool:
        """
        Buat direktori lokal yang diperlukan
        
        Returns:
            Success status
        """
        required_dirs = ['data', 'exports', 'logs', 'models', 'output', 'configs']
        
        if self._is_colab():
            base_path = Path("/content")
        else:
            base_path = self.env_manager.base_dir
        
        try:
            created_count = 0
            for dir_name in required_dirs:
                dir_path = base_path / dir_name
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"📁 Dibuat direktori: {dir_name}")
                    created_count += 1
            
            if created_count > 0:
                self.logger.success(f"✅ Dibuat {created_count} direktori lokal")
            else:
                self.logger.info("📁 Semua direktori lokal sudah ada")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating local directories: {str(e)}")
            return False
    
    def create_symlinks(self, drive_paths: Dict[str, Path]) -> bool:
        """
        Buat symlinks dari Drive ke lokal (Colab only)
        
        Args:
            drive_paths: Dictionary path Drive
            
        Returns:
            Success status
        """
        if not self._is_colab() or not drive_paths:
            return True
        
        try:
            created_count = 0
            for name, drive_path in drive_paths.items():
                if name == 'base':
                    continue
                
                local_path = Path(f"/content/{name}")
                
                # Skip configs - handled separately
                if name == 'configs':
                    continue
                
                # Backup existing directory jika bukan symlink
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = Path(f"/content/{name}_backup")
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(local_path, backup_path)
                    self.logger.info(f"🔄 Backup direktori: {name} → {name}_backup")
                
                # Buat symlink jika belum ada
                if not local_path.exists():
                    local_path.symlink_to(drive_path)
                    self.logger.info(f"🔗 Symlink dibuat: {name}")
                    created_count += 1
            
            if created_count > 0:
                self.logger.success(f"✅ Dibuat {created_count} symlink")
            else:
                self.logger.info("🔗 Semua symlink sudah ada")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating symlinks: {str(e)}")
            return False
    
    def verify_directories(self) -> Dict[str, bool]:
        """
        Verifikasi bahwa semua direktori tersedia
        
        Returns:
            Dictionary status direktori
        """
        required_dirs = ['data', 'exports', 'logs', 'models', 'output', 'configs']
        
        if self._is_colab():
            base_path = Path("/content")
        else:
            base_path = self.env_manager.base_dir
        
        status = {}
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            exists = dir_path.exists()
            status[dir_name] = exists
            
            if not exists:
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self.logger.warning(f"⚠️ Direktori belum ada: {', '.join(missing_dirs)}")
        else:
            self.logger.success("✅ Semua direktori tersedia")
        
        return status
    
    def _is_colab(self) -> bool:
        """Check apakah running di Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False