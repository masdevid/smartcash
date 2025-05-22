"""
File: smartcash/ui/setup/env_config/handlers/drive_setup_handler.py
Deskripsi: Handler khusus untuk setup Drive dengan minimal logging dan error handling yang robust
"""

import os
import shutil
from typing import Dict, Any, Tuple, List
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

class DriveSetupHandler:
    """Handler untuk setup Drive operations tanpa logging berlebihan"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi handler dengan minimal logging"""
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "drive_setup")
        self.required_folders = ['data', 'configs', 'exports', 'logs', 'models', 'output']
        self.config_templates = [
            'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
            'model_config.yaml', 'training_config.yaml', 'augmentation_config.yaml',
            'preprocessing_config.yaml', 'hyperparameters_config.yaml'
        ]
    
    def ensure_drive_mounted(self) -> Tuple[bool, str]:
        """Mount Drive jika belum ter-mount"""
        if not self._is_colab():
            return True, "Local environment"
        
        drive_path = Path('/content/drive/MyDrive')
        if drive_path.exists():
            return True, f"Drive sudah terhubung"
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            return True, "Drive berhasil terhubung"
        except Exception as e:
            return False, f"Gagal mount drive: {str(e)}"
    
    def create_drive_folders(self, drive_base_path: str) -> Dict[str, bool]:
        """Buat folders di Drive dengan handling yang robust"""
        results = {}
        drive_base = Path(drive_base_path)
        
        for folder in self.required_folders:
            folder_path = drive_base / folder
            try:
                if not folder_path.exists():
                    folder_path.mkdir(parents=True, exist_ok=True)
                    results[folder] = True
                else:
                    results[folder] = True  # Already exists
            except Exception as e:
                results[folder] = False
                self.logger.error(f"❌ Gagal buat folder {folder}: {str(e)}")
        
        created_count = sum(1 for success in results.values() if success)
        if created_count > 0:
            self.logger.success(f"📁 Setup {created_count} folder di Drive")
        
        return results
    
    def clone_config_templates(self, drive_base_path: str) -> Dict[str, bool]:
        """Clone config templates dari repo ke Drive"""
        results = {}
        repo_config_path = Path('/content/smartcash/configs')
        drive_config_path = Path(drive_base_path) / 'configs'
        
        # Pastikan drive config folder ada
        drive_config_path.mkdir(parents=True, exist_ok=True)
        
        cloned_count = 0
        for config_file in self.config_templates:
            src_file = repo_config_path / config_file
            dst_file = drive_config_path / config_file
            
            try:
                if src_file.exists() and not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    results[config_file] = True
                    cloned_count += 1
                else:
                    results[config_file] = dst_file.exists()
            except Exception as e:
                results[config_file] = False
                self.logger.error(f"❌ Gagal clone {config_file}: {str(e)}")
        
        if cloned_count > 0:
            self.logger.success(f"📋 Clone {cloned_count} config template")
        
        return results
    
    def create_symlinks(self, drive_base_path: str) -> Dict[str, bool]:
        """Buat symlinks dari Drive ke local dengan error handling"""
        if not self._is_colab():
            return {folder: True for folder in self.required_folders}
        
        results = {}
        drive_base = Path(drive_base_path)
        
        for folder in self.required_folders:
            local_path = Path(f'/content/{folder}')
            drive_path = drive_base / folder
            
            try:
                # Skip jika sudah ada symlink yang valid
                if local_path.is_symlink() and local_path.exists():
                    results[folder] = True
                    continue
                
                # Backup existing directory jika bukan symlink
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = Path(f'/content/{folder}_backup')
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(local_path, backup_path)
                
                # Buat symlink
                if not local_path.exists():
                    local_path.symlink_to(drive_path)
                
                results[folder] = local_path.exists() and local_path.is_symlink()
                
            except Exception as e:
                results[folder] = False
                self.logger.error(f"❌ Gagal buat symlink {folder}: {str(e)}")
        
        success_count = sum(1 for success in results.values() if success)
        if success_count > 0:
            self.logger.success(f"🔗 Setup {success_count} symlink")
        
        return results
    
    def perform_complete_setup(self, drive_base_path: str) -> Dict[str, Any]:
        """Lakukan complete setup Drive dengan progress tracking"""
        # Step 1: Create drive folders
        self._update_progress(0.3, "📁 Membuat folder di Drive...")
        folder_results = self.create_drive_folders(drive_base_path)
        
        # Step 2: Clone config templates
        self._update_progress(0.5, "📋 Clone config templates...")
        config_results = self.clone_config_templates(drive_base_path)
        
        # Step 3: Create symlinks
        self._update_progress(0.7, "🔗 Membuat symlinks...")
        symlink_results = self.create_symlinks(drive_base_path)
        
        return {
            'folders': folder_results,
            'configs': config_results,
            'symlinks': symlink_results,
            'success': (
                all(folder_results.values()) and
                sum(config_results.values()) >= 3 and  # Minimal 3 configs
                all(symlink_results.values())
            )
        }
    
    def _update_progress(self, value: float, message: str = ""):
        """Update progress tanpa error jika komponen tidak ada"""
        if 'progress_bar' in self.ui_components:
            try:
                from smartcash.ui.setup.env_config.components.progress_tracking import update_progress
                update_progress(self.ui_components, int(value * 100), 100, message)
            except ImportError:
                pass
    
    def _is_colab(self) -> bool:
        """Check Colab environment"""
        try:
            import google.colab
            return True
        except ImportError:
            return False