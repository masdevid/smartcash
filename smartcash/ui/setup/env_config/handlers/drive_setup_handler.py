"""
File: smartcash/ui/setup/env_config/handlers/drive_setup_handler.py
Deskripsi: Drive setup dengan wait mechanism untuk mount completion
"""

import os
import shutil
import time
from typing import Dict, Any, Tuple
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

class DriveSetupHandler:
    """Drive setup dengan proper mount detection"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = create_ui_logger_bridge(ui_components, "drive_setup")
        self.required_folders = ['data', 'configs', 'exports', 'logs', 'models', 'output']
        self.config_templates = [
            'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
            'model_config.yaml', 'training_config.yaml', 'augmentation_config.yaml',
            'preprocessing_config.yaml', 'hyperparameters_config.yaml'
        ]
    
    def ensure_drive_mounted(self) -> Tuple[bool, str]:
        """Mount Drive dengan wait mechanism"""
        if not self._is_colab():
            return True, "Local environment"
        
        # Check existing mount
        if self._check_drive_ready():
            return True, "Drive sudah terhubung"
        
        try:
            from google.colab import drive
            self.logger.info("📱 Mounting Google Drive...")
            drive.mount('/content/drive')
            
            # Wait untuk mount completion dengan timeout
            return self._wait_for_drive_ready(max_wait=30)
            
        except Exception as e:
            return False, f"Mount error: {str(e)}"
    
    def _check_drive_ready(self) -> bool:
        """Check apakah Drive benar-benar ready"""
        drive_path = Path('/content/drive/MyDrive')
        if not drive_path.exists():
            return False
            
        # Test write access
        try:
            test_file = drive_path / '.smartcash_test'
            test_file.write_text('test')
            test_file.unlink()
            return True
        except Exception:
            return False
    
    def _wait_for_drive_ready(self, max_wait: int = 30) -> Tuple[bool, str]:
        """Wait untuk Drive ready dengan progress"""
        for i in range(max_wait):
            if self._check_drive_ready():
                return True, "Drive siap digunakan"
            
            if i % 5 == 0 and i > 0:  # Update setiap 5 detik
                self.logger.info(f"⏳ Menunggu Drive ready... ({i}s)")
            
            time.sleep(1)
        
        return False, "Timeout waiting for Drive"
    
    def create_drive_folders(self, drive_base_path: str) -> Dict[str, bool]:
        """Create folders di Drive dengan retry"""
        results = {}
        drive_base = Path(drive_base_path)
        
        # Ensure base path exists
        drive_base.mkdir(parents=True, exist_ok=True)
        
        for folder in self.required_folders:
            folder_path = drive_base / folder
            try:
                if not folder_path.exists():
                    folder_path.mkdir(parents=True, exist_ok=True)
                    # Verify creation
                    time.sleep(0.1)  # Brief wait
                    results[folder] = folder_path.exists()
                else:
                    results[folder] = True
            except Exception as e:
                results[folder] = False
                self.logger.error(f"❌ Folder {folder}: {str(e)}")
        
        success_count = sum(results.values())
        if success_count > 0:
            self.logger.success(f"📁 Setup {success_count} folder di Drive")
        
        return results
    
    def clone_config_templates(self, drive_base_path: str) -> Dict[str, bool]:
        """Clone configs dengan validation"""
        results = {}
        repo_config_path = Path('/content/smartcash/configs')
        drive_config_path = Path(drive_base_path) / 'configs'
        
        drive_config_path.mkdir(parents=True, exist_ok=True)
        
        cloned_count = 0
        for config_file in self.config_templates:
            src_file = repo_config_path / config_file
            dst_file = drive_config_path / config_file
            
            try:
                if src_file.exists() and not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    # Verify copy
                    time.sleep(0.1)
                    if dst_file.exists():
                        results[config_file] = True
                        cloned_count += 1
                    else:
                        results[config_file] = False
                else:
                    results[config_file] = dst_file.exists()
            except Exception as e:
                results[config_file] = False
                self.logger.error(f"❌ Config {config_file}: {str(e)}")
        
        if cloned_count > 0:
            self.logger.success(f"📋 Clone {cloned_count} config")
        
        return results
    
    def create_symlinks(self, drive_base_path: str) -> Dict[str, bool]:
        """Create symlinks dengan validation"""
        if not self._is_colab():
            return {folder: True for folder in self.required_folders}
        
        results = {}
        drive_base = Path(drive_base_path)
        
        for folder in self.required_folders:
            local_path = Path(f'/content/{folder}')
            drive_path = drive_base / folder
            
            try:
                # Skip jika symlink valid sudah ada
                if local_path.is_symlink() and local_path.exists() and local_path.resolve() == drive_path.resolve():
                    results[folder] = True
                    continue
                
                # Backup existing
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = Path(f'/content/{folder}_backup')
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                    shutil.move(local_path, backup_path)
                
                # Remove broken symlink
                if local_path.is_symlink():
                    local_path.unlink()
                
                # Create symlink
                local_path.symlink_to(drive_path)
                
                # Verify symlink
                results[folder] = local_path.exists() and local_path.is_symlink()
                
            except Exception as e:
                results[folder] = False
                self.logger.error(f"❌ Symlink {folder}: {str(e)}")
        
        success_count = sum(results.values())
        if success_count > 0:
            self.logger.success(f"🔗 Setup {success_count} symlink")
        
        return results
    
    def perform_complete_setup(self, drive_base_path: str) -> Dict[str, Any]:
        """Complete setup dengan proper sequencing"""
        # Ensure Drive ready first
        if not self._check_drive_ready():
            return {'success': False, 'error': 'Drive not ready'}
        
        self._update_progress(0.3, "📁 Creating Drive folders...")
        folder_results = self.create_drive_folders(drive_base_path)
        
        self._update_progress(0.5, "📋 Cloning configs...")
        config_results = self.clone_config_templates(drive_base_path)
        
        self._update_progress(0.7, "🔗 Creating symlinks...")
        symlink_results = self.create_symlinks(drive_base_path)
        
        return {
            'folders': folder_results,
            'configs': config_results,
            'symlinks': symlink_results,
            'success': (
                all(folder_results.values()) and
                sum(config_results.values()) >= 3 and
                all(symlink_results.values())
            )
        }
    
    def _update_progress(self, value: float, message: str = ""):
        if 'progress_bar' in self.ui_components:
            try:
                from smartcash.ui.setup.env_config.components.progress_tracking import update_progress
                update_progress(self.ui_components, int(value * 100), 100, message)
            except ImportError:
                pass
    
    def _is_colab(self) -> bool:
        try:
            import google.colab
            return True
        except ImportError:
            return False