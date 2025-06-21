"""
File: smartcash/ui/setup/env_config/handlers/drive_setup_handler.py
Deskripsi: Drive setup handler dengan constants dan utils integration
"""

import os
import shutil
import time
from typing import Dict, Any, Tuple
from pathlib import Path

from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.setup.env_config.constants import (
    REQUIRED_FOLDERS, CONFIG_TEMPLATES, DRIVE_MOUNT_POINT, 
    REPO_CONFIG_PATH, STATUS_MESSAGES, PROGRESS_MESSAGES, RETRY_CONFIG
)
from smartcash.ui.setup.env_config.utils import (
    update_progress_safe, is_colab_environment, test_drive_readiness,
    wait_for_drive_ready, create_folder_with_retry, validate_setup_integrity
)

class DriveSetupHandler:
    """Drive setup handler dengan constants dan utils integration"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = None
        self.required_folders = REQUIRED_FOLDERS
        self.config_templates = CONFIG_TEMPLATES
    
    def _ensure_logger(self):
        """Initialize logger dengan one-liner"""
        if self.logger is None:
            self.logger = create_ui_logger_bridge(self.ui_components, "drive_setup")
    
    def ensure_drive_mounted(self) -> Tuple[bool, str]:
        """Mount Drive dengan constants dan utils integration"""
        self._ensure_logger()
        
        if not is_colab_environment():
            return True, "Local environment"
        
        if test_drive_readiness(Path(DRIVE_MOUNT_POINT)):
            return True, STATUS_MESSAGES['drive_ready']
        
        try:
            from google.colab import drive
            self.logger.info("📱 Mounting Google Drive...")
            update_progress_safe(self.ui_components, 10, PROGRESS_MESSAGES['drive_mount'])
            
            drive.mount('/content/drive')
            return wait_for_drive_ready(RETRY_CONFIG['drive_mount_timeout'], self.ui_components)
            
        except Exception as e:
            return False, f"Mount error: {str(e)}"
    
    def create_drive_folders(self, drive_base_path: str) -> Dict[str, bool]:
        """Create folders dengan progress tracking dan utils"""
        self._ensure_logger()
        update_progress_safe(self.ui_components, 30, PROGRESS_MESSAGES['folders_create'])
        
        results = {}
        drive_base = Path(drive_base_path)
        
        # Ensure base path dengan retry
        if not create_folder_with_retry(drive_base):
            self.logger.error("❌ Gagal create base path")
            return {folder: False for folder in self.required_folders}
        
        # Create folders dengan progress tracking
        total_folders = len(self.required_folders)
        for idx, folder in enumerate(self.required_folders):
            folder_path = drive_base / folder
            progress = 30 + (idx / total_folders) * 15  # 30-45% range
            update_progress_safe(self.ui_components, int(progress), f"📁 Creating {folder}...")
            
            success = create_folder_with_retry(folder_path, RETRY_CONFIG['drive_ready_attempts'])
            results[folder] = success
        
        success_count = sum(results.values())
        update_progress_safe(self.ui_components, 45, f"📁 Created {success_count}/{total_folders} folders")
        
        if success_count > 0:
            self.logger.success(f"📁 Setup {success_count}/{total_folders} folder di Drive")
        
        return results
    
    def clone_config_templates(self, drive_base_path: str) -> Dict[str, bool]:
        """Clone configs dengan constants dan progress tracking"""
        self._ensure_logger()
        update_progress_safe(self.ui_components, 50, PROGRESS_MESSAGES['configs_clone'])
        
        results = {}
        repo_config_path = Path(REPO_CONFIG_PATH)
        drive_config_path = Path(drive_base_path) / 'configs'
        
        # Ensure destination exists
        if not create_folder_with_retry(drive_config_path):
            self.logger.error("❌ Gagal create config directory")
            return {config: False for config in self.config_templates}
        
        cloned_count = 0
        total_configs = len(self.config_templates)
        
        for idx, config_file in enumerate(self.config_templates):
            src_file = repo_config_path / config_file
            dst_file = drive_config_path / config_file
            success = False
            
            progress = 50 + (idx / total_configs) * 15  # 50-65% range
            update_progress_safe(self.ui_components, int(progress), f"📋 Copying {config_file}...")
            
            try:
                if src_file.exists():
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
                        time.sleep(0.1)
                        success = dst_file.exists() and dst_file.stat().st_size > 0
                        if success:
                            cloned_count += 1
                    else:
                        success = dst_file.exists() and dst_file.stat().st_size > 0
                else:
                    self.logger.warning(f"⚠️ Source config tidak ditemukan: {config_file}")
                    
            except Exception as e:
                self.logger.error(f"❌ Config {config_file}: {str(e)}")
            
            results[config_file] = success
        
        update_progress_safe(self.ui_components, 65, f"📋 Cloned {cloned_count} config templates")
        if cloned_count > 0:
            self.logger.success(f"📋 Clone {cloned_count} config template")
        
        return results
    
    def create_symlinks(self, drive_base_path: str) -> Dict[str, bool]:
        """Create symlinks dengan constants dan utils integration"""
        self._ensure_logger()
        update_progress_safe(self.ui_components, 70, PROGRESS_MESSAGES['symlinks_create'])
        
        if not is_colab_environment():
            return {folder: True for folder in self.required_folders}
        
        results = {}
        drive_base = Path(drive_base_path)
        total_symlinks = len(self.required_folders)
        
        for idx, folder in enumerate(self.required_folders):
            local_path = Path(f'/content/{folder}')
            drive_path = drive_base / folder
            success = False
            
            progress = 70 + (idx / total_symlinks) * 20  # 70-90% range
            update_progress_safe(self.ui_components, int(progress), f"🔗 Linking {folder}...")
            
            try:
                # Check existing valid symlink
                if (local_path.is_symlink() and local_path.exists() and 
                    local_path.resolve() == drive_path.resolve()):
                    success = True
                    results[folder] = success
                    continue
                
                # Backup existing directory
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = Path(f'/content/{folder}_backup_{int(time.time())}')
                    try:
                        shutil.move(local_path, backup_path)
                        self.logger.info(f"📦 Backup {folder} ke {backup_path.name}")
                    except Exception:
                        if local_path.is_dir():
                            shutil.rmtree(local_path)
                        else:
                            local_path.unlink()
                
                # Remove broken symlink
                if local_path.is_symlink():
                    local_path.unlink()
                
                # Ensure drive path exists
                if not drive_path.exists():
                    create_folder_with_retry(drive_path)
                
                # Create symlink dengan retry
                for attempt in range(RETRY_CONFIG['symlink_attempts']):
                    try:
                        local_path.symlink_to(drive_path)
                        time.sleep(0.2)
                        
                        if (local_path.exists() and local_path.is_symlink() and 
                            local_path.resolve() == drive_path.resolve()):
                            success = True
                            break
                            
                    except Exception as e:
                        if attempt == RETRY_CONFIG['symlink_attempts'] - 1:
                            self.logger.error(f"❌ Symlink {folder}: {str(e)}")
                        else:
                            time.sleep(RETRY_CONFIG['symlink_delay'])
                
            except Exception as e:
                self.logger.error(f"❌ Symlink {folder}: {str(e)}")
            
            results[folder] = success
        
        success_count = sum(results.values())
        update_progress_safe(self.ui_components, 90, f"🔗 Created {success_count}/{total_symlinks} symlinks")
        
        if success_count > 0:
            self.logger.success(f"🔗 Setup {success_count}/{total_symlinks} symlink")
        
        return results
    
    def perform_complete_setup(self, drive_base_path: str) -> Dict[str, Any]:
        """Complete setup dengan constants/utils integration"""
        self._ensure_logger()
        
        if not test_drive_readiness(Path(DRIVE_MOUNT_POINT)):
            return {'success': False, 'error': 'Drive not ready for setup'}
        
        self.logger.info(f"🚀 Memulai setup lengkap di: {drive_base_path}")
        update_progress_safe(self.ui_components, 5, "🚀 Starting complete setup...")
        
        # Execute setup steps
        folder_results = self.create_drive_folders(drive_base_path)
        config_results = self.clone_config_templates(drive_base_path)
        
        time.sleep(1)  # Settling time
        symlink_results = self.create_symlinks(drive_base_path)
        
        # Evaluate success dengan utils
        folder_success_rate = sum(folder_results.values()) / len(folder_results)
        config_success_count = sum(config_results.values())
        symlink_success_rate = sum(symlink_results.values()) / len(symlink_results)
        
        folder_success = folder_success_rate >= 0.8
        config_success = config_success_count >= 5  # Essential configs
        symlink_success = symlink_success_rate >= 0.8
        
        overall_success = folder_success and config_success and symlink_success
        
        # Final validation
        update_progress_safe(self.ui_components, 95, PROGRESS_MESSAGES['validation'])
        if overall_success:
            validation_passed = validate_setup_integrity(drive_base_path)
            if not validation_passed:
                self.logger.warning("⚠️ Setup validation menunjukkan beberapa issue")
                overall_success = False
        
        # Complete progress
        final_message = PROGRESS_MESSAGES['complete'] if overall_success else "⚠️ Setup completed with issues"
        update_progress_safe(self.ui_components, 100, final_message)
        
        return {
            'folders': folder_results,
            'configs': config_results,
            'symlinks': symlink_results,
            'success': overall_success,
            'validation': {
                'folder_success_rate': folder_success_rate,
                'config_count': config_success_count,
                'symlink_success_rate': symlink_success_rate
            }
        }