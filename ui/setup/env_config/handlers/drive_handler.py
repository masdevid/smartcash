"""
File: smartcash/ui/setup/env_config/handlers/drive_handler.py
Deskripsi: Handler untuk Google Drive operations dan config sync
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from smartcash.ui.setup.env_config.utils.environment_helpers import discover_config_templates
from smartcash.ui.setup.env_config.constants import (
    REQUIRED_FOLDERS, SMARTCASH_DRIVE_PATH, CONFIG_SOURCE_PATH
)

class DriveHandler:
    """📱 Handler untuk Drive operations"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.drive_base_path = Path(SMARTCASH_DRIVE_PATH)
        
    def mount_drive(self) -> Dict[str, Any]:
        """📱 Mount Google Drive"""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Verify mount
            if Path('/content/drive/MyDrive').exists():
                if self.logger:
                    self.logger.info("✅ Drive mounted successfully")
                return {'success': True, 'path': '/content/drive/MyDrive'}
            else:
                return {'success': False, 'error': 'Mount verification failed'}
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Drive mount failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_drive_folders(self) -> Dict[str, Any]:
        """📁 Create required folders in Drive"""
        try:
            created_folders = []
            failed_folders = []
            
            # Create base SmartCash folder
            self.drive_base_path.mkdir(parents=True, exist_ok=True)
            
            # Create required subfolders
            for folder in REQUIRED_FOLDERS:
                folder_path = self.drive_base_path / folder
                try:
                    folder_path.mkdir(parents=True, exist_ok=True)
                    created_folders.append(folder)
                except Exception as e:
                    failed_folders.append({'folder': folder, 'error': str(e)})
            
            if self.logger:
                self.logger.info(f"📁 Created {len(created_folders)}/{len(REQUIRED_FOLDERS)} folders")
                
            return {
                'success': len(failed_folders) == 0,
                'created': created_folders,
                'failed': failed_folders
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Folder creation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def sync_config_templates(self) -> Dict[str, Any]:
        """📋 Sync config templates dari repo ke Drive"""
        try:
            source_path = Path(CONFIG_SOURCE_PATH)
            target_path = self.drive_base_path / 'configs'
            
            if not source_path.exists():
                return {'success': False, 'error': 'Source configs not found'}
            
            # Auto-discover configs
            config_files = discover_config_templates()
            
            copied_files = []
            failed_files = []
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            for config_file in config_files:
                try:
                    source_file = source_path / config_file
                    target_file = target_path / config_file
                    
                    # Copy only if source is newer or target doesn't exist
                    if self._should_copy_file(source_file, target_file):
                        shutil.copy2(source_file, target_file)
                        copied_files.append(config_file)
                        
                except Exception as e:
                    failed_files.append({'file': config_file, 'error': str(e)})
            
            if self.logger:
                self.logger.info(f"📋 Synced {len(copied_files)} config files")
                
            return {
                'success': len(failed_files) == 0,
                'copied': copied_files,
                'failed': failed_files,
                'total_discovered': len(config_files)
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Config sync failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def create_symlinks(self) -> Dict[str, Any]:
        """🔗 Create symlinks dari Drive ke local"""
        try:
            symlinks_created = []
            symlinks_failed = []
            
            symlink_mappings = {
                'data': '/content/data',
                'configs': '/content/configs',
                'models': '/content/models',
                'logs': '/content/logs'
            }
            
            for folder, local_path in symlink_mappings.items():
                try:
                    drive_path = self.drive_base_path / folder
                    local_path_obj = Path(local_path)
                    
                    # Remove existing if it's not a symlink
                    if local_path_obj.exists() and not local_path_obj.is_symlink():
                        if local_path_obj.is_dir():
                            shutil.rmtree(local_path_obj)
                        else:
                            local_path_obj.unlink()
                    
                    # Create symlink
                    if not local_path_obj.exists():
                        local_path_obj.symlink_to(drive_path)
                        symlinks_created.append(folder)
                        
                except Exception as e:
                    symlinks_failed.append({'folder': folder, 'error': str(e)})
            
            if self.logger:
                self.logger.info(f"🔗 Created {len(symlinks_created)} symlinks")
                
            return {
                'success': len(symlinks_failed) == 0,
                'created': symlinks_created,
                'failed': symlinks_failed
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Symlink creation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def perform_complete_setup(self, drive_path: str = None) -> Dict[str, Any]:
        """🚀 Perform complete Drive setup"""
        setup_results = {
            'success': False,
            'steps_completed': [],
            'steps_failed': [],
            'details': {}
        }
        
        try:
            # Step 1: Create folders
            folder_result = self.create_drive_folders()
            setup_results['details']['folders'] = folder_result
            
            if folder_result['success']:
                setup_results['steps_completed'].append('folders')
                if self.logger:
                    self.logger.info("✅ Drive folders created successfully")
            else:
                setup_results['steps_failed'].append('folders')
                
            # Step 2: Sync configs
            config_result = self.sync_config_templates()
            setup_results['details']['configs'] = config_result
            
            if config_result['success']:
                setup_results['steps_completed'].append('configs')
                if self.logger:
                    self.logger.info("✅ Config templates synced successfully")
            else:
                setup_results['steps_failed'].append('configs')
                
            # Step 3: Create symlinks
            symlink_result = self.create_symlinks()
            setup_results['details']['symlinks'] = symlink_result
            
            if symlink_result['success']:
                setup_results['steps_completed'].append('symlinks')
                if self.logger:
                    self.logger.info("✅ Symlinks created successfully")
            else:
                setup_results['steps_failed'].append('symlinks')
            
            # Overall success
            setup_results['success'] = len(setup_results['steps_failed']) == 0
            
            if setup_results['success']:
                if self.logger:
                    self.logger.info("🎉 Complete Drive setup finished successfully")
            else:
                if self.logger:
                    self.logger.warning(f"⚠️ Setup completed with {len(setup_results['steps_failed'])} failed steps")
                    
            return setup_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Complete setup failed: {str(e)}")
            setup_results['error'] = str(e)
            return setup_results
    
    def get_drive_status(self) -> Dict[str, Any]:
        """📊 Get comprehensive Drive status"""
        try:
            drive_mounted = Path('/content/drive/MyDrive').exists()
            smartcash_exists = self.drive_base_path.exists()
            
            folder_status = {}
            if smartcash_exists:
                for folder in REQUIRED_FOLDERS:
                    folder_path = self.drive_base_path / folder
                    folder_status[folder] = folder_path.exists()
            
            return {
                'drive_mounted': drive_mounted,
                'smartcash_folder_exists': smartcash_exists,
                'folder_status': folder_status,
                'ready': drive_mounted and smartcash_exists and all(folder_status.values())
            }
            
        except Exception as e:
            return {'ready': False, 'error': str(e)}
    
    def _should_copy_file(self, source_file: Path, target_file: Path) -> bool:
        """Check apakah file perlu di-copy"""
        if not target_file.exists():
            return True
            
        try:
            source_mtime = source_file.stat().st_mtime
            target_mtime = target_file.stat().st_mtime
            return source_mtime > target_mtime
        except Exception:
            return True  # Copy jika tidak bisa compare