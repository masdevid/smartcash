"""
File: smartcash/ui/setup/env_config/handlers/environment_status_checker.py  
Deskripsi: Status checker dengan constants dan utils integration
"""

from typing import Dict, Any
from pathlib import Path

from smartcash.ui.setup.env_config.constants import (
    REQUIRED_FOLDERS, CONFIG_TEMPLATES, ESSENTIAL_CONFIGS,
    DRIVE_MOUNT_POINT, SMARTCASH_DRIVE_PATH
)
from smartcash.ui.setup.env_config.utils import (
    is_colab_environment, test_drive_readiness, check_symlink_status,
    validate_folders, validate_configs, validate_repo_configs,
    evaluate_readiness, get_missing_items
)

class EnvironmentStatusChecker:
    """Environment status checker dengan constants dan utils integration"""
    
    def __init__(self):
        self.required_folders = REQUIRED_FOLDERS
        self.config_templates = CONFIG_TEMPLATES
        self.essential_configs = ESSENTIAL_CONFIGS
    
    def check_drive_mount_status(self) -> Dict[str, Any]:
        """Check Drive status dengan utils integration - one-liner"""
        if not is_colab_environment():
            return {'mounted': True, 'path': None, 'type': 'local', 'ready': True}
        
        drive_mount_point = Path(DRIVE_MOUNT_POINT)
        smartcash_drive_path = Path(SMARTCASH_DRIVE_PATH)
        
        if not drive_mount_point.exists():
            return {'mounted': False, 'path': None, 'type': 'colab', 'ready': False}
        
        ready = test_drive_readiness(drive_mount_point)
        return {
            'mounted': True, 'path': str(smartcash_drive_path),
            'type': 'colab', 'ready': ready
        }
    
    def check_drive_folders(self, drive_path: str = None) -> Dict[str, bool]:
        """Check Drive folders dengan utils - one-liner"""
        return validate_folders(drive_path) if drive_path else {folder: False for folder in self.required_folders}
    
    def check_drive_configs(self, drive_path: str = None) -> Dict[str, bool]:
        """Check Drive configs dengan utils - one-liner"""
        return validate_configs(drive_path) if drive_path else {config: False for config in self.config_templates}
    
    def validate_repo_configs(self) -> Dict[str, bool]:
        """Validate repo configs dengan utils - one-liner"""
        return validate_repo_configs()
    
    def check_local_symlinks(self) -> Dict[str, Dict[str, Any]]:
        """Check symlinks dengan utils - one-liner"""
        if not is_colab_environment():
            return {folder: {'exists': True, 'is_symlink': False, 'valid': True} 
                    for folder in self.required_folders}
        
        return {folder: check_symlink_status(Path(f'/content/{folder}')) 
                for folder in self.required_folders}
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status dengan utils integration"""
        drive_status = self.check_drive_mount_status()
        repo_configs = self.validate_repo_configs()
        
        # Check Drive components hanya jika Drive ready
        if drive_status['mounted'] and drive_status.get('ready', False) and drive_status['path']:
            drive_folders = self.check_drive_folders(drive_status['path'])
            drive_configs = self.check_drive_configs(drive_status['path'])
        else:
            drive_folders = {folder: False for folder in self.required_folders}
            drive_configs = {config: False for config in self.config_templates}
        
        symlinks = self.check_local_symlinks()
        
        # Evaluate readiness dengan utils
        overall_ready = evaluate_readiness(drive_status, repo_configs, drive_folders, drive_configs, symlinks)
        
        # Get missing items dengan utils
        missing_items = get_missing_items(repo_configs, drive_folders, drive_configs, symlinks)
        
        return {
            'ready': overall_ready,
            'drive': drive_status,
            'repo_configs': repo_configs,
            'drive_folders': drive_folders,
            'drive_configs': drive_configs,
            'symlinks': symlinks,
            **missing_items
        }