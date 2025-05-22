"""
File: smartcash/ui/setup/env_config/handlers/environment_status_checker.py
Deskripsi: Checker untuk status environment dengan validasi yang lebih spesifik dan tanpa logging berlebihan
"""

from typing import Dict, Any, List
from pathlib import Path

class EnvironmentStatusChecker:
    """Handler khusus untuk checking status environment tanpa logging berlebihan"""
    
    def __init__(self):
        """Inisialisasi checker tanpa dependency external"""
        self.required_folders = ['data', 'configs', 'exports', 'logs', 'models', 'output']
        self.config_templates = [
            'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
            'model_config.yaml', 'training_config.yaml', 'augmentation_config.yaml',
            'preprocessing_config.yaml', 'hyperparameters_config.yaml'
        ]
    
    def is_colab_environment(self) -> bool:
        """Check apakah berjalan di Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def check_drive_mount_status(self) -> Dict[str, Any]:
        """Check status Google Drive mount tanpa logging berlebihan"""
        if not self.is_colab_environment():
            return {'mounted': True, 'path': None, 'type': 'local'}
        
        drive_mount_point = Path('/content/drive/MyDrive')
        smartcash_drive_path = drive_mount_point / 'SmartCash'
        
        return {
            'mounted': drive_mount_point.exists(),
            'path': str(smartcash_drive_path) if drive_mount_point.exists() else None,
            'type': 'colab'
        }
    
    def validate_repo_configs(self) -> Dict[str, bool]:
        """Validasi config templates di repo"""
        repo_config_path = Path('/content/smartcash/configs')
        return {config: (repo_config_path / config).exists() for config in self.config_templates}
    
    def check_drive_folders(self, drive_path: str = None) -> Dict[str, bool]:
        """Check required folders di Drive"""
        if not drive_path:
            return {folder: False for folder in self.required_folders}
        
        drive_base = Path(drive_path)
        return {folder: (drive_base / folder).exists() for folder in self.required_folders}
    
    def check_drive_configs(self, drive_path: str = None) -> Dict[str, bool]:
        """Check config templates di Drive"""
        if not drive_path:
            return {config: False for config in self.config_templates}
        
        drive_config_path = Path(drive_path) / 'configs'
        return {config: (drive_config_path / config).exists() for config in self.config_templates}
    
    def check_local_symlinks(self) -> Dict[str, Dict[str, Any]]:
        """Check symlinks di Colab"""
        if not self.is_colab_environment():
            return {folder: {'exists': True, 'is_symlink': False, 'valid': True} for folder in self.required_folders}
        
        results = {}
        for folder in self.required_folders:
            local_path = Path(f'/content/{folder}')
            results[folder] = {
                'exists': local_path.exists(),
                'is_symlink': local_path.is_symlink() if local_path.exists() else False,
                'valid': local_path.exists() and (local_path.is_symlink() or local_path.is_dir())
            }
        
        return results
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get status lengkap environment tanpa logging"""
        drive_status = self.check_drive_mount_status()
        repo_configs = self.validate_repo_configs()
        
        # Check drive components jika mounted
        if drive_status['mounted'] and drive_status['path']:
            drive_folders = self.check_drive_folders(drive_status['path'])
            drive_configs = self.check_drive_configs(drive_status['path'])
        else:
            drive_folders = {folder: False for folder in self.required_folders}
            drive_configs = {config: False for config in self.config_templates}
        
        symlinks = self.check_local_symlinks()
        
        # Determine readiness
        repo_configs_ready = all(repo_configs.values())
        drive_folders_ready = all(drive_folders.values())
        drive_configs_ready = sum(drive_configs.values()) >= 3  # Minimal 3 config files
        symlinks_ready = all(link['valid'] for link in symlinks.values())
        
        overall_ready = (
            drive_status['mounted'] and 
            repo_configs_ready and 
            drive_folders_ready and 
            drive_configs_ready and 
            symlinks_ready
        )
        
        return {
            'ready': overall_ready,
            'drive': drive_status,
            'repo_configs': repo_configs,
            'drive_folders': drive_folders,
            'drive_configs': drive_configs,
            'symlinks': symlinks,
            'missing_repo_configs': [k for k, v in repo_configs.items() if not v],
            'missing_drive_folders': [k for k, v in drive_folders.items() if not v],
            'missing_drive_configs': [k for k, v in drive_configs.items() if not v],
            'invalid_symlinks': [k for k, v in symlinks.items() if not v['valid']]
        }