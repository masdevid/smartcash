"""
File: smartcash/ui/setup/env_config/handlers/environment_status_checker.py  
Deskripsi: Status checker dengan proper Drive detection timing
"""

import time
from typing import Dict, Any
from pathlib import Path

class EnvironmentStatusChecker:
    """Environment status checker dengan Drive ready validation"""
    
    def __init__(self):
        self.required_folders = ['data', 'configs', 'exports', 'logs', 'models', 'output']
        self.config_templates = [
            'base_config.yaml', 'colab_config.yaml', 'dataset_config.yaml',
            'model_config.yaml', 'training_config.yaml', 'augmentation_config.yaml',
            'preprocessing_config.yaml', 'hyperparameters_config.yaml'
        ]
    
    def is_colab_environment(self) -> bool:
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def check_drive_mount_status(self) -> Dict[str, Any]:
        """Check Drive status dengan proper validation"""
        if not self.is_colab_environment():
            return {'mounted': True, 'path': None, 'type': 'local', 'ready': True}
        
        drive_mount_point = Path('/content/drive/MyDrive')
        smartcash_drive_path = drive_mount_point / 'SmartCash'
        
        if not drive_mount_point.exists():
            return {'mounted': False, 'path': None, 'type': 'colab', 'ready': False}
        
        # Test Drive readiness
        ready = self._test_drive_readiness(drive_mount_point)
        
        return {
            'mounted': True,
            'path': str(smartcash_drive_path),
            'type': 'colab',
            'ready': ready
        }
    
    def _test_drive_readiness(self, drive_path: Path) -> bool:
        """Test apakah Drive benar-benar ready untuk operations"""
        try:
            # Test basic access
            if not drive_path.exists():
                return False
            
            # Test write capability
            test_file = drive_path / '.smartcash_ready_test'
            test_file.write_text('ready')
            
            # Immediate read test
            content = test_file.read_text()
            test_file.unlink()
            
            return content == 'ready'
            
        except Exception:
            return False
    
    def validate_repo_configs(self) -> Dict[str, bool]:
        """Validate config templates di repo"""
        repo_config_path = Path('/content/smartcash/configs')
        return {config: (repo_config_path / config).exists() for config in self.config_templates}
    
    def check_drive_folders(self, drive_path: str = None) -> Dict[str, bool]:
        """Check folders di Drive dengan validation"""
        if not drive_path:
            return {folder: False for folder in self.required_folders}
        
        drive_base = Path(drive_path)
        results = {}
        
        for folder in self.required_folders:
            folder_path = drive_base / folder
            results[folder] = folder_path.exists() and folder_path.is_dir()
        
        return results
    
    def check_drive_configs(self, drive_path: str = None) -> Dict[str, bool]:
        """Check config files di Drive"""
        if not drive_path:
            return {config: False for config in self.config_templates}
        
        drive_config_path = Path(drive_path) / 'configs'
        results = {}
        
        for config in self.config_templates:
            config_file = drive_config_path / config
            results[config] = config_file.exists() and config_file.is_file()
        
        return results
    
    def check_local_symlinks(self) -> Dict[str, Dict[str, Any]]:
        """Check symlinks dengan validation yang proper"""
        if not self.is_colab_environment():
            return {folder: {'exists': True, 'is_symlink': False, 'valid': True} for folder in self.required_folders}
        
        results = {}
        for folder in self.required_folders:
            local_path = Path(f'/content/{folder}')
            
            if not local_path.exists():
                results[folder] = {'exists': False, 'is_symlink': False, 'valid': False}
            elif local_path.is_symlink():
                # Test symlink validity
                try:
                    target_exists = local_path.resolve().exists()
                    results[folder] = {
                        'exists': True,
                        'is_symlink': True,
                        'valid': target_exists
                    }
                except Exception:
                    results[folder] = {'exists': True, 'is_symlink': True, 'valid': False}
            else:
                results[folder] = {'exists': True, 'is_symlink': False, 'valid': True}
        
        return results
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status dengan proper timing"""
        drive_status = self.check_drive_mount_status()
        repo_configs = self.validate_repo_configs()
        
        # Only check Drive components jika Drive ready
        if drive_status['mounted'] and drive_status.get('ready', False) and drive_status['path']:
            drive_folders = self.check_drive_folders(drive_status['path'])
            drive_configs = self.check_drive_configs(drive_status['path'])
        else:
            drive_folders = {folder: False for folder in self.required_folders}
            drive_configs = {config: False for config in self.config_templates}
        
        symlinks = self.check_local_symlinks()
        
        # Determine readiness dengan Drive ready check
        repo_configs_ready = all(repo_configs.values())
        drive_folders_ready = all(drive_folders.values())
        drive_configs_ready = sum(drive_configs.values()) >= 3
        symlinks_ready = all(link['valid'] for link in symlinks.values())
        
        overall_ready = (
            drive_status['mounted'] and 
            drive_status.get('ready', False) and
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