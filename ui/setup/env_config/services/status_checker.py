"""
File: smartcash/ui/setup/env_config/services/status_checker.py

Service untuk checking status environment setup.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import os

from smartcash.ui.setup.env_config.constants import (
    REQUIRED_FOLDERS, SYMLINK_MAP, DRIVE_PATH, COLAB_PATH, DEFAULT_CONFIG_DIR
)
from smartcash.common.environment import get_environment_manager


class EnvironmentStatusChecker:
    """Service untuk checking comprehensive environment status."""
    
    def __init__(self):
        self.env_manager = get_environment_manager()
    
    def check_all_status(self) -> Dict[str, Any]:
        """Check semua status environment.
        
        Returns:
            Dictionary berisi comprehensive status
        """
        return {
            'drive_status': self.check_drive_status(),
            'folders_status': self.check_folders_status(),
            'symlinks_status': self.check_symlinks_status(),
            'configs_status': self.check_configs_status(),
            'overall_status': self.get_overall_status()
        }
    
    def check_drive_status(self) -> Dict[str, Any]:
        """Check Google Drive mount status."""
        try:
            is_mounted = self.env_manager.is_drive_mounted()
            mount_path = self.env_manager.get_drive_path() if is_mounted else None
            
            status = {
                'mounted': is_mounted,
                'mount_path': str(mount_path) if mount_path else None,
                'accessible': False,
                'space_info': {}
            }
            
            if is_mounted and mount_path:
                # Check accessibility
                try:
                    status['accessible'] = mount_path.exists() and os.access(mount_path, os.R_OK | os.W_OK)
                    
                    # Get space info
                    if status['accessible']:
                        import shutil
                        total, used, free = shutil.disk_usage(mount_path)
                        status['space_info'] = {
                            'total_gb': round(total / (1024**3), 2),
                            'used_gb': round(used / (1024**3), 2),
                            'free_gb': round(free / (1024**3), 2),
                            'usage_percent': round((used / total) * 100, 1)
                        }
                except Exception:
                    pass
            
            return status
            
        except Exception as e:
            return {
                'mounted': False,
                'error': str(e)
            }
    
    def check_folders_status(self) -> Dict[str, Any]:
        """Check required folders status."""
        try:
            existing_folders = []
            missing_folders = []
            folder_details = {}
            
            for folder_path in REQUIRED_FOLDERS:
                path = Path(folder_path)
                if path.exists() and path.is_dir():
                    existing_folders.append(str(path))
                    folder_details[str(path)] = {
                        'exists': True,
                        'writable': os.access(path, os.W_OK),
                        'size_mb': self._get_folder_size(path)
                    }
                else:
                    missing_folders.append(str(path))
                    folder_details[str(path)] = {
                        'exists': False,
                        'writable': False,
                        'size_mb': 0
                    }
            
            return {
                'total_required': len(REQUIRED_FOLDERS),
                'existing_count': len(existing_folders),
                'missing_count': len(missing_folders),
                'existing_folders': existing_folders,
                'missing_folders': missing_folders,
                'folder_details': folder_details,
                'all_exist': len(missing_folders) == 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'all_exist': False
            }
    
    def check_symlinks_status(self) -> Dict[str, Any]:
        """Check symlinks status."""
        try:
            existing_symlinks = []
            missing_symlinks = []
            broken_symlinks = []
            symlink_details = {}
            
            for source, target in SYMLINK_MAP.items():
                source_path = Path(source)
                target_path = Path(target)
                
                link_info = {
                    'source': str(source_path),
                    'target': str(target_path),
                    'exists': target_path.exists(),
                    'is_symlink': target_path.is_symlink(),
                    'target_exists': source_path.exists(),
                    'valid': False
                }
                
                if target_path.exists() and target_path.is_symlink():
                    try:
                        # Check if symlink points to correct source
                        actual_target = target_path.resolve()
                        expected_target = source_path.resolve()
                        link_info['valid'] = actual_target == expected_target
                        
                        if link_info['valid']:
                            existing_symlinks.append((str(source_path), str(target_path)))
                        else:
                            broken_symlinks.append((str(source_path), str(target_path)))
                    except Exception:
                        broken_symlinks.append((str(source_path), str(target_path)))
                else:
                    missing_symlinks.append((str(source_path), str(target_path)))
                
                symlink_details[str(target_path)] = link_info
            
            return {
                'total_required': len(SYMLINK_MAP),
                'existing_count': len(existing_symlinks),
                'missing_count': len(missing_symlinks),
                'broken_count': len(broken_symlinks),
                'existing_symlinks': existing_symlinks,
                'missing_symlinks': missing_symlinks,
                'broken_symlinks': broken_symlinks,
                'symlink_details': symlink_details,
                'all_valid': len(missing_symlinks) == 0 and len(broken_symlinks) == 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'all_valid': False
            }
    
    def check_configs_status(self) -> Dict[str, Any]:
        """Check config files status."""
        try:
            drive_config_path = Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR
            colab_config_path = Path(COLAB_PATH) / DEFAULT_CONFIG_DIR
            
            drive_configs = []
            colab_configs = []
            config_details = {}
            
            # Check drive configs
            if drive_config_path.exists():
                config_extensions = ['*.yaml', '*.yml', '*.json']
                for pattern in config_extensions:
                    for config_file in drive_config_path.glob(pattern):
                        drive_configs.append(config_file.name)
                        config_details[config_file.name] = {
                            'drive_exists': True,
                            'drive_size': config_file.stat().st_size,
                            'colab_exists': False,
                            'colab_size': 0,
                            'synced': False
                        }
            
            # Check colab configs
            if colab_config_path.exists():
                config_extensions = ['*.yaml', '*.yml', '*.json']
                for pattern in config_extensions:
                    for config_file in colab_config_path.glob(pattern):
                        colab_configs.append(config_file.name)
                        if config_file.name in config_details:
                            config_details[config_file.name]['colab_exists'] = True
                            config_details[config_file.name]['colab_size'] = config_file.stat().st_size
                            # Check if synced (same size)
                            config_details[config_file.name]['synced'] = (
                                config_details[config_file.name]['drive_size'] == 
                                config_details[config_file.name]['colab_size']
                            )
                        else:
                            config_details[config_file.name] = {
                                'drive_exists': False,
                                'drive_size': 0,
                                'colab_exists': True,
                                'colab_size': config_file.stat().st_size,
                                'synced': False
                            }
            
            # Count synced configs
            synced_count = sum(1 for details in config_details.values() if details['synced'])
            
            return {
                'drive_config_path': str(drive_config_path),
                'colab_config_path': str(colab_config_path),
                'drive_configs': drive_configs,
                'colab_configs': colab_configs,
                'total_configs': len(config_details),
                'synced_count': synced_count,
                'config_details': config_details,
                'all_synced': synced_count == len(config_details) and len(config_details) > 0
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'all_synced': False
            }
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get overall environment status."""
        try:
            drive_status = self.check_drive_status()
            folders_status = self.check_folders_status()
            symlinks_status = self.check_symlinks_status()
            configs_status = self.check_configs_status()
            
            # Calculate overall readiness
            checks = [
                drive_status.get('mounted', False),
                folders_status.get('all_exist', False),
                symlinks_status.get('all_valid', False),
                configs_status.get('all_synced', False)
            ]
            
            ready_count = sum(checks)
            total_checks = len(checks)
            readiness_percent = (ready_count / total_checks) * 100
            
            # Determine status
            if readiness_percent == 100:
                status = 'ready'
                message = 'Environment fully configured and ready'
            elif readiness_percent >= 75:
                status = 'mostly_ready'
                message = 'Environment mostly ready, minor issues'
            elif readiness_percent >= 50:
                status = 'partially_ready'
                message = 'Environment partially configured'
            else:
                status = 'not_ready'
                message = 'Environment requires setup'
            
            return {
                'status': status,
                'message': message,
                'readiness_percent': readiness_percent,
                'checks_passed': ready_count,
                'total_checks': total_checks,
                'details': {
                    'drive_ready': drive_status.get('mounted', False),
                    'folders_ready': folders_status.get('all_exist', False),
                    'symlinks_ready': symlinks_status.get('all_valid', False),
                    'configs_ready': configs_status.get('all_synced', False)
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Status check failed: {str(e)}',
                'readiness_percent': 0,
                'error': str(e)
            }
    
    def _get_folder_size(self, folder_path: Path) -> float:
        """Get folder size in MB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return round(total_size / (1024 * 1024), 2)
        except Exception:
            return 0.0
    
    def get_status_summary(self) -> str:
        """Get human-readable status summary."""
        try:
            overall = self.get_overall_status()
            drive = self.check_drive_status()
            folders = self.check_folders_status()
            symlinks = self.check_symlinks_status()
            configs = self.check_configs_status()
            
            summary = f"""
            🔍 Environment Status Summary:
            
            Overall: {overall['status'].upper()} ({overall['readiness_percent']:.0f}%)
            {overall['message']}
            
            📊 Component Status:
            • Drive: {'✅ Mounted' if drive.get('mounted') else '❌ Not mounted'}
            • Folders: {folders.get('existing_count', 0)}/{folders.get('total_required', 0)} created
            • Symlinks: {symlinks.get('existing_count', 0)}/{symlinks.get('total_required', 0)} valid
            • Configs: {configs.get('synced_count', 0)}/{configs.get('total_configs', 0)} synced
            """
            
            return summary.strip()
            
        except Exception as e:
            return f"❌ Status check failed: {str(e)}"