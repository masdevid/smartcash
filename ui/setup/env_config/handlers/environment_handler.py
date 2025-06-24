"""
File: smartcash/ui/setup/env_config/handlers/environment_handler.py
Deskripsi: Handler untuk environment management dan validation
"""

from typing import Dict, Any, Optional
from smartcash.ui.setup.env_config.utils.environment_helpers import (
    get_environment_info, create_directories_if_missing, check_symlink_validity
)
from smartcash.ui.setup.env_config.constants import REQUIRED_FOLDERS, RETRY_CONFIG

class EnvironmentHandler:
    """🌍 Handler untuk environment management"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.environment_info = None
        
    def get_environment_status(self) -> Dict[str, Any]:
        """🔍 Get comprehensive environment status"""
        try:
            self.environment_info = get_environment_info()
            
            status = {
                'is_colab': self.environment_info['is_colab'],
                'drive_mounted': self.environment_info['drive_mounted'],
                'config_ready': self.environment_info['config_source_exists'],
                'configs_complete': self.environment_info['essential_configs_complete'],
                'python_ready': True,  # Selalu True di Colab
                'ready': self._calculate_overall_readiness()
            }
            
            if self.logger:
                self._log_environment_status(status)
                
            return status
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error getting environment status: {str(e)}")
            return {'ready': False, 'error': str(e)}
    
    def validate_drive_access(self) -> Dict[str, Any]:
        """📱 Validate Google Drive access"""
        try:
            from pathlib import Path
            
            drive_path = Path('/content/drive/MyDrive')
            if not drive_path.exists():
                return {'accessible': False, 'reason': 'Drive not mounted'}
            
            # Test write access
            test_file = drive_path / '.smartcash_test'
            try:
                test_file.write_text('test')
                test_file.unlink()
                write_access = True
            except Exception:
                write_access = False
            
            return {
                'accessible': True,
                'writable': write_access,
                'path': str(drive_path)
            }
            
        except Exception as e:
            return {'accessible': False, 'error': str(e)}
    
    def prepare_environment_directories(self) -> Dict[str, Any]:
        """📁 Prepare required directories"""
        try:
            from pathlib import Path
            
            base_path = Path('/content/drive/MyDrive/SmartCash')
            directories_to_create = [str(base_path / folder) for folder in REQUIRED_FOLDERS]
            
            results = create_directories_if_missing(directories_to_create)
            
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            if self.logger:
                self.logger.info(f"📁 Created {success_count}/{total_count} directories")
                
            return {
                'success': success_count == total_count,
                'created': success_count,
                'total': total_count,
                'results': results
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error preparing directories: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def validate_symlinks(self) -> Dict[str, Any]:
        """🔗 Validate existing symlinks"""
        try:
            from pathlib import Path
            
            base_path = Path('/content')
            symlinks_to_check = {
                'data': base_path / 'data',
                'configs': base_path / 'configs', 
                'models': base_path / 'models',
                'logs': base_path / 'logs'
            }
            
            results = {}
            for name, path in symlinks_to_check.items():
                results[name] = check_symlink_validity(str(path))
            
            valid_count = sum(1 for valid in results.values() if valid)
            total_count = len(results)
            
            return {
                'valid': valid_count == total_count,
                'valid_count': valid_count,
                'total_count': total_count,
                'results': results
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error validating symlinks: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """💻 Get system information summary"""
        if not self.environment_info:
            self.environment_info = get_environment_info()
        
        return {
            'platform': 'Google Colab' if self.environment_info['is_colab'] else 'Local',
            'python_version': self.environment_info['python_version'],
            'working_directory': self.environment_info['working_directory'],
            'configs_found': self.environment_info['total_configs_found']
        }
    
    def _calculate_overall_readiness(self) -> bool:
        """Calculate overall environment readiness"""
        if not self.environment_info:
            return False
            
        return all([
            self.environment_info.get('is_colab', False),
            self.environment_info.get('drive_mounted', False),
            self.environment_info.get('config_source_exists', False),
            self.environment_info.get('essential_configs_complete', False)
        ])
    
    def _log_environment_status(self, status: Dict[str, Any]) -> None:
        """Log environment status dengan emoji"""
        if not self.logger or not status:
            return
            
        self.logger.info("🔍 Environment Status:")
        self.logger.info(f"  📱 Colab Environment: {'✅' if status.get('is_colab') else '❌'}")
        self.logger.info(f"  💾 Drive Mounted: {'✅' if status.get('drive_mounted') else '❌'}")
        self.logger.info(f"  📋 Config Ready: {'✅' if status.get('config_ready') else '❌'}")
        self.logger.info(f"  🔧 Overall Ready: {'✅' if status.get('ready') else '❌'}")