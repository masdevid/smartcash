"""
File: smartcash/ui/setup/env_config/handlers/status_handler.py
Deskripsi: Handler untuk comprehensive status checking dan validation
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.utils.environment_helpers import validate_config_completeness

class StatusHandler:
    """📊 Handler untuk comprehensive status checking"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.env_handler = EnvironmentHandler(logger)
        self.drive_handler = DriveHandler(logger)
        
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """🔍 Get comprehensive environment status"""
        try:
            # Environment status
            env_status = self.env_handler.get_environment_status()
            
            # Drive status
            drive_status = self.drive_handler.get_drive_status()
            
            # Config validation
            config_validation = validate_config_completeness()
            
            # Symlink validation
            symlink_status = self.env_handler.validate_symlinks()
            
            # System info
            system_info = self.env_handler.get_system_info()
            
            # Overall readiness calculation
            overall_ready = self._calculate_overall_readiness(
                env_status, drive_status, config_validation, symlink_status
            )
            
            comprehensive_status = {
                'ready': overall_ready,
                'environment': env_status,
                'drive': drive_status,
                'configs': config_validation,
                'symlinks': symlink_status,
                'system': system_info,
                'summary': self._generate_status_summary(
                    env_status, drive_status, config_validation, symlink_status
                )
            }
            
            if self.logger:
                self._log_comprehensive_status(comprehensive_status)
                
            return comprehensive_status
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error getting comprehensive status: {str(e)}")
            return {
                'ready': False,
                'error': str(e),
                'summary': {'error': 'Status check failed'}
            }
    
    def check_setup_requirements(self) -> Dict[str, Any]:
        """🔧 Check what setup steps are needed"""
        status = self.get_comprehensive_status()
        
        requirements = {
            'drive_mount_needed': not status.get('drive', {}).get('drive_mounted', False),
            'folders_needed': not status.get('drive', {}).get('smartcash_folder_exists', False),
            'configs_needed': not status.get('configs', {}).get('is_complete', False),
            'symlinks_needed': not status.get('symlinks', {}).get('valid', False)
        }
        
        # Priority order untuk setup steps
        setup_steps = []
        if requirements['drive_mount_needed']:
            setup_steps.append('mount_drive')
        if requirements['folders_needed']:
            setup_steps.append('create_folders')
        if requirements['configs_needed']:
            setup_steps.append('sync_configs')
        if requirements['symlinks_needed']:
            setup_steps.append('create_symlinks')
            
        return {
            'setup_needed': len(setup_steps) > 0,
            'requirements': requirements,
            'setup_steps': setup_steps,
            'estimated_steps': len(setup_steps)
        }
    
    def validate_post_setup(self) -> Dict[str, Any]:
        """✅ Validate environment after setup completion"""
        try:
            # Re-check comprehensive status
            status = self.get_comprehensive_status()
            
            # Specific validations
            validations = {
                'drive_accessible': self._validate_drive_access(),
                'folders_complete': self._validate_folder_structure(),
                'configs_available': self._validate_config_availability(),
                'symlinks_functional': self._validate_symlink_functionality()
            }
            
            overall_valid = all(validations.values()) and status.get('ready', False)
            
            validation_result = {
                'valid': overall_valid,
                'validations': validations,
                'status': status,
                'issues': self._identify_validation_issues(validations)
            }
            
            if self.logger:
                if overall_valid:
                    self.logger.info("✅ Post-setup validation passed")
                else:
                    self.logger.warning("⚠️ Post-setup validation found issues")
                    
            return validation_result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Post-setup validation failed: {str(e)}")
            return {'valid': False, 'error': str(e)}
    
    def _calculate_overall_readiness(self, env_status: Dict[str, Any], 
                                   drive_status: Dict[str, Any],
                                   config_validation: Dict[str, Any],
                                   symlink_status: Dict[str, Any]) -> bool:
        """Calculate overall environment readiness"""
        return all([
            env_status.get('ready', False),
            drive_status.get('ready', False),
            config_validation.get('is_complete', False),
            symlink_status.get('valid', False)
        ])
    
    def _generate_status_summary(self, env_status: Dict[str, Any],
                               drive_status: Dict[str, Any], 
                               config_validation: Dict[str, Any],
                               symlink_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate readable status summary"""
        return {
            'environment_ready': env_status.get('ready', False),
            'drive_ready': drive_status.get('ready', False),
            'configs_complete': config_validation.get('is_complete', False),
            'symlinks_valid': symlink_status.get('valid', False),
            'total_configs': config_validation.get('total_configs', 0),
            'essential_configs': len(config_validation.get('essential_configs', [])),
            'valid_symlinks': symlink_status.get('valid_count', 0)
        }
    
    def _validate_drive_access(self) -> bool:
        """Validate Drive access"""
        drive_validation = self.env_handler.validate_drive_access()
        return drive_validation.get('accessible', False) and drive_validation.get('writable', False)
    
    def _validate_folder_structure(self) -> bool:
        """Validate folder structure"""
        drive_status = self.drive_handler.get_drive_status()
        folder_status = drive_status.get('folder_status', {})
        return all(folder_status.values()) if folder_status else False
    
    def _validate_config_availability(self) -> bool:
        """Validate config availability"""
        config_validation = validate_config_completeness()
        return config_validation.get('is_complete', False)
    
    def _validate_symlink_functionality(self) -> bool:
        """Validate symlink functionality"""
        symlink_status = self.env_handler.validate_symlinks()
        return symlink_status.get('valid', False)
    
    def _identify_validation_issues(self, validations: Dict[str, bool]) -> List[str]:
        """Identify specific validation issues"""
        issues = []
        for validation_name, is_valid in validations.items():
            if not is_valid:
                issues.append(validation_name.replace('_', ' ').title())
        return issues
    
    def _log_comprehensive_status(self, status: Dict[str, Any]) -> None:
        """Log comprehensive status dengan detail"""
        if not self.logger:
            return
            
        self.logger.info("📊 Comprehensive Environment Status:")
        
        summary = status.get('summary', {})
        self.logger.info(f"  🌍 Environment: {'✅' if summary.get('environment_ready') else '❌'}")
        self.logger.info(f"  📱 Drive: {'✅' if summary.get('drive_ready') else '❌'}")
        self.logger.info(f"  📋 Configs: {'✅' if summary.get('configs_complete') else '❌'} ({summary.get('total_configs', 0)} total)")
        self.logger.info(f"  🔗 Symlinks: {'✅' if summary.get('symlinks_valid') else '❌'} ({summary.get('valid_symlinks', 0)} valid)")
        self.logger.info(f"  ✅ Overall Ready: {'✅' if status.get('ready') else '❌'}")