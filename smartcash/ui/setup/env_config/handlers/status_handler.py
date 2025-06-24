"""
File: smartcash/ui/setup/env_config/handlers/status_handler.py
Deskripsi: Fixed handler untuk comprehensive status checking dengan SystemInfoHandler integration
"""

from typing import Dict, Any, List
from smartcash.ui.setup.env_config.handlers.environment_handler import EnvironmentHandler
from smartcash.ui.setup.env_config.handlers.drive_handler import DriveHandler
from smartcash.ui.setup.env_config.handlers.system_info_handler import SystemInfoHandler
from smartcash.ui.setup.env_config.utils.environment_helpers import validate_config_completeness

class StatusHandler:
    """📊 Fixed handler untuk comprehensive status checking"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.env_handler = EnvironmentHandler(logger)
        self.drive_handler = DriveHandler(logger)
        self.system_info_handler = SystemInfoHandler(logger)
        
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """🔍 Get comprehensive environment status dengan SystemInfoHandler integration"""
        try:
            # Environment status dengan fallback
            try:
                env_status = self.env_handler.get_environment_status()
            except Exception:
                env_status = {'ready': False, 'error': 'Environment check failed'}
            
            # Drive status dengan fallback
            try:
                drive_status = self.drive_handler.get_drive_status()
            except Exception:
                drive_status = {'ready': False, 'error': 'Drive check failed'}
            
            # Config validation dengan fallback
            try:
                config_validation = validate_config_completeness()
            except Exception:
                config_validation = {'is_complete': False, 'total_configs': 0}
            
            # Symlink validation dengan fallback
            try:
                symlink_status = self.env_handler.validate_symlinks()
            except Exception:
                symlink_status = {'valid': False, 'valid_count': 0}
            
            # System info dengan fallback - menggunakan SystemInfoHandler
            try:
                system_info = self.system_info_handler.env_manager.get_system_info()
            except Exception:
                system_info = {'environment': 'Unknown', 'python_version': 'Unknown'}
            
            # Calculate overall readiness
            overall_ready = self._calculate_overall_readiness(
                env_status, drive_status, config_validation, symlink_status
            )
            
            # Generate summary
            summary = self._generate_status_summary(
                env_status, drive_status, config_validation, symlink_status
            )
            
            # Compile comprehensive status
            comprehensive_status = {
                'ready': overall_ready,
                'environment': env_status,
                'drive': drive_status,
                'configs': config_validation,
                'symlinks': symlink_status,
                'system': system_info,
                'summary': summary,
                'timestamp': self._get_timestamp()
            }
            
            # Log comprehensive status
            self._log_comprehensive_status(comprehensive_status)
            
            return comprehensive_status
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Comprehensive status check failed: {str(e)}")
            return self._get_fallback_status(str(e))
    
    def validate_post_setup(self) -> Dict[str, Any]:
        """✅ Validate environment after setup completion"""
        try:
            if self.logger:
                self.logger.info("🔍 Validating post-setup environment...")
            
            # Run all validations
            validations = {
                'environment_ready': self._validate_environment_readiness(),
                'drive_accessible': self._validate_drive_access(),
                'folders_created': self._validate_folder_structure(),
                'configs_available': self._validate_config_availability(),
                'symlinks_functional': self._validate_symlink_functionality()
            }
            
            # Check overall validation
            all_valid = all(validations.values())
            
            # Identify issues
            issues = self._identify_validation_issues(validations)
            
            validation_result = {
                'valid': all_valid,
                'validations': validations,
                'issues': issues,
                'timestamp': self._get_timestamp()
            }
            
            if self.logger:
                if all_valid:
                    self.logger.info("✅ Post-setup validation passed")
                else:
                    self.logger.warning(f"⚠️ Post-setup validation found issues: {', '.join(issues)}")
                    
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
        """Generate readable status summary untuk UI updates"""
        return {
            'environment_ready': env_status.get('ready', False),
            'drive_ready': drive_status.get('ready', False),
            'configs_complete': config_validation.get('is_complete', False),
            'symlinks_valid': symlink_status.get('valid', False),
            'directories_ready': drive_status.get('folder_status', {}).get('all_created', False),
            'total_configs': config_validation.get('total_configs', 0),
            'essential_configs': len(config_validation.get('essential_configs', [])),
            'valid_symlinks': symlink_status.get('valid_count', 0)
        }
    
    def _validate_environment_readiness(self) -> bool:
        """Validate basic environment readiness"""
        env_status = self.env_handler.get_environment_status()
        return env_status.get('ready', False)
    
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
    
    def _get_fallback_status(self, error_msg: str) -> Dict[str, Any]:
        """Get fallback status untuk error cases"""
        return {
            'ready': False,
            'error': error_msg,
            'environment': {'ready': False, 'error': 'Check failed'},
            'drive': {'ready': False, 'error': 'Check failed'},
            'configs': {'is_complete': False, 'total_configs': 0},
            'symlinks': {'valid': False, 'valid_count': 0},
            'system': {'environment': 'Unknown'},
            'summary': {
                'environment_ready': False,
                'drive_ready': False,
                'configs_complete': False,
                'symlinks_valid': False,
                'directories_ready': False
            },
            'timestamp': self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")