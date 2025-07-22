"""
Verify Operation (Optimized) - Enhanced Mixin Integration
Verify complete setup integrity with cross-module validation.
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from smartcash.common.environment import get_environment_manager
from .base_colab_operation import BaseColabOperation


class VerifyOperation(BaseColabOperation):
    """Optimized verify operation with enhanced mixin integration."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        super().__init__(operation_name, config, operation_container, **kwargs)
    
    def get_operations(self) -> Dict[str, Callable]:
        return {'verify_setup': self.execute_verify_setup}
    
    def execute_verify_setup(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Verify complete setup integrity with enhanced cross-module validation."""
        def execute_operation():
            progress_steps = self.get_progress_steps('verify')
            
            # Step 1: Initialize Verification Process (Direct Implementation)
            self.update_progress_safe(progress_callback, progress_steps[0]['progress'], progress_steps[0]['message'], progress_steps[0].get('phase_progress', 0))
            
            # Direct verification process without backend services
            self.log_info("Starting setup verification process...")
            init_result = {'success': True, 'message': 'Verification initialized directly'}
            
            # Step 2: Environment Verification
            self.update_progress_safe(progress_callback, progress_steps[1]['progress'], progress_steps[1]['message'], progress_steps[1].get('phase_progress', 0))
            
            env_verification = self._verify_environment()
            if not env_verification['success']:
                return self.create_error_result(f"Environment verification failed: {env_verification['errors']}")
            
            # Step 3: Path and Structure Verification
            self.update_progress_safe(progress_callback, progress_steps[2]['progress'], progress_steps[2]['message'], progress_steps[2].get('phase_progress', 0))
            
            structure_verification = self._verify_directory_structure()
            
            # Step 4: Cross-Module Verification
            self.update_progress_safe(progress_callback, progress_steps[3]['progress'], progress_steps[3]['message'], progress_steps[3].get('phase_progress', 0))
            
            cross_module_verification = self._verify_cross_module_integrity()
            
            # Step 5: Final Integrity Check
            self.update_progress_safe(progress_callback, progress_steps[4]['progress'], progress_steps[4]['message'], progress_steps[4].get('phase_progress', 0))
            
            integrity_check = self._perform_final_integrity_check(env_verification, structure_verification, cross_module_verification)
            
            return self.create_success_result(
                'Setup verification completed',
                environment_verification=env_verification,
                structure_verification=structure_verification,
                cross_module_verification=cross_module_verification,
                integrity_check=integrity_check,
                overall_success=integrity_check['success']
            )
        
        return self.execute_with_error_handling(execute_operation)
    
    def _verify_environment(self) -> Dict[str, Any]:
        """Verify environment setup and configuration."""
        try:
            # Get logger using mixin method and pass to environment manager
            logger = self._get_logger() if hasattr(self, '_get_logger') else None
            env_manager = get_environment_manager(logger=logger)
            system_info = env_manager.get_system_info()
            
            checks = []
            errors = []
            
            # Check environment type
            if env_manager.is_colab:
                checks.append({'check': 'environment_type', 'status': 'success', 'details': 'Running in Google Colab'})
                
                # Check drive mount if in Colab
                if env_manager.is_drive_mounted:
                    checks.append({'check': 'drive_mount', 'status': 'success', 'details': f"Drive mounted at {system_info.get('drive_path')}"})
                else:
                    checks.append({'check': 'drive_mount', 'status': 'warning', 'details': 'Google Drive not mounted'})
            else:
                checks.append({'check': 'environment_type', 'status': 'info', 'details': 'Running in local environment'})
            
            # Check memory
            memory_gb = system_info.get('total_memory_gb', 0)
            if memory_gb >= 8:
                checks.append({'check': 'memory', 'status': 'success', 'details': f'{memory_gb:.1f}GB RAM available'})
            elif memory_gb >= 4:
                checks.append({'check': 'memory', 'status': 'warning', 'details': f'{memory_gb:.1f}GB RAM (recommended: 8GB+)'})
            else:
                errors.append(f'Insufficient RAM: {memory_gb:.1f}GB (minimum: 4GB)')
                checks.append({'check': 'memory', 'status': 'error', 'details': f'{memory_gb:.1f}GB RAM insufficient'})
            
            # Check CUDA availability
            if system_info.get('cuda_available'):
                checks.append({'check': 'gpu', 'status': 'success', 'details': 'CUDA available'})
            else:
                checks.append({'check': 'gpu', 'status': 'info', 'details': 'No GPU/CUDA available'})
            
            return {
                'success': len(errors) == 0,
                'checks': checks,
                'errors': errors,
                'system_info': system_info
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'checks': [], 'errors': [str(e)]}
    
    def _verify_directory_structure(self) -> Dict[str, Any]:
        """Verify directory structure and symlinks."""
        try:
            paths_config = self.config.get('paths', {})
            colab_base = paths_config.get('colab_base', '/content')
            
            required_dirs = ['data', 'models', 'configs', 'outputs', 'logs']
            checks = []
            errors = []
            
            for dir_name in required_dirs:
                dir_path = os.path.join(colab_base, dir_name)
                
                if os.path.exists(dir_path):
                    if os.path.islink(dir_path):
                        # Verify symlink integrity
                        link_target = os.readlink(dir_path)
                        if os.path.exists(link_target):
                            checks.append({'check': f'symlink_{dir_name}', 'status': 'success', 'details': f'Symlink to {link_target}'})
                        else:
                            errors.append(f'Broken symlink: {dir_path} -> {link_target}')
                            checks.append({'check': f'symlink_{dir_name}', 'status': 'error', 'details': f'Broken symlink to {link_target}'})
                    else:
                        checks.append({'check': f'directory_{dir_name}', 'status': 'success', 'details': f'Directory exists: {dir_path}'})
                else:
                    errors.append(f'Missing directory: {dir_path}')
                    checks.append({'check': f'directory_{dir_name}', 'status': 'error', 'details': f'Missing: {dir_path}'})
            
            return {
                'success': len(errors) == 0,
                'checks': checks,
                'errors': errors,
                'base_directory': colab_base
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'checks': [], 'errors': [str(e)]}
    
    def _verify_cross_module_integrity(self) -> Dict[str, Any]:
        """Verify cross-module configuration integrity using mixins."""
        try:
            # Get configurations from related modules
            module_configs = self.get_module_configs(['model', 'dataset', 'setup'])
            
            # Validate cross-module configurations
            cross_validation = self.validate_cross_module_configs(['model', 'dataset', 'setup'])
            
            # Check configuration synchronization
            sync_check = self.check_configuration_sync(['model', 'dataset'])
            
            checks = []
            errors = []
            warnings = []
            
            # Process validation results
            if cross_validation['valid']:
                checks.append({'check': 'cross_module_config', 'status': 'success', 'details': 'All module configurations are compatible'})
            else:
                for warning in cross_validation.get('warnings', []):
                    warnings.append(warning)
                    checks.append({'check': 'cross_module_warning', 'status': 'warning', 'details': warning})
            
            # Process sync results
            if sync_check['synchronized']:
                checks.append({'check': 'config_sync', 'status': 'success', 'details': 'Configurations are synchronized'})
            else:
                for discrepancy in sync_check.get('discrepancies', []):
                    warnings.append(f"Config discrepancy: {discrepancy}")
                    checks.append({'check': 'config_sync_warning', 'status': 'warning', 'details': f"Sync issue: {discrepancy}"})
            
            return {
                'success': len(errors) == 0,
                'checks': checks,
                'errors': errors,
                'warnings': warnings,
                'module_configs': module_configs,
                'cross_validation': cross_validation,
                'sync_check': sync_check
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'checks': [], 'errors': [str(e)], 'warnings': []}
    
    def _perform_final_integrity_check(self, env_verification: Dict, structure_verification: Dict, cross_module_verification: Dict) -> Dict[str, Any]:
        """Perform final integrity check combining all verification results."""
        try:
            total_errors = (
                len(env_verification.get('errors', [])) +
                len(structure_verification.get('errors', [])) +
                len(cross_module_verification.get('errors', []))
            )
            
            total_warnings = (
                len(cross_module_verification.get('warnings', []))
            )
            
            total_checks = (
                len(env_verification.get('checks', [])) +
                len(structure_verification.get('checks', [])) +
                len(cross_module_verification.get('checks', []))
            )
            
            success_checks = sum(1 for verification in [env_verification, structure_verification, cross_module_verification] 
                               for check in verification.get('checks', []) 
                               if check.get('status') == 'success')
            
            overall_success = total_errors == 0
            setup_quality = 'excellent' if total_errors == 0 and total_warnings == 0 else \
                          'good' if total_errors == 0 and total_warnings <= 2 else \
                          'fair' if total_errors <= 1 else 'poor'
            
            return {
                'success': overall_success,
                'setup_quality': setup_quality,
                'total_checks': total_checks,
                'success_checks': success_checks,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'completion_percentage': (success_checks / total_checks * 100) if total_checks > 0 else 0,
                'summary': f'{success_checks}/{total_checks} checks passed, {total_errors} errors, {total_warnings} warnings'
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'setup_quality': 'unknown'}
    
    def get_progress_steps(self, operation_type: str = 'verify') -> list:
        """Get optimized progress steps for verification operation."""
        return [
            {'progress': 10, 'message': '🔧 Initializing verification...', 'phase_progress': 20},
            {'progress': 30, 'message': '🌍 Verifying environment...', 'phase_progress': 40},
            {'progress': 50, 'message': '📁 Checking directory structure...', 'phase_progress': 60},
            {'progress': 75, 'message': '🔗 Cross-module verification...', 'phase_progress': 80},
            {'progress': 100, 'message': '✅ Verification complete', 'phase_progress': 100}
        ]