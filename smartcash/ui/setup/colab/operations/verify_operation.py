"""
File: smartcash/ui/setup/colab/operations/verify_operation.py
Description: Comprehensive verification of the setup
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from .base_colab_operation import BaseColabOperation
from ..constants import SYMLINK_MAP, REQUIRED_FOLDERS
from ..utils.env_detector import detect_environment_info


class VerifyOperation(BaseColabOperation):
    """Comprehensive verification of the setup."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        """Initialize verify operation.
        
        Args:
            operation_name: Name of the operation
            config: Configuration dictionary
            operation_container: Optional operation container for UI integration
            **kwargs: Additional arguments
        """
        super().__init__(operation_name, config, operation_container, **kwargs)
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'verify_setup': self.execute_verify_setup
        }
    
    def execute_verify_setup(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Comprehensive verification of the setup.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with verification results
        """
        def execute_operation():
            progress_steps = self.get_progress_steps('verify')
            verification_results = {}
            issues = []
            
            # Step 1: Starting verification
            self.update_progress_safe(
                progress_callback, 
                progress_steps[0]['progress'], 
                progress_steps[0]['message'],
                progress_steps[0].get('phase_progress', 0)
            )
            
            # Only verify drive mount if explicitly requested in config
            if self.config.get('check_drive', False):
                # Verify Drive mount with check_drive=True
                drive_verification = self._verify_drive_mount()
                verification_results['drive_mount'] = drive_verification
                
                # Add drive status to the config for other components to use
                self.config['drive_mounted'] = drive_verification.get('mounted', False)
                self.config['drive_path'] = drive_verification.get('path')
                
                if not drive_verification.get('mounted', False):
                    issues.append("Google Drive is not mounted")
                elif not drive_verification.get('write_access', False):
                    issues.append("No write access to Google Drive")
            else:
                # Skip drive verification if not requested
                verification_results['drive_mount'] = {
                    'status': 'skipped',
                    'message': 'Drive verification skipped as check_drive=False'
                }
            
            # Step 2: Verifying symlinks
            self.update_progress_safe(
                progress_callback, 
                progress_steps[1]['progress'], 
                progress_steps[1]['message'],
                progress_steps[1].get('phase_progress', 0)
            )
            
            # Verify symlinks if drive is mounted
            if self.config.get('drive_mounted', False):
                symlink_verification = self.verify_symlinks_batch(SYMLINK_MAP)
                verification_results['symlinks'] = symlink_verification
                issues.extend(symlink_verification['issues'])
            else:
                # Skip symlink verification if drive is not mounted
                verification_results['symlinks'] = {
                    'status': 'skipped',
                    'message': 'Symlink verification skipped as drive is not mounted'
                }
            
            # Step 3: Verifying folders
            self.update_progress_safe(
                progress_callback, 
                progress_steps[2]['progress'], 
                progress_steps[2]['message'],
                progress_steps[2].get('phase_progress', 0)
            )
            
            # Verify folders
            folder_verification = self.validate_items_exist(REQUIRED_FOLDERS, "folder")
            verification_results['folders'] = folder_verification
            if not folder_verification['all_exist']:
                issues.extend([f"Missing folder: {folder}" for folder in folder_verification['missing_items']])
            
            # Step 4: Verifying environment variables
            self.update_progress_safe(
                progress_callback, 
                progress_steps[3]['progress'], 
                progress_steps[3]['message'],
                progress_steps[3].get('phase_progress', 0)
            )
            
            # Verify environment variables
            required_vars = ['SMARTCASH_ROOT', 'SMARTCASH_ENV', 'SMARTCASH_DATA_ROOT']
            env_verification = self.verify_environment_variables(required_vars)
            verification_results['env_vars'] = env_verification
            issues.extend(env_verification['issues'])
            
            # Step 5: Gathering system info
            self.update_progress_safe(
                progress_callback, 
                progress_steps[4]['progress'], 
                progress_steps[4]['message'],
                progress_steps[4].get('phase_progress', 0)
            )
            
            # Enhanced system information using env_detector
            system_info = self._get_enhanced_system_info()
            verification_results['system_info'] = system_info
            
            # Step 6: Verification complete
            self.update_progress_safe(
                progress_callback, 
                progress_steps[5]['progress'], 
                progress_steps[5]['message'],
                progress_steps[5].get('phase_progress', 0)
            )
            
            # Determine overall success
            all_verified = len(issues) == 0
            
            return self.create_success_result(
                'All verifications passed' if all_verified else f'{len(issues)} issues found',
                verification=verification_results,
                issues=issues,
                system_info=system_info,
                summary=self.create_verification_summary(verification_results, issues)
            )
            
        return self.execute_with_error_handling(execute_operation)
    
    def _verify_drive_mount(self) -> Dict[str, Any]:
        """Verify Google Drive mount status.
        
        Returns:
            Dictionary with drive mount verification results
        """
        # Use detect_environment_info with check_drive=True to get drive status
        env_info = detect_environment_info(check_drive=True)
        
        drive_mounted = env_info.get('drive_mounted', False)
        drive_path = env_info.get('drive_mount_path', '/content/drive')
        mydrive_path = os.path.join(drive_path, 'MyDrive') if drive_path else '/content/drive/MyDrive'
        
        # Get write access status from the environment info if available
        write_access = False
        if drive_mounted and drive_path:
            # Test write access if not already available in env_info
            if 'drive_write_access' in env_info:
                write_access = env_info['drive_write_access']
            else:
                try:
                    test_file = os.path.join(mydrive_path, '.smartcash_verify_test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    write_access = True
                except Exception as e:
                    self.logger.warning(f"Failed to verify write access to {mydrive_path}: {str(e)}")
        
        return {
            'mounted': drive_mounted,
            'path': drive_path if drive_mounted else None,
            'mydrive_path': mydrive_path if drive_mounted else None,
            'write_access': write_access,
            'accessible': drive_mounted and write_access,
            'status': 'verified' if drive_mounted and write_access else 'not_verified'
        }
    
    
    
    
    def _get_enhanced_system_info(self) -> Dict[str, Any]:
        """Get enhanced system information using env_detector.
        
        Returns:
            Dictionary with comprehensive system information
        """
        try:
            # Use enhanced env_detector for comprehensive info
            env_info = self.detect_environment_enhanced()
            
            # Extract and format system information
            return self.format_system_info(env_info)
            
        except Exception as e:
            self.logger.warning(f"Error getting enhanced system info: {str(e)}")
            return {'error': str(e)}
    
