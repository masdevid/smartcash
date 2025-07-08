"""
File: smartcash/ui/setup/colab/operations/verify_operation.py
Description: Comprehensive verification of the setup
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..constants import SYMLINK_MAP, REQUIRED_FOLDERS
from ..utils.env_detector import detect_environment_info


class VerifyOperation(OperationHandler):
    """Comprehensive verification of the setup."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize verify operation.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='verify_operation',
            parent_module='colab',
            **kwargs
        )
        self.config = config
    
    def initialize(self) -> None:
        """Initialize the verify operation."""
        self.logger.info("🚀 Initializing verify operation")
        # No specific initialization needed for verify operation
        self.logger.info("✅ Verify operation initialization complete")
    
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
        try:
            if progress_callback:
                progress_callback(10, "🔍 Verifying mount status...")
            
            verification_results = {}
            issues = []
            
            # Verify Drive mount
            drive_verification = self._verify_drive_mount()
            verification_results['drive_mount'] = drive_verification
            if not drive_verification['mounted']:
                issues.append("Google Drive not mounted")
            else:
                self.log("✅ Google Drive mount verified", 'info')
            
            if progress_callback:
                progress_callback(30, "🔗 Verifying symlinks...")
            
            # Verify symlinks
            symlink_verification = self._verify_symlinks()
            verification_results['symlinks'] = symlink_verification
            issues.extend(symlink_verification['issues'])
            
            if progress_callback:
                progress_callback(50, "📁 Verifying folders...")
            
            # Verify folders
            folder_verification = self._verify_folders()
            verification_results['folders'] = folder_verification
            issues.extend(folder_verification['issues'])
            
            if progress_callback:
                progress_callback(70, "🌍 Verifying environment variables...")
            
            # Verify environment variables
            env_verification = self._verify_environment_variables()
            verification_results['env_vars'] = env_verification
            issues.extend(env_verification['issues'])
            
            if progress_callback:
                progress_callback(90, "💻 Gathering system info...")
            
            # Enhanced system information using env_detector
            system_info = self._get_enhanced_system_info()
            verification_results['system_info'] = system_info
            
            if progress_callback:
                progress_callback(100, "✅ Verification complete")
            
            # Determine overall success
            all_verified = len(issues) == 0
            
            return {
                'success': all_verified,
                'verification': verification_results,
                'issues': issues,
                'system_info': system_info,
                'summary': self._generate_verification_summary(verification_results, issues),
                'message': 'All verifications passed' if all_verified else f'{len(issues)} issues found'
            }
            
        except Exception as e:
            self.log(f"Verification failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Verification failed: {str(e)}'
            }
    
    def _verify_drive_mount(self) -> Dict[str, Any]:
        """Verify Google Drive mount status.
        
        Returns:
            Dictionary with drive mount verification results
        """
        drive_path = '/content/drive'
        mydrive_path = '/content/drive/MyDrive'
        
        mounted = os.path.exists(mydrive_path)
        write_access = False
        
        if mounted:
            # Test write access
            try:
                test_file = os.path.join(mydrive_path, '.smartcash_verify_test')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                write_access = True
            except Exception:
                pass
        
        return {
            'mounted': mounted,
            'path': drive_path if mounted else None,
            'mydrive_path': mydrive_path if mounted else None,
            'write_access': write_access,
            'accessible': mounted and write_access
        }
    
    def _verify_symlinks(self) -> Dict[str, Any]:
        """Verify symbolic links.
        
        Returns:
            Dictionary with symlink verification results
        """
        symlink_status = {}
        issues = []
        valid_count = 0
        
        for source, target in SYMLINK_MAP.items():
            exists = os.path.islink(target) and os.path.exists(target)
            valid = False
            
            if exists:
                try:
                    valid = os.path.samefile(source, target)
                    if valid:
                        valid_count += 1
                except Exception:
                    pass
            
            symlink_status[target] = {
                'exists': exists,
                'source': source,
                'valid': valid,
                'target_basename': os.path.basename(target)
            }
            
            if not exists:
                issues.append(f"Symlink missing: {target}")
            elif not valid:
                issues.append(f"Symlink invalid: {target}")
        
        self.log(f"✅ {valid_count}/{len(SYMLINK_MAP)} symlinks verified", 'info')
        
        return {
            'symlink_status': symlink_status,
            'valid_count': valid_count,
            'total_count': len(SYMLINK_MAP),
            'all_valid': valid_count == len(SYMLINK_MAP),
            'issues': issues
        }
    
    def _verify_folders(self) -> Dict[str, Any]:
        """Verify required folders.
        
        Returns:
            Dictionary with folder verification results
        """
        folder_status = {}
        issues = []
        existing_count = 0
        
        for folder in REQUIRED_FOLDERS:
            exists = os.path.exists(folder)
            is_dir = os.path.isdir(folder) if exists else False
            writable = False
            
            if exists and is_dir:
                existing_count += 1
                # Test write access
                try:
                    test_file = os.path.join(folder, '.smartcash_write_test')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    writable = True
                except Exception:
                    pass
            
            folder_status[folder] = {
                'exists': exists,
                'is_directory': is_dir,
                'writable': writable,
                'basename': os.path.basename(folder)
            }
            
            if not exists:
                issues.append(f"Required folder missing: {folder}")
            elif not is_dir:
                issues.append(f"Path exists but not a directory: {folder}")
            elif not writable:
                issues.append(f"Folder not writable: {folder}")
        
        self.log(f"✅ {existing_count}/{len(REQUIRED_FOLDERS)} folders verified", 'info')
        
        return {
            'folder_status': folder_status,
            'existing_count': existing_count,
            'total_count': len(REQUIRED_FOLDERS),
            'all_exist': existing_count == len(REQUIRED_FOLDERS),
            'issues': issues
        }
    
    def _verify_environment_variables(self) -> Dict[str, Any]:
        """Verify environment variables.
        
        Returns:
            Dictionary with environment variable verification results
        """
        required_vars = ['SMARTCASH_ROOT', 'SMARTCASH_ENV', 'SMARTCASH_DATA_ROOT']
        env_vars_status = {}
        issues = []
        valid_count = 0
        
        for var in required_vars:
            exists = var in os.environ
            value = os.environ.get(var) if exists else None
            valid_path = False
            
            if exists and value:
                valid_count += 1
                # For path variables, check if they exist
                if 'ROOT' in var or 'PATH' in var:
                    valid_path = os.path.exists(value)
            
            env_vars_status[var] = {
                'exists': exists,
                'value': value,
                'valid_path': valid_path if 'ROOT' in var or 'PATH' in var else None
            }
            
            if not exists:
                issues.append(f"Environment variable missing: {var}")
            elif 'ROOT' in var and not valid_path:
                issues.append(f"Environment variable path invalid: {var} = {value}")
        
        self.log(f"✅ Environment variables verified", 'info')
        
        return {
            'env_vars_status': env_vars_status,
            'valid_count': valid_count,
            'total_count': len(required_vars),
            'all_valid': valid_count == len(required_vars),
            'python_path': os.environ.get('PYTHONPATH', ''),
            'issues': issues
        }
    
    def _get_enhanced_system_info(self) -> Dict[str, Any]:
        """Get enhanced system information using env_detector.
        
        Returns:
            Dictionary with comprehensive system information
        """
        try:
            # Use enhanced env_detector for comprehensive info
            env_info = detect_environment_info()
            
            # Extract and format system information
            system_info = {
                'environment': {
                    'type': env_info.get('runtime', {}).get('type', 'unknown'),
                    'is_colab': env_info.get('is_colab', False),
                    'display': env_info.get('runtime', {}).get('display', 'Unknown')
                },
                'hardware': {
                    'cpu_cores': env_info.get('cpu_cores', 'Unknown'),
                    'total_ram_gb': round(env_info.get('total_ram', 0) / (1024**3), 2) if env_info.get('total_ram') else 0,
                    'gpu_info': env_info.get('gpu', 'No GPU available')
                },
                'operating_system': env_info.get('os', {}),
                'storage': env_info.get('storage_info', {}),
                'drive_status': {
                    'mounted': env_info.get('drive_mounted', False),
                    'mount_path': env_info.get('drive_mount_path', '')
                },
                'python': {
                    'version': env_info.get('python_version', 'Unknown'),
                    'executable': os.sys.executable
                }
            }
            
            return system_info
            
        except Exception as e:
            self.log(f"Error getting enhanced system info: {str(e)}", 'warning')
            return {'error': str(e)}
    
    def _generate_verification_summary(self, verification_results: Dict[str, Any], issues: list) -> Dict[str, Any]:
        """Generate a summary of verification results.
        
        Args:
            verification_results: Complete verification results
            issues: List of issues found
            
        Returns:
            Dictionary with verification summary
        """
        summary = {
            'overall_status': 'PASS' if len(issues) == 0 else 'FAIL',
            'total_issues': len(issues),
            'components': {}
        }
        
        # Drive mount summary
        drive_mount = verification_results.get('drive_mount', {})
        summary['components']['drive_mount'] = {
            'status': 'PASS' if drive_mount.get('mounted', False) else 'FAIL',
            'details': 'Mounted and accessible' if drive_mount.get('accessible', False) else 'Not mounted or not accessible'
        }
        
        # Symlinks summary
        symlinks = verification_results.get('symlinks', {})
        symlink_status = 'PASS' if symlinks.get('all_valid', False) else 'FAIL'
        valid_count = symlinks.get('valid_count', 0)
        total_count = symlinks.get('total_count', 0)
        summary['components']['symlinks'] = {
            'status': symlink_status,
            'details': f'{valid_count}/{total_count} symlinks valid'
        }
        
        # Folders summary
        folders = verification_results.get('folders', {})
        folder_status = 'PASS' if folders.get('all_exist', False) else 'FAIL'
        existing_count = folders.get('existing_count', 0)
        total_folders = folders.get('total_count', 0)
        summary['components']['folders'] = {
            'status': folder_status,
            'details': f'{existing_count}/{total_folders} folders exist'
        }
        
        # Environment variables summary
        env_vars = verification_results.get('env_vars', {})
        env_status = 'PASS' if env_vars.get('all_valid', False) else 'FAIL'
        valid_env_count = env_vars.get('valid_count', 0)
        total_env_count = env_vars.get('total_count', 0)
        summary['components']['environment_variables'] = {
            'status': env_status,
            'details': f'{valid_env_count}/{total_env_count} variables valid'
        }
        
        return summary