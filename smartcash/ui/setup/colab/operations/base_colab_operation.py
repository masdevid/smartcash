"""
Base class for all Colab operations to eliminate code duplication.
"""

import os
from typing import Dict, Any, Optional, Callable, List, Tuple
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from smartcash.ui.components.operation_container import OperationContainer
from ..utils.env_detector import detect_environment_info


class BaseColabOperation(LoggingMixin, OperationMixin):
    """Base class for all Colab operations with common functionality."""
    
    def __init__(self, 
                 operation_name: str, 
                 config: Dict[str, Any], 
                 operation_container: Optional[OperationContainer] = None, 
                 **kwargs):
        """Initialize base colab operation.
        
        Args:
            operation_name: Name of the operation (e.g., 'init_operation')
            config: Configuration dictionary
            operation_container: Optional operation container for UI integration
            **kwargs: Additional arguments
        """
        # Initialize mixins
        super().__init__()
        
        # Core operation state
        self.config = config
        self.operation_container = operation_container
        
        # Module identification for mixins
        self.module_name = operation_name
        self.parent_module = 'colab'
        
        # Set operation container for logging
        if operation_container:
            self._operation_container = operation_container
        
        self.log(f"✅ {operation_name} initialized", 'debug')
    
    def initialize(self) -> None:
        """Initialize the operation."""
        operation_display_name = self.module_name.replace('_', ' ').title()
        self.log(f"🚀 Initializing {operation_display_name}", 'info')
        
        # Call subclass-specific initialization if needed
        self._initialize_operation()
        
        self.log(f"✅ {operation_display_name} initialization complete", 'info')
    
    def _initialize_operation(self) -> None:
        """Override in subclasses for specific initialization logic."""
        pass
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations. Override in subclasses."""
        return {}
    
    def execute_with_error_handling(self, operation_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Execute operation with standardized error handling.
        
        Args:
            operation_func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Dictionary with operation results including error details and traceback if any
        """
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            import traceback
            import sys
            
            # Format operation name for display
            operation_name = self.module_name.replace('_', ' ').title()
            
            # Get exception details
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_msg = str(exc_value) or "Unknown error occurred"
            
            # Format full error message with operation context
            full_error_msg = f"{operation_name} failed: {error_msg}"
            
            # Get full traceback with chained exceptions
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            
            # Log the error with traceback using available logging methods
            if hasattr(self, 'log_error'):
                # Format error message with traceback for logging
                error_log = f"❌ {full_error_msg}\n\nTraceback (paling baru terakhir):\n{tb_str}"
                self.log_error(error_log)
            elif hasattr(self, 'log'):
                # Fallback to standard log method if log_error is not available
                self.log(f"❌ {full_error_msg}\n\n{tb_str}", 'error')
            
            # Also log to the standard logger
            if hasattr(self, 'logger'):
                self.log(full_error_msg, 'error', exc_info=True)
            
            # Return detailed error information
            return self.create_error_result(
                full_error_msg,
                traceback=tb_str,
                exception_type=exc_type.__name__ if exc_type else 'UnknownError',
                operation=self.module_name
            )
    
    def update_progress_safe(self, progress_callback: Optional[Callable], 
                           progress: int, message: str, phase_progress: Optional[int] = None) -> None:
        """Safely update progress with callback.
        
        Args:
            progress_callback: Optional callback function
            progress: Overall progress percentage (0-100)
            message: Progress message
            phase_progress: Optional phase progress percentage (0-100)
            
        Note:
            When phase_progress is provided, this will update both the overall progress bar
            and the phase progress bar. The progress_callback should accept either:
            - progress, message (for single progress bar)
            - progress, message, phase_progress (for dual progress bars)
        """
        if not progress_callback:
            return
            
        try:
            if phase_progress is not None:
                # Update both overall and phase progress with proper level information
                if hasattr(progress_callback, 'update_progress'):
                    # If it's an OperationContainer or similar with update_progress method
                    progress_callback.update_progress(
                        progress=progress,
                        message=message,
                        level='primary'
                    )
                    progress_callback.update_progress(
                        progress=phase_progress,
                        message=f"{message} (Phase Progress)",
                        level='secondary'
                    )
                else:
                    # Fallback to direct callback with three arguments
                    try:
                        progress_callback(progress, message, phase_progress)
                    except TypeError:
                        # If callback doesn't accept three arguments, try with just overall progress
                        progress_callback(progress, message)
            else:
                # Update only overall progress
                if hasattr(progress_callback, 'update_progress'):
                    progress_callback.update_progress(
                        progress=progress,
                        message=message,
                        level='primary'
                    )
                else:
                    progress_callback(progress, message)
        except Exception as e:
            self.log(f"Progress update failed: {e}", 'warning')
            # Try with just the basic callback as last resort
            try:
                progress_callback(progress, message)
            except Exception as e2:
                self.log(f"Fallback progress update also failed: {e2}", 'warning')
    
    def create_success_result(self, message: str, **additional_data) -> Dict[str, Any]:
        """Create standardized success result.
        
        Args:
            message: Success message
            **additional_data: Additional data to include in result
            
        Returns:
            Success result dictionary
        """
        result = {
            'success': True,
            'message': message
        }
        result.update(additional_data)
        return result
    
    def create_error_result(self, error: str, **additional_data) -> Dict[str, Any]:
        """Create standardized error result.
        
        Args:
            error: Error message
            **additional_data: Additional data to include in result
            
        Returns:
            Error result dictionary with error details and traceback
        """
        # Create base result with error information
        result = {
            'success': False,
            'error': error,
            'traceback': additional_data.get('traceback')
        }
        
        # Include any additional data in the result
        result.update({k: v for k, v in additional_data.items() if k != 'traceback'})
        
        # Format error message for display
        error_display = f"❌ {error}"
        
        # Log the error with traceback if available
        if hasattr(self, 'log_error'):
            if 'traceback' in additional_data:
                # Format error message with traceback for logging
                error_msg = f"Operasi gagal: {error}\n\nTraceback (paling baru terakhir):\n{additional_data['traceback']}"
                self.log_error(error_msg)
            else:
                self.log_error(f"Operasi gagal: {error}")
        
        # Also log to the standard logger
        if hasattr(self, 'logger'):
            if 'traceback' in additional_data:
                self.log(f"Operasi gagal: {error}\n{additional_data['traceback']}", 'error')
            else:
                self.log(f"Operasi gagal: {error}", 'error')
                
        # Update progress tracker error state if available
        if hasattr(self, 'operation_container') and self.operation_container:
            try:
                # Get progress tracker from container
                progress_tracker = self.operation_container.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'error'):
                    # Call error method if available (newer versions)
                    progress_tracker.error(error_display)
                elif hasattr(progress_tracker, 'set_all_error'):
                    # Fallback to set_all_error if available
                    progress_tracker.set_all_error(error_display)
                # Also update progress to show error state
                if hasattr(self, 'update_progress_safe'):
                    self.update_progress_safe(self.operation_container.get('update_progress'), 
                                          100, error_display)
            except Exception as e:
                self.log(f"Failed to update progress tracker error state: {e}", 'warning')
        
        return result
    
    def test_write_access(self, directory_path: str, test_filename: str = '.smartcash_write_test') -> bool:
        """Test write access to a directory.
        
        Args:
            directory_path: Path to directory to test
            test_filename: Name of test file to create
            
        Returns:
            True if write access available, False otherwise
        """
        try:
            test_file = os.path.join(directory_path, test_filename)
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            self.log(f"Write access test failed for {directory_path}: {e}", 'warning')
            return False
    
    def ensure_directory_exists(self, directory_path: str) -> bool:
        """Ensure directory exists, create if necessary.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            True if directory exists or was created, False otherwise
        """
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
                self.log(f"📁 Created directory: {directory_path}", 'info')
            return True
        except Exception as e:
            self.log(f"❌ Failed to create directory {directory_path}: {e}", 'error')
            return False
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration from config.
        
        Returns:
            Environment configuration dictionary
        """
        return self.config.get('environment', {})
    
    def is_colab_environment(self) -> bool:
        """Check if running in Colab environment.
        
        Returns:
            True if in Colab, False otherwise
        """
        env_config = self.get_environment_config()
        return env_config.get('type') == 'colab'
    
    def validate_items_exist(self, items: list, item_type: str = "item") -> Dict[str, Any]:
        """Validate that items exist and return summary.
        
        Args:
            items: List of items (file paths, directories, etc.) to check
            item_type: Type of items being checked (for logging)
            
        Returns:
            Dictionary with validation results
        """
        existing_items = []
        missing_items = []
        
        for item in items:
            if os.path.exists(item):
                existing_items.append(item)
            else:
                missing_items.append(item)
        
        all_exist = len(missing_items) == 0
        
        self.log(f"✅ {len(existing_items)}/{len(items)} {item_type}s exist", 'info')
        
        return {
            'all_exist': all_exist,
            'existing_items': existing_items,
            'missing_items': missing_items,
            'existing_count': len(existing_items),
            'missing_count': len(missing_items),
            'total_count': len(items)
        }
    
    def detect_environment_enhanced(self, check_drive: bool = False) -> Dict[str, Any]:
        """Enhanced environment detection with caching.
        
        Args:
            check_drive: Whether to check drive mount status
            
        Returns:
            Environment information dictionary
        """
        try:
            env_info = detect_environment_info(check_drive=check_drive)
            self.log(f"Environment detected: {env_info.get('runtime', {}).get('type', 'unknown')}", 'debug')
            return env_info
        except Exception as e:
            self.log(f"Environment detection failed: {e}", 'error')
            return {
                'runtime': {'type': 'unknown'},
                'error': str(e)
            }
    
    def format_system_info(self, env_info: Dict[str, Any]) -> Dict[str, Any]:
        """Format system information from environment detection.
        
        Args:
            env_info: Environment information from detect_environment_info
            
        Returns:
            Formatted system information
        """
        os_info = env_info.get('os', {})
        ram_bytes = env_info.get('total_ram', 0)
        
        return {
            'os': os_info.get('system', 'Unknown'),
            'release': os_info.get('release', 'Unknown'),
            'machine': os_info.get('machine', 'Unknown'),
            'os_display': f"{os_info.get('system', 'Unknown')} {os_info.get('release', '')}".strip(),
            'ram_gb': ram_bytes / (1024**3) if ram_bytes > 0 else 0,
            'cpu_cores': env_info.get('cpu_cores', 'Unknown'),
            'gpu_available': env_info.get('gpu', 'No GPU available') != 'No GPU available',
            'gpu_name': env_info.get('gpu', 'None'),
            'is_colab': env_info.get('is_colab', False),
            'drive_mounted': env_info.get('drive_mounted', False),
            'drive_mount_path': env_info.get('drive_mount_path', '')
        }
    
    def validate_colab_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Colab environment configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation results
        """
        validation = {'valid': True, 'issues': []}
        
        try:
            # Validate environment config
            env_config = config.get('environment', {})
            if not env_config:
                validation['issues'].append('Missing environment configuration')
                validation['valid'] = False
                
            env_type = env_config.get('type', 'local')
            
            # Validate Colab-specific requirements
            if env_type == 'colab':
                try:
                    import google.colab  # noqa: F401
                except ImportError:
                    validation['issues'].append('Configuration set to Colab but not running in Colab environment')
                    validation['valid'] = False
            
            return validation
            
        except Exception as e:
            validation['issues'].append(f'Validation error: {str(e)}')
            validation['valid'] = False
            return validation
    
    def create_directories_batch(self, directories: List[str]) -> Tuple[List[str], List[str]]:
        """Create multiple directories in batch.
        
        Args:
            directories: List of directory paths to create
            
        Returns:
            Tuple of (created_dirs, failed_dirs)
        """
        created_dirs = []
        failed_dirs = []
        
        for directory in directories:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    created_dirs.append(directory)
                    self.log(f"📁 Created directory: {directory}", 'info')
                else:
                    created_dirs.append(directory)  # Already exists
                    
            except Exception as e:
                failed_dirs.append(directory)
                self.log(f"❌ Failed to create directory {directory}: {e}", 'error')
        
        return created_dirs, failed_dirs
    
    def verify_symlinks_batch(self, symlink_map: Dict[str, str]) -> Dict[str, Any]:
        """Verify multiple symlinks in batch.
        
        Args:
            symlink_map: Dictionary mapping source to target paths
            
        Returns:
            Dictionary with verification results
        """
        symlink_status = {}
        issues = []
        valid_count = 0
        
        for source, target in symlink_map.items():
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
        
        self.log(f"✅ {valid_count}/{len(symlink_map)} symlinks verified", 'info')
        
        return {
            'symlink_status': symlink_status,
            'valid_count': valid_count,
            'total_count': len(symlink_map),
            'all_valid': valid_count == len(symlink_map),
            'issues': issues
        }
    
    def verify_environment_variables(self, required_vars: List[str]) -> Dict[str, Any]:
        """Verify environment variables exist and are valid.
        
        Args:
            required_vars: List of required environment variable names
            
        Returns:
            Dictionary with verification results
        """
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
        
        self.log("✅ Environment variables verified", 'info')
        
        return {
            'env_vars_status': env_vars_status,
            'valid_count': valid_count,
            'total_count': len(required_vars),
            'all_valid': valid_count == len(required_vars),
            'python_path': os.environ.get('PYTHONPATH', ''),
            'issues': issues
        }
    
    def test_drive_access(self, drive_path: str) -> Dict[str, Any]:
        """Test Google Drive access and write permissions.
        
        Args:
            drive_path: Path to Google Drive mount
            
        Returns:
            Dictionary with access test results
        """
        if not drive_path or not os.path.exists(drive_path):
            return {
                'accessible': False,
                'write_access': False,
                'error': f"Drive path does not exist: {drive_path}"
            }
        
        # Test write access
        write_access = False
        try:
            test_file = os.path.join(drive_path, '.smartcash_drive_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            write_access = True
            self.log(f"✅ Drive write access verified: {drive_path}", 'info')
        except Exception as e:
            self.log(f"⚠️ Drive write access limited: {e}", 'warning')
        
        return {
            'accessible': os.path.exists(drive_path),
            'write_access': write_access,
            'path': drive_path
        }
    
    def create_verification_summary(self, component_results: Dict[str, Dict[str, Any]], 
                                  issues: List[str]) -> Dict[str, Any]:
        """Create a comprehensive verification summary.
        
        Args:
            component_results: Results from different verification components
            issues: List of all issues found
            
        Returns:
            Verification summary dictionary
        """
        summary = {
            'overall_status': 'PASS' if len(issues) == 0 else 'FAIL',
            'total_issues': len(issues),
            'components': {}
        }
        
        # Process each component's results
        for component_name, results in component_results.items():
            if component_name == 'folders':
                summary['components'][component_name] = {
                    'status': 'PASS' if results.get('all_exist', False) else 'FAIL',
                    'details': f"{results.get('existing_count', 0)}/{results.get('total_count', 0)} folders exist"
                }
            elif component_name == 'symlinks':
                summary['components'][component_name] = {
                    'status': 'PASS' if results.get('all_valid', False) else 'FAIL',
                    'details': f"{results.get('valid_count', 0)}/{results.get('total_count', 0)} symlinks valid"
                }
            elif component_name == 'environment_variables':
                summary['components'][component_name] = {
                    'status': 'PASS' if results.get('all_valid', False) else 'FAIL',
                    'details': f"{results.get('valid_count', 0)}/{results.get('total_count', 0)} variables valid"
                }
            elif component_name == 'drive_mount':
                summary['components'][component_name] = {
                    'status': 'PASS' if results.get('accessible', False) else 'FAIL',
                    'details': 'Mounted and accessible' if results.get('accessible', False) else 'Not mounted or not accessible'
                }
        
        return summary
    
    def get_progress_steps(self, operation_type: str) -> List[Dict[str, Any]]:
        """Get standardized progress steps for different operations.
        
        Args:
            operation_type: Type of operation (init, verify, mount, etc.)
            
        Returns:
            List of progress step dictionaries with phase progress information
        """
        progress_steps = {
            'init': [
                {'progress': 10, 'message': '🔍 [1/5] Detecting runtime environment...', 'phase_progress': 25},
                {'progress': 30, 'message': '✅ [2/5] Environment detected', 'phase_progress': 50},
                {'progress': 50, 'message': '🔧 [3/5] Checking system requirements...', 'phase_progress': 75},
                {'progress': 80, 'message': '🔍 [4/5] Validating configuration...', 'phase_progress': 100},
                {'progress': 100, 'message': '✅ [5/5] Initialization complete', 'phase_progress': 100}
            ],
            'verify': [
                {'progress': 10, 'message': '🔍 [1/6] Starting verification...', 'phase_progress': 10},
                {'progress': 30, 'message': '🔗 [2/6] Verifying symlinks...', 'phase_progress': 30},
                {'progress': 50, 'message': '📁 [3/6] Verifying folders...', 'phase_progress': 50},
                {'progress': 70, 'message': '🌍 [4/6] Verifying environment variables...', 'phase_progress': 70},
                {'progress': 90, 'message': '💻 [5/6] Gathering system info...', 'phase_progress': 90},
                {'progress': 100, 'message': '✅ [6/6] Verification complete', 'phase_progress': 100}
            ],
            'mount': [
                {'progress': 10, 'message': '🔍 [1/5] Checking environment status...', 'phase_progress': 20},
                {'progress': 30, 'message': '🔍 [2/5] Checking Drive mount status...', 'phase_progress': 40},
                {'progress': 50, 'message': '📁 [3/5] Mounting Google Drive...', 'phase_progress': 60},
                {'progress': 90, 'message': '🔍 [4/5] Verifying mount...', 'phase_progress': 80},
                {'progress': 100, 'message': '✅ [5/5] Google Drive mounted successfully', 'phase_progress': 100}
            ],
            'folders': [
                {'progress': 10, 'message': '🔍 Checking folder configuration...', 'phase_progress': 25},
                {'progress': 30, 'message': '📁 Creating required folders...', 'phase_progress': 50},
                {'progress': 70, 'message': '✅ Verifying folder structure...', 'phase_progress': 75},
                {'progress': 100, 'message': '✅ Folders ready', 'phase_progress': 100}
            ],
            'env_setup': [
                {'progress': 10, 'message': '🔍 Checking environment configuration...', 'phase_progress': 25},
                {'progress': 30, 'message': '🌍 Setting up environment variables...', 'phase_progress': 50},
                {'progress': 70, 'message': '🔍 Verifying environment setup...', 'phase_progress': 75},
                {'progress': 100, 'message': '✅ Environment ready', 'phase_progress': 100}
            ],
            'symlink': [
                {'progress': 10, 'message': '🔍 Checking symlink configuration...', 'phase_progress': 25},
                {'progress': 30, 'message': '🔗 Creating symlinks and backing up existing folders...', 'phase_progress': 50},
                {'progress': 70, 'message': '✅ Verifying symlinks...', 'phase_progress': 75},
                {'progress': 100, 'message': '✅ Symlinks ready', 'phase_progress': 100}
            ],
            'config_sync': [
                {'progress': 10, 'message': '🔍 Checking configuration...', 'phase_progress': 25},
                {'progress': 30, 'message': '⚙️ Syncing configuration...', 'phase_progress': 50},
                {'progress': 70, 'message': '✅ Validating sync...', 'phase_progress': 75},
                {'progress': 100, 'message': '✅ Configuration synchronized', 'phase_progress': 100}
            ]
        }
        
        return progress_steps.get(operation_type, [
            {'progress': 50, 'message': '🔄 Processing...'},
            {'progress': 100, 'message': '✅ Operation complete'}
        ])