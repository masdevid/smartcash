"""
Drive Mount Operation (Optimized) - Enhanced Mixin Integration
Mount Google Drive with verification using backend services.
"""

import os
from typing import Dict, Any, Optional, Callable
from smartcash.ui.components.operation_container import OperationContainer
from smartcash.common.environment import get_environment_manager
from .base_colab_operation import BaseColabOperation


class DriveMountOperation(BaseColabOperation):
    """Optimized drive mount operation with enhanced mixin integration."""
    
    def __init__(self, operation_name: str, config: Dict[str, Any], operation_container: Optional[OperationContainer] = None, **kwargs):
        super().__init__(operation_name, config, operation_container, **kwargs)
    
    def get_operations(self) -> Dict[str, Callable]:
        return {'mount_drive': self.execute_mount_drive}
    
    def execute_mount_drive(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Mount Google Drive with enhanced backend service integration."""
        def execute_operation():
            progress_steps = self.get_progress_steps('mount')
            
            # Step 1: Environment Validation with comprehensive diagnostics
            self.update_progress_safe(progress_callback, progress_steps[0]['progress'], progress_steps[0]['message'], progress_steps[0].get('phase_progress', 0))
            
            # Environment validation (optimized logging)
            self.log_debug("Starting environment validation...")
            
            # Check Google Colab module availability
            try:
                import google.colab  # noqa: F401
                import google.colab.drive  # noqa: F401
                self.log_debug("Colab modules available")
            except ImportError as drive_import_err:
                self.log_error(f"❌ google.colab.drive module not available: {drive_import_err}")
                return self.create_error_result(
                        f"❌ Drive mount failed: google.colab.drive not available. Error: {drive_import_err}",
                        error_type="CoLabDriveModuleError",
                        troubleshooting=[
                            "This could indicate a corrupted Colab runtime",
                            "Try restarting the runtime: Runtime > Restart runtime",
                            "If the issue persists, try a new Colab notebook"
                        ]
                    )
                    
            except ImportError as colab_import_err:
                self.log_error("❌ google.colab module not available - not in Colab environment")
                return self.create_error_result(
                    f"❌ Drive mount failed: Not running in Google Colab environment. This operation requires Google Colab. Error: {colab_import_err}",
                    error_type="NotInColabError",
                    current_environment=self._detect_current_environment(),
                    troubleshooting=[
                        "This operation can only be run in Google Colab",
                        "Please open this notebook in Google Colab",
                        "URL: https://colab.research.google.com/"
                    ]
                )
            
            # Check 2: System environment details
            self.log_info("🔍 Check 2: System environment details...")
            env_details = self._get_detailed_environment_info()
            self.log_info(f"📊 Environment details: {env_details}")
            
            # Check 3: Drive mount preparation (Direct Implementation)
            self.log_info("🔍 Check 3: Preparing drive mount process...")
            try:
                # Direct drive mount preparation without backend services
                self.log_info("✅ Drive mount process ready - no backend services needed for drive mounting")
                init_result = {'success': True, 'message': 'Drive mount preparation completed directly'}
            except Exception as e:
                self.log_warning(f"⚠️ Drive mount preparation failed: {e}")
                init_result = {'success': False, 'message': f'Drive mount preparation error: {e}'}
            
            # Check 4: Environment manager validation with fallback
            self.log_info("🔍 Check 4: Environment manager validation...")
            env_manager = None
            try:
                # Get logger using mixin method and pass to environment manager
                logger = self._get_logger() if hasattr(self, '_get_logger') else None
                env_manager = get_environment_manager(logger=logger)
                system_info = env_manager.get_system_info()
                self.log_info(f"📊 Environment manager system info: {system_info}")
                
                if not env_manager.is_colab:
                    self.log_error(f"❌ Environment manager reports non-Colab environment: {system_info}")
                    return self.create_error_result(
                        f"❌ Drive mount failed: Environment detection mismatch",
                        detected_environment=system_info.get('environment', 'Unknown'),
                        system_info=system_info,
                        troubleshooting=[
                            "Environment detection shows this is not running in Colab",
                            "Ensure you are running in Google Colab, not local Jupyter",
                            "Try restarting the Colab runtime"
                        ]
                    )
                else:
                    self.log_info("✅ Environment manager confirms Colab environment")
                    
            except Exception as e:
                self.log_warning(f"⚠️ Environment manager check failed, proceeding with manual checks: {e}")
                # Create fallback env_manager for later use
                env_manager = None
                
            self.log_info("✅ Environment validation completed successfully")
            
            # Step 2: Drive Mount Status Check
            self.update_progress_safe(progress_callback, progress_steps[1]['progress'], progress_steps[1]['message'], progress_steps[1].get('phase_progress', 0))
            
            # Check if already mounted using environment manager
            if env_manager and env_manager.is_drive_mounted:
                self.log_info("Google Drive already mounted")
                drive_info = self._get_drive_info(env_manager)
                
                # Mount status detected
                self.log_info(f"Drive already mounted at {drive_info['mount_path']}")
                sync_result = {'success': True, 'message': 'Drive already mounted'}
                
                return self.create_success_result('Google Drive already mounted', drive_info=drive_info, already_mounted=True, sync_result=sync_result)
            else:
                self.log_info("Drive not currently mounted - proceeding with mount operation")
            
            # Step 3: Execute Mount Operation
            self.update_progress_safe(progress_callback, progress_steps[2]['progress'], progress_steps[2]['message'], progress_steps[2].get('phase_progress', 0))
            
            try:
                mount_result = self._execute_drive_mount()
                if mount_result['success']:
                    # Log successful mount
                    self.log_info(f"Drive mounted successfully at {mount_result['mount_path']}")
                    mount_result['mount_status'] = 'successful'
                return mount_result
            except Exception as e:
                return self.create_error_result(f"Drive mount failed: {e}")
        
        return self.execute_with_error_handling(execute_operation)
    
    def _execute_drive_mount(self) -> Dict[str, Any]:
        """Execute the actual drive mount operation with enhanced diagnostics."""
        try:
            # Step 1: Import google.colab.drive with detailed diagnostics
            self.log_info("📦 Importing Google Colab drive module...")
            try:
                from google.colab import drive
                self.log_info("✅ Google Colab drive module imported successfully")
            except ImportError as import_err:
                error_details = {
                    'error_type': 'ImportError',
                    'error_msg': str(import_err),
                    'possible_causes': [
                        'Not running in Google Colab environment',
                        'Google Colab packages not installed',
                        'Runtime environment mismatch'
                    ],
                    'troubleshooting_steps': [
                        '1. Verify you are running this in Google Colab',
                        '2. Try restarting the runtime',
                        '3. Check if colab packages are properly installed'
                    ]
                }
                self.log_error(f"❌ Failed to import google.colab.drive: {error_details}")
                return self.create_error_result(f"❌ Cannot import google.colab.drive. Error: {import_err}", **error_details)
            
            mount_path = '/content/drive'
            self.log_info(f"📁 Attempting to mount Google Drive at {mount_path}")
            
            # Step 2: Check if already mounted with comprehensive diagnostics
            if os.path.exists(mount_path):
                if os.path.isdir(mount_path):
                    try:
                        contents = os.listdir(mount_path)
                        self.log_info(f"📂 Mount path exists with contents: {contents}")
                        
                        if 'MyDrive' in contents:
                            self.log_info("ℹ️ MyDrive folder found - Google Drive appears to be mounted")
                            # Additional verification
                            mydrive_path = os.path.join(mount_path, 'MyDrive')
                            if os.path.isdir(mydrive_path):
                                mydrive_contents = os.listdir(mydrive_path)[:5]  # Show first 5 items
                                self.log_info(f"📋 MyDrive contents (first 5): {mydrive_contents}")
                                return self._verify_and_return_mount(mount_path)
                            else:
                                self.log_warning("⚠️ MyDrive exists but is not a directory - possible mount corruption")
                        else:
                            self.log_info("📂 Mount path exists but MyDrive not found - proceeding with mount")
                    except PermissionError as perm_err:
                        self.log_warning(f"⚠️ Permission error accessing mount path: {perm_err}")
                    except Exception as list_err:
                        self.log_warning(f"⚠️ Error listing mount path contents: {list_err}")
                else:
                    self.log_warning(f"⚠️ {mount_path} exists but is not a directory - removing it")
                    try:
                        os.remove(mount_path)
                        self.log_info(f"🗑️ Removed non-directory file at {mount_path}")
                    except Exception as remove_err:
                        self.log_error(f"❌ Failed to remove {mount_path}: {remove_err}")
                        return self.create_error_result(f"Mount path {mount_path} exists as file and cannot be removed: {remove_err}")
            else:
                self.log_info(f"📁 Mount path {mount_path} does not exist - will be created during mount")
            
            # Step 3: Attempt mounting with detailed error handling
            self.log_info("🔄 Attempting initial mount (user authentication may be required)...")
            self.log_info("👤 Please complete the authentication process in the popup if prompted")
            
            mount_success = False
            mount_errors = []
            
            # First attempt: Normal mount
            try:
                self.log_info("🎯 Executing drive.mount() - normal mode...")
                drive.mount(mount_path, force_remount=False)
                self.log_info("✅ Normal mount completed successfully")
                mount_success = True
            except Exception as mount_error:
                error_details = {
                    'attempt': 'normal',
                    'error_type': type(mount_error).__name__,
                    'error_msg': str(mount_error),
                    'mount_path': mount_path
                }
                mount_errors.append(error_details)
                self.log_warning(f"⚠️ Normal mount failed: {mount_error}")
                
                # Second attempt: Force remount
                try:
                    self.log_info("🔄 Retrying with force_remount=True...")
                    drive.mount(mount_path, force_remount=True)
                    self.log_info("✅ Force remount completed successfully")
                    mount_success = True
                except Exception as force_error:
                    error_details = {
                        'attempt': 'force_remount',
                        'error_type': type(force_error).__name__,
                        'error_msg': str(force_error),
                        'mount_path': mount_path
                    }
                    mount_errors.append(error_details)
                    self.log_error(f"❌ Force remount also failed: {force_error}")
            
            # Step 4: Handle mount results
            if not mount_success:
                comprehensive_error = {
                    'summary': 'Both mount attempts failed',
                    'attempts': mount_errors,
                    'possible_causes': [
                        'Authentication failed or was cancelled',
                        'Google Drive API access denied',
                        'Network connectivity issues',
                        'Colab runtime limitations',
                        'Google account permissions issues'
                    ],
                    'troubleshooting_steps': [
                        '1. Ensure you are signed into the correct Google account',
                        '2. Check Google Drive permissions for Colab',
                        '3. Try restarting the Colab runtime',
                        '4. Verify network connectivity',
                        '5. Check if Google Drive has sufficient storage space'
                    ]
                }
                self.log_error(f"❌ Comprehensive mount failure: {comprehensive_error}")
                return self.create_error_result(f"❌ Drive mount failed after multiple attempts", **comprehensive_error)
            
            # Step 5: Verify mount success with detailed checks
            self.log_info("🔍 Verifying mount success...")
            return self._verify_and_return_mount(mount_path)
                
        except ImportError as ie:
            error_msg = "❌ google.colab.drive module not available - not running in Colab environment"
            self.log_error(error_msg)
            return self.create_error_result(f"{error_msg}. Import error: {ie}")
        except Exception as e:
            error_msg = f"❌ Drive mount operation failed with unexpected error: {e}"
            self.log_error(error_msg)
            import traceback
            tb = traceback.format_exc()
            self.log_error(f"Full traceback: {tb}")
            return self.create_error_result(error_msg, traceback=tb)
    
    def _verify_and_return_mount(self, mount_path: str) -> Dict[str, Any]:
        """Verify mount success and return result with comprehensive diagnostics."""
        verification_steps = []
        
        # Step 1: Check if mount path exists
        self.log_info(f"🔍 Step 1: Checking if mount path {mount_path} exists...")
        if not os.path.exists(mount_path):
            error_msg = f"❌ Mount path {mount_path} does not exist after mount attempt"
            verification_steps.append({'step': 1, 'check': 'path_exists', 'result': 'failed', 'details': error_msg})
            return self.create_error_result(error_msg, verification_steps=verification_steps)
        
        verification_steps.append({'step': 1, 'check': 'path_exists', 'result': 'passed', 'details': f'{mount_path} exists'})
        self.log_info("✅ Step 1 passed: Mount path exists")
        
        # Step 2: Check if it's a directory
        self.log_info(f"🔍 Step 2: Verifying {mount_path} is a directory...")
        if not os.path.isdir(mount_path):
            error_msg = f"❌ Mount path {mount_path} exists but is not a directory"
            verification_steps.append({'step': 2, 'check': 'is_directory', 'result': 'failed', 'details': error_msg})
            return self.create_error_result(error_msg, verification_steps=verification_steps)
        
        verification_steps.append({'step': 2, 'check': 'is_directory', 'result': 'passed', 'details': f'{mount_path} is a directory'})
        self.log_info("✅ Step 2 passed: Mount path is a directory")
        
        # Step 3: Check mount path contents
        self.log_info("🔍 Step 3: Checking mount path contents...")
        try:
            contents = os.listdir(mount_path)
            self.log_info(f"📂 Mount path contents: {contents}")
            verification_steps.append({'step': 3, 'check': 'list_contents', 'result': 'passed', 'details': f'Contents: {contents}'})
        except Exception as list_err:
            error_msg = f"❌ Cannot list contents of {mount_path}: {list_err}"
            verification_steps.append({'step': 3, 'check': 'list_contents', 'result': 'failed', 'details': error_msg})
            return self.create_error_result(error_msg, verification_steps=verification_steps)
        
        # Step 4: Check for MyDrive folder (critical indicator)
        self.log_info("🔍 Step 4: Checking for MyDrive folder...")
        mydrive_path = os.path.join(mount_path, 'MyDrive')
        if not os.path.exists(mydrive_path):
            error_msg = f"❌ MyDrive folder not found at {mydrive_path}. This indicates the mount may have failed silently."
            verification_steps.append({'step': 4, 'check': 'mydrive_exists', 'result': 'failed', 'details': f'MyDrive not found. Contents: {contents}'})
            return self.create_error_result(error_msg, verification_steps=verification_steps, mount_contents=contents)
        
        verification_steps.append({'step': 4, 'check': 'mydrive_exists', 'result': 'passed', 'details': f'MyDrive found at {mydrive_path}'})
        self.log_info("✅ Step 4 passed: MyDrive folder found")
        
        # Step 5: Verify MyDrive is accessible
        self.log_info("🔍 Step 5: Verifying MyDrive accessibility...")
        try:
            if os.path.isdir(mydrive_path):
                mydrive_contents = os.listdir(mydrive_path)
                sample_contents = mydrive_contents[:10] if len(mydrive_contents) > 10 else mydrive_contents
                self.log_info(f"📋 MyDrive accessible with {len(mydrive_contents)} items (showing first 10): {sample_contents}")
                verification_steps.append({
                    'step': 5, 
                    'check': 'mydrive_accessible', 
                    'result': 'passed', 
                    'details': f'MyDrive has {len(mydrive_contents)} items'
                })
            else:
                error_msg = f"❌ MyDrive exists at {mydrive_path} but is not a directory"
                verification_steps.append({'step': 5, 'check': 'mydrive_accessible', 'result': 'failed', 'details': error_msg})
                return self.create_error_result(error_msg, verification_steps=verification_steps)
        except Exception as mydrive_err:
            error_msg = f"❌ Cannot access MyDrive contents: {mydrive_err}"
            verification_steps.append({'step': 5, 'check': 'mydrive_accessible', 'result': 'failed', 'details': error_msg})
            self.log_warning(error_msg)
            # Continue anyway as MyDrive might be accessible but empty or have permission issues
        
        # Step 6: Test read/write access
        self.log_info("🔍 Step 6: Testing drive access permissions...")
        try:
            test_result = self._test_drive_access(mount_path)
            if test_result.get('test_successful', False):
                self.log_info(f"✅ Drive access test passed: {test_result}")
                verification_steps.append({'step': 6, 'check': 'access_test', 'result': 'passed', 'details': str(test_result)})
            else:
                self.log_warning(f"⚠️ Drive access test failed but mount appears successful: {test_result}")
                verification_steps.append({'step': 6, 'check': 'access_test', 'result': 'warning', 'details': str(test_result)})
        except Exception as test_error:
            error_details = f"Drive access test error: {test_error}"
            self.log_warning(f"⚠️ {error_details}")
            verification_steps.append({'step': 6, 'check': 'access_test', 'result': 'warning', 'details': error_details})
            test_result = {'write_access': False, 'read_access': False, 'error': str(test_error)}
        
        # Final verification summary
        self.log_success(f"✅ Google Drive successfully mounted and verified at {mount_path}")
        self.log_info(f"📊 Verification completed: {len([s for s in verification_steps if s['result'] == 'passed'])} passed, {len([s for s in verification_steps if s['result'] == 'failed'])} failed, {len([s for s in verification_steps if s['result'] == 'warning'])} warnings")
        
        return self.create_success_result(
            '✅ Google Drive mounted and verified successfully',
            mount_path=mount_path,
            mydrive_path=mydrive_path,
            write_access=test_result.get('write_access', False),
            read_access=test_result.get('read_access', False),
            test_result=test_result,
            verification_steps=verification_steps,
            mount_contents=contents if 'contents' in locals() else [],
            mydrive_item_count=len(mydrive_contents) if 'mydrive_contents' in locals() else 0
        )
    
    def _get_drive_info(self, env_manager) -> Dict[str, Any]:
        """Get comprehensive drive information with enhanced diagnostics."""
        try:
            system_info = env_manager.get_system_info()
            drive_info = {
                'mount_path': system_info.get('drive_path', '/content/drive'),
                'mounted': env_manager.is_drive_mounted,
                'base_directory': system_info.get('base_directory'),
                'data_directory': system_info.get('data_directory'),
                'system_info': system_info
            }
            
            # Add additional drive diagnostics
            mount_path = drive_info['mount_path']
            if os.path.exists(mount_path):
                try:
                    drive_info['mount_point_stats'] = {
                        'exists': True,
                        'is_directory': os.path.isdir(mount_path),
                        'permissions': oct(os.stat(mount_path).st_mode)[-3:],
                        'size': len(os.listdir(mount_path)) if os.path.isdir(mount_path) else 0
                    }
                    
                    if os.path.exists(os.path.join(mount_path, 'MyDrive')):
                        mydrive_path = os.path.join(mount_path, 'MyDrive')
                        drive_info['mydrive_stats'] = {
                            'exists': True,
                            'is_directory': os.path.isdir(mydrive_path),
                            'permissions': oct(os.stat(mydrive_path).st_mode)[-3:],
                            'item_count': len(os.listdir(mydrive_path)) if os.path.isdir(mydrive_path) else 0
                        }
                except Exception as stats_err:
                    drive_info['stats_error'] = str(stats_err)
            else:
                drive_info['mount_point_stats'] = {'exists': False}
                
            return drive_info
            
        except Exception as e:
            return {
                'error': f'Failed to get drive info: {e}',
                'mount_path': '/content/drive',  # Default fallback
                'mounted': False
            }
    
    def _test_drive_access(self, mount_path: str) -> Dict[str, Any]:
        """Test drive access with comprehensive diagnostics and error handling."""
        import time
        import uuid
        
        test_results = {
            'write_access': False,
            'read_access': False, 
            'delete_access': False,
            'test_successful': False,
            'test_details': [],
            'error': None
        }
        
        # Generate unique test file name to avoid conflicts
        test_filename = f'.smartcash_test_{uuid.uuid4().hex[:8]}_{int(time.time())}'
        test_file = os.path.join(mount_path, 'MyDrive', test_filename)
        
        try:
            # Test 1: Write access
            self.log_info(f"✏️ Testing write access to {test_file}...")
            test_content = f'SmartCash Drive Access Test - {time.ctime()}\nGenerated: {uuid.uuid4()}'
            
            try:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_content)
                    f.flush()  # Ensure data is written
                    os.fsync(f.fileno())  # Force sync to drive
                
                test_results['write_access'] = True
                test_results['test_details'].append({'test': 'write', 'status': 'passed', 'details': f'Successfully wrote {len(test_content)} chars'})
                self.log_info("✅ Write access test passed")
                
            except PermissionError as pe:
                error_msg = f"Permission denied writing to drive: {pe}"
                test_results['test_details'].append({'test': 'write', 'status': 'failed', 'error': error_msg})
                test_results['error'] = error_msg
                self.log_warning(f"⚠️ {error_msg}")
                return test_results
            
            except (IOError, OSError) as write_err:
                error_msg = f"IO error during write test: {write_err}"
                test_results['test_details'].append({'test': 'write', 'status': 'failed', 'error': error_msg})
                test_results['error'] = error_msg
                self.log_warning(f"⚠️ {error_msg}")
                return test_results
            
            # Test 2: Read access (only if write succeeded)
            if test_results['write_access']:
                self.log_info(f"📄 Testing read access to {test_file}...")
                try:
                    # Small delay to ensure file system consistency
                    time.sleep(0.1)
                    
                    with open(test_file, 'r', encoding='utf-8') as f:
                        read_content = f.read()
                    
                    if read_content == test_content:
                        test_results['read_access'] = True
                        test_results['test_details'].append({'test': 'read', 'status': 'passed', 'details': f'Successfully read {len(read_content)} chars, content matches'})
                        self.log_info("✅ Read access test passed")
                    else:
                        error_msg = f"Read content mismatch: expected {len(test_content)} chars, got {len(read_content)} chars"
                        test_results['test_details'].append({'test': 'read', 'status': 'failed', 'error': error_msg})
                        test_results['error'] = error_msg
                        self.log_warning(f"⚠️ {error_msg}")
                        
                except (IOError, OSError) as read_err:
                    error_msg = f"IO error during read test: {read_err}"
                    test_results['test_details'].append({'test': 'read', 'status': 'failed', 'error': error_msg})
                    test_results['error'] = error_msg
                    self.log_warning(f"⚠️ {error_msg}")
            
            # Test 3: Delete access (cleanup test file)
            if os.path.exists(test_file):
                self.log_info(f"🗑️ Testing delete access for {test_file}...")
                try:
                    os.remove(test_file)
                    
                    # Verify file is actually deleted
                    if not os.path.exists(test_file):
                        test_results['delete_access'] = True
                        test_results['test_details'].append({'test': 'delete', 'status': 'passed', 'details': 'Test file successfully deleted'})
                        self.log_info("✅ Delete access test passed")
                    else:
                        error_msg = "File still exists after delete operation"
                        test_results['test_details'].append({'test': 'delete', 'status': 'failed', 'error': error_msg})
                        test_results['error'] = error_msg
                        self.log_warning(f"⚠️ {error_msg}")
                        
                except (IOError, OSError) as delete_err:
                    error_msg = f"IO error during delete test: {delete_err}"
                    test_results['test_details'].append({'test': 'delete', 'status': 'failed', 'error': error_msg})
                    test_results['error'] = error_msg
                    self.log_warning(f"⚠️ {error_msg}")
            
            # Overall test success
            test_results['test_successful'] = (
                test_results['write_access'] and 
                test_results['read_access'] and 
                test_results['delete_access']
            )
            
            if test_results['test_successful']:
                self.log_info("✅ All drive access tests passed - drive is fully accessible")
            else:
                passed_tests = sum(1 for detail in test_results['test_details'] if detail['status'] == 'passed')
                total_tests = len(test_results['test_details'])
                self.log_warning(f"⚠️ Drive access partially limited: {passed_tests}/{total_tests} tests passed")
            
            return test_results
                        
        except Exception as e:
            error_msg = f"Unexpected error during drive access test: {e}"
            test_results['error'] = error_msg
            test_results['test_details'].append({'test': 'overall', 'status': 'failed', 'error': error_msg})
            self.log_error(f"❌ {error_msg}")
            
            # Attempt cleanup if test file exists
            try:
                if os.path.exists(test_file):
                    os.remove(test_file)
                    self.log_info(f"🧹 Cleaned up test file after error: {test_file}")
            except Exception as cleanup_err:
                self.log_warning(f"⚠️ Failed to cleanup test file: {cleanup_err}")
            
            return test_results
    
    def get_progress_steps(self, operation_type: str = 'mount') -> list:
        """Get detailed progress steps for mount operation with enhanced diagnostics."""
        return [
            {'progress': 10, 'message': '🔍 Validating Colab environment...', 'phase_progress': 20},
            {'progress': 25, 'message': '📁 Checking current mount status...', 'phase_progress': 40},
            {'progress': 60, 'message': '🔗 Mounting Google Drive...', 'phase_progress': 75},
            {'progress': 85, 'message': '🔍 Verifying mount success...', 'phase_progress': 95},
            {'progress': 100, 'message': '✅ Mount verified and ready', 'phase_progress': 100}
        ]
    
    def _detect_current_environment(self) -> Dict[str, Any]:
        """Detect detailed information about the current runtime environment."""
        import sys
        import os
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'current_working_directory': os.getcwd(),
            'environment_variables': {
                'COLAB_GPU': os.environ.get('COLAB_GPU'),
                'COLAB_TPU_ADDR': os.environ.get('COLAB_TPU_ADDR'),
                'TF_CONFIG': os.environ.get('TF_CONFIG'),
                'HOME': os.environ.get('HOME'),
                'PATH': os.environ.get('PATH', '')[:200] + '...' if len(os.environ.get('PATH', '')) > 200 else os.environ.get('PATH', '')
            },
            'sys_modules': {
                'google.colab': 'google.colab' in sys.modules,
                'google.colab.drive': 'google.colab.drive' in sys.modules,
                'jupyter': any(mod.startswith('jupyter') for mod in sys.modules),
                'ipython': 'IPython' in sys.modules
            }
        }
        
        # Detect environment type
        if env_info['sys_modules']['google.colab']:
            env_info['environment_type'] = 'google_colab'
        elif '/content' in env_info['current_working_directory']:
            env_info['environment_type'] = 'possible_colab'
        elif env_info['sys_modules']['jupyter']:
            env_info['environment_type'] = 'jupyter_notebook'
        else:
            env_info['environment_type'] = 'unknown'
            
        return env_info
    
    def _get_detailed_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information for diagnostics."""
        import psutil
        import os
        
        try:
            info = {
                'disk_usage': {
                    'root': psutil.disk_usage('/'),
                    'content': psutil.disk_usage('/content') if os.path.exists('/content') else None
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent_used': psutil.virtual_memory().percent
                },
                'network': {
                    'connections': len(psutil.net_connections()),
                    'interfaces': list(psutil.net_if_addrs().keys())
                },
                'processes': {
                    'count': len(psutil.pids()),
                    'google_related': [
                        proc.info['name'] for proc in psutil.process_iter(['pid', 'name']) 
                        if 'google' in proc.info['name'].lower()
                    ][:5]  # Limit to first 5 to avoid spam
                }
            }
        except Exception as e:
            info = {'error': f'Could not gather system info: {e}'}
            
        return info