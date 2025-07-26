"""
Pure mixin-based operation handler for dependency operations.
Uses composition over inheritance with core mixins.
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
# ProgressTrackingMixin removed - use operation_container.update_progress() instead
from smartcash.ui.core.mixins.operation_mixin import OperationMixin


class BaseOperationHandler(LoggingMixin, OperationMixin):
    """
    Pure mixin-based operation handler for dependency operations.
    
    Uses composition over inheritance - no BaseHandler or OperationHandler inheritance chain.
    This follows the mixin pattern used throughout the UI module system.
    """
    
    def __init__(self, operation_type: str, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize operation handler with pure mixin pattern.
        
        Args:
            operation_type: Type of operation (install/update/uninstall/check_status)
            ui_components: Dictionary of UI components
            config: Configuration dictionary with operation settings
        """
        # Initialize mixins
        super().__init__()
        
        # Setup logger
        self.logger = get_module_logger(f"smartcash.ui.setup.dependency.operations.{operation_type}")
        
        # Core operation state
        self.operation_type = operation_type
        self.ui_components = ui_components
        self.config = config
        self._cancelled = False
        
        # Module identification for mixins
        self.module_name = f'dependency_{operation_type}'
        self.parent_module = 'dependency.setup'
        
        # Set UI components for mixins
        self._set_ui_components(ui_components)
        
        # Progress tracking for dual progress system
        self._overall_progress = 0
        self._secondary_progress = 0
        self._ui_components = ui_components
        
        # Set operation container for logging
        if 'operation_container' in ui_components:
            self._operation_container = ui_components['operation_container']
        
        self.logger.debug(f"✅ BaseOperationHandler initialized: {operation_type}")
        
    def _set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components for the operation handler.
        
        Args:
            ui_components: Dictionary of UI components
        """
        # Store UI components for use by mixins
        self.ui_components = ui_components
        
        # Set up operation container if available
        if 'operation_container' in ui_components:
            self.operation_container = ui_components['operation_container']
            
        # Set up progress tracking if available
        if 'progress_tracker' in ui_components:
            self.progress_tracker = ui_components['progress_tracker']
            
        # Set up log accordion if available
        if 'log_accordion' in ui_components:
            self.log_accordion = ui_components['log_accordion']
            
        # Set up primary button if available
        if 'primary_button' in ui_components:
            self.primary_button = ui_components['primary_button']
    
    def _get_config_path(self, filename: str = 'dependency_config.yaml') -> str:
        """Get environment-aware config file path with log suppression."""
        try:
            # Suppress config manager logs by temporarily disabling logging
            import logging
            config_logger = logging.getLogger('smartcash.common.config.manager')
            original_level = config_logger.level
            config_logger.setLevel(logging.CRITICAL)  # Suppress all logs below CRITICAL
            
            try:
                from smartcash.common.config.manager import get_environment_config_path
                return get_environment_config_path(filename)
            finally:
                # Restore original logging level
                config_logger.setLevel(original_level)
                
        except Exception:
            return os.path.join('./configs', filename)
    
    def _get_packages_to_process(self) -> List[str]:
        """Get list of packages to process based on current selection."""
        packages = []
        
        try:
            # First, check if explicit packages are provided in config
            if self.config and 'explicit_packages' in self.config:
                explicit_packages = self.config['explicit_packages']
                if isinstance(explicit_packages, list):
                    packages.extend(explicit_packages)
                    return list(set(packages))  # Remove duplicates and return early
            
            # Get selected packages from UI components
            if 'package_checkboxes' in self.ui_components:
                checkboxes = self.ui_components['package_checkboxes']
                for checkbox_list in checkboxes.values():
                    for checkbox in checkbox_list:
                        if hasattr(checkbox, 'value') and checkbox.value:
                            if hasattr(checkbox, 'package_name'):
                                packages.append(checkbox.package_name)
                            elif hasattr(checkbox, 'description'):
                                desc = checkbox.description
                                if '(' in desc:
                                    package_name = desc.split('(')[0].strip()
                                    packages.append(package_name)
            
            # Get custom packages
            if 'custom_packages' in self.ui_components:
                custom_widget = self.ui_components['custom_packages']
                if hasattr(custom_widget, 'value') and custom_widget.value.strip():
                    custom_packages = [pkg.strip() for pkg in custom_widget.value.split(',') if pkg.strip()]
                    packages.extend(custom_packages)
            
            return list(set(packages))  # Remove duplicates
            
        except Exception as e:
            self.log(f"Error getting packages to process: {str(e)}", 'error')
            return []
    
    def _execute_command(
        self, 
        command: List[str], 
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a shell command with error handling and progress tracking."""
        def run_command() -> Dict[str, Any]:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd or os.getcwd(),
                    env=env or os.environ,
                    bufsize=1,  # Line buffered
                    universal_newlines=True
                )
                
                stdout_lines = []
                stderr_lines = []
                
                # Read output in real-time
                while True:
                    # Check for process completion
                    if process.poll() is not None:
                        break
                        
                    # Read stdout
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        stdout_lines.append(stdout_line)
                        if progress_callback:
                            progress_callback(0, stdout_line.strip())
                    
                    # Read stderr
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        stderr_lines.append(stderr_line)
                        if progress_callback:
                            progress_callback(0, f"ERROR: {stderr_line.strip()}")
                
                # Read any remaining output with timeout
                try:
                    remaining_stdout, remaining_stderr = process.communicate(timeout=timeout)
                    if remaining_stdout:
                        stdout_lines.append(remaining_stdout)
                    if remaining_stderr:
                        stderr_lines.append(remaining_stderr)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        remaining_stdout, remaining_stderr = process.communicate(timeout=5)
                        if remaining_stdout:
                            stdout_lines.append(remaining_stdout)
                        if remaining_stderr:
                            stderr_lines.append(remaining_stderr)
                    except subprocess.TimeoutExpired:
                        pass
                    raise subprocess.TimeoutExpired(command, timeout)
                
                return {
                    'success': process.returncode == 0,
                    'returncode': process.returncode,
                    'stdout': '\n'.join(stdout_lines) if stdout_lines else '',
                    'stderr': '\n'.join(stderr_lines) if stderr_lines else ''
                }
                
            except subprocess.TimeoutExpired as e:
                return {
                    'success': False,
                    'error': f"Command timed out after {e.timeout} seconds",
                    'returncode': -1,
                    'stdout': e.stdout.decode('utf-8') if e.stdout else '',
                    'stderr': e.stderr.decode('utf-8') if e.stderr else ''
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'returncode': -1
                }
        
        return run_command()
    
    def _process_packages(
        self,
        packages: List[str],
        process_func: Callable[[str], Dict[str, Any]],
        progress_message: str = "Processing packages",
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """Process multiple packages in parallel with progress tracking."""
        results = {
            'success': True,
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'details': []
        }
        
        total = len(packages)
        if total == 0:
            return results
            
        processed = 0
        
        # Create a thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_func, pkg): pkg
                for pkg in packages
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                pkg = futures[future]
                processed += 1
                
                try:
                    result = future.result()
                    results['details'].append({
                        'package': pkg,
                        'success': result.get('success', False),
                        'message': result.get('message', '')
                    })
                    
                    if result.get('success'):
                        results['succeeded'] += 1
                    else:
                        results['failed'] += 1
                        results['success'] = False
                        
                    # Update progress using mixin method
                    progress = (processed / total) * 100
                    self.update_progress(progress, f"{progress_message}: {pkg} ({processed}/{total})")
                    
                except Exception as e:
                    results['failed'] += 1
                    results['success'] = False
                    results['details'].append({
                        'package': pkg,
                        'success': False,
                        'message': str(e)
                    })
                    
                    self.log(f"Error processing {pkg}: {str(e)}", 'error')
        
        results['processed'] = processed
        return results
    
    def _categorize_packages(self, packages: List[str]) -> Tuple[List[str], List[str]]:
        """Categorize packages into regular and custom (with version specifiers)."""
        selected: List[str] = []
        custom: List[str] = []
        for pkg in packages:
            if any(c in pkg for c in '><='):
                custom.append(pkg)
            else:
                selected.append(pkg)
        return selected, custom
    
    def _save_config_to_file(self, config: Dict[str, Any]) -> None:
        """Save configuration to dependency_config.yaml file."""
        try:
            import yaml
            import os
            
            config_path = self._get_config_path('dependency_config.yaml')
            
            # Read existing config
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = yaml.safe_load(f) or {}
            else:
                existing_config = {}
            
            # Remove successfully installed packages from uninstalled_defaults
            uninstalled_defaults = existing_config.get('uninstalled_defaults', [])
            selected_packages = config.get('selected_packages', [])
            custom_packages = config.get('custom_packages', '')
            
            # Parse installed packages from both selected and custom
            all_installed_packages = set(selected_packages)
            if custom_packages:
                for line in custom_packages.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name = line.split('>')[0].split('<')[0].split('=')[0].strip()
                        all_installed_packages.add(package_name)
            
            # Remove any installed packages from uninstalled_defaults
            updated_uninstalled_defaults = [pkg for pkg in uninstalled_defaults 
                                          if pkg not in all_installed_packages]
            
            # Update with new values
            existing_config.update({
                'selected_packages': selected_packages,
                'custom_packages': custom_packages,
                'uninstalled_defaults': updated_uninstalled_defaults
            })
            
            # Write back to file
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True)
            
            self.log(f"💾 Configuration saved to {config_path}", 'info')
            
        except Exception as e:
            self.log(f"❌ Failed to save config: {str(e)}", 'error')
    
    def cancel_operation(self) -> None:
        """Cancel the current operation."""
        self._cancelled = True
        self.log("Permintaan pembatalan diterima, menunggu proses saat ini selesai...", 'warning')
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler."""
        return {
            'execute': self.execute_operation,
            'cancel': self.cancel_operation
        }
    
    # ===== Consolidated Package Management Methods =====
    
    def _is_package_installed(self, package: str) -> bool:
        """
        Check if a package is installed using pip show.
        
        Args:
            package: Package name to check
            
        Returns:
            True if package is installed, False otherwise
        """
        try:
            # Extract package name without version specifier
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
            
            # Use pip show to check installation
            result = subprocess.run(
                ["pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def _get_package_info(self, package: str) -> Dict[str, Any]:
        """
        Get detailed information about an installed package.
        
        Args:
            package: Package name to get info for
            
        Returns:
            Dictionary with package information or error details
        """
        try:
            # Extract package name without version specifier
            package_name = package.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
            
            # Use pip show to get package information
            result = subprocess.run(
                ['pip', 'show', package_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse pip show output
                version_info = self._parse_pip_show_output(result.stdout)
                return {
                    'package': package_name,
                    'installed': True,
                    'version': version_info.get('version', 'Unknown'),
                    'location': version_info.get('location', 'Unknown'),
                    'summary': version_info.get('summary', ''),
                    'requires': version_info.get('requires', '')
                }
            else:
                return {
                    'package': package_name,
                    'installed': False,
                    'error': 'Package not found',
                    'message': 'Not installed'
                }
                
        except subprocess.TimeoutExpired:
            return {
                'package': package_name,
                'installed': False,
                'error': 'Timeout saat memeriksa paket',
                'message': 'Timeout'
            }
        except Exception as e:
            return {
                'package': package_name,
                'installed': False,
                'error': str(e),
                'message': 'Error checking package'
            }
    
    def _parse_pip_show_output(self, output: str) -> Dict[str, str]:
        """Parse output from pip show command."""
        info = {}
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip().lower()] = value.strip()
        return info
    
    def _build_pip_command(self, operation: str, package: str, **kwargs) -> List[str]:
        """
        Build a pip command based on operation type and configuration.
        
        Args:
            operation: Type of operation ('install', 'uninstall', 'upgrade')
            package: Package name
            **kwargs: Additional options
            
        Returns:
            List of command arguments
        """
        command = ["pip", operation]
        
        if operation == "install":
            command.append("--upgrade")
            if self.config.get('use_index_url'):
                command.extend(["-i", self.config['index_url']])
        elif operation == "uninstall":
            command.append("-y")  # Skip confirmation
            
        command.append(package)
        return command
    
    def update_progress(
        self, 
        progress: int, 
        message: str = "", 
        secondary_progress: Optional[int] = None, 
        secondary_message: str = ""
    ) -> None:
        """
        Update dual progress tracker with main and secondary progress.
        
        Args:
            progress: Main progress percentage (0-100)
            message: Main progress message
            secondary_progress: Secondary progress percentage (0-100), optional
            secondary_message: Secondary progress message
        """
        try:
            # Update internal progress tracking
            self._overall_progress = max(0, min(100, progress))
            if secondary_progress is not None:
                self._secondary_progress = max(0, min(100, secondary_progress))
            
            # Update operation container if available
            if hasattr(self, '_ui_components') and 'operation_container' in self._ui_components:
                operation_container = self._ui_components['operation_container']
                
                # Call operation container's update_progress method
                if hasattr(operation_container, 'get') and operation_container.get('update_progress'):
                    update_func = operation_container.get('update_progress')
                    if secondary_progress is not None:
                        # Dual progress mode
                        update_func(
                            progress=self._overall_progress,
                            message=message,
                            secondary_progress=self._secondary_progress,
                            secondary_message=secondary_message
                        )
                    else:
                        # Single progress mode
                        update_func(progress=self._overall_progress, message=message)
                        
        except Exception as e:
            # Fail silently on progress update errors to not break operations
            if hasattr(self, 'logger'):
                self.logger.debug(f"Progress update failed: {e}")
    
    def reset_progress(self) -> None:
        """Reset both progress trackers to 0."""
        self._overall_progress = 0
        self._secondary_progress = 0
        self.update_progress(0, "Initializing...")
    
    def complete_progress(self, message: str = "Operation completed") -> None:
        """Set progress to 100% and show completion message."""
        self.update_progress(100, message)
        
    def _run_pip_command(self, operation: str, package: str, **kwargs) -> Dict[str, Any]:
        """
        Run a pip command with error handling and progress updates.
        
        Args:
            operation: The pip operation to run ('install', 'uninstall', 'upgrade')
            package: The package name to operate on
            **kwargs: Additional arguments for the pip command
            
        Returns:
            Dictionary with command results
        """
        try:
            # Build the pip command
            command = self._build_pip_command(operation, package, **kwargs)
            
            # Log the command being run
            self.log(f"🔧 Running command: {' '.join(command)}", 'info')
            
            # Define progress callback
            def progress_callback(progress: float, message: str) -> None:
                self.update_progress(progress, message)
            
            # Execute the command
            result = self._execute_command(
                command=command,
                progress_callback=progress_callback,
                timeout=300  # 5 minute timeout
            )
            
            # Log the result
            if result['success']:
                self.log(f"✅ Successfully completed {operation} for {package}", 'success')
            else:
                error_msg = result.get('error', 'Unknown error')
                self.log(f"❌ Failed to {operation} {package}: {error_msg}", 'error')
                if 'stdout' in result and result['stdout']:
                    self.log(f"📝 Output: {result['stdout']}", 'info')
                if 'stderr' in result and result['stderr']:
                    self.log(f"❌ Error: {result['stderr']}", 'error')
            
            return result
            
        except Exception as e:
            error_msg = f"Error running pip {operation} for {package}: {str(e)}"
            self.log(error_msg, 'error')
            return {
                'success': False,
                'error': error_msg,
                'returncode': -1
            }