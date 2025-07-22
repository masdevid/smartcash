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
        self._ui_components = ui_components
        
        # Set operation container for logging
        if 'operation_container' in ui_components:
            self._operation_container = ui_components['operation_container']
        
        self.logger.debug(f"✅ BaseOperationHandler initialized: {operation_type}")
    
    def _get_config_path(self, filename: str = 'dependency_config.yaml') -> str:
        """Get environment-aware config file path."""
        try:
            from smartcash.common.config.manager import get_environment_config_path
            return get_environment_config_path(filename)
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