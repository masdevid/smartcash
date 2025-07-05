"""
Base operation handler for dependency management operations.
"""
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import subprocess
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.ui.core.shared.error_handler import get_error_handler
from smartcash.ui.core.handlers.operation_handler import (
    OperationHandler, 
    OperationStatus,
    OperationResult
)


class BaseOperationHandler(OperationHandler):
    """Base class for dependency management operations.
    
    This class extends OperationHandler to provide common functionality
    for dependency management operations like install, update, and uninstall.
    """
    
    def __init__(self, operation_type: str, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize base operation handler.
        
        Args:
            operation_type: Type of operation (install/update/uninstall/check_status)
            ui_components: Dictionary of UI components
            config: Configuration dictionary with operation settings
        """
        super().__init__(operation_type, 'dependency.setup')
        self.ui_components = ui_components
        self.config = config
        self.error_handler = get_error_handler('dependency')
        
        # Set default operation container if available in UI components
        if 'operation_container' in ui_components:
            self.operation_container = ui_components['operation_container']
    
    async def _get_packages_to_process(self) -> List[str]:
        """Get list of packages to process based on current selection.
        
        Returns:
            List of package names to process
        """
        from ..components.package_selector import get_selected_packages, get_custom_packages_text
        
        packages = []
        
        try:
            # Get selected packages from categories
            selected_packages = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: get_selected_packages(self.ui_components)
            )
            packages.extend(selected_packages)
            
            # Get custom packages
            custom_packages_text = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: get_custom_packages_text(self.ui_components)
            )
            
            if custom_packages_text:
                for line in custom_packages_text.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        packages.append(line)
            
            return list(set(packages))  # Remove duplicates
            
        except Exception as e:
            self.log(f"Error getting packages to process: {str(e)}", 'error')
            return []
    
    async def _execute_command(
        self, 
        command: List[str], 
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 300,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Execute a shell command with error handling and progress tracking.
        
        Args:
            command: List of command arguments
            cwd: Working directory for the command
            env: Environment variables to use
            timeout: Command timeout in seconds
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with command execution results
        """
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
                
                # Read any remaining output
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout_lines.append(remaining_stdout)
                if remaining_stderr:
                    stderr_lines.append(remaining_stderr)
                
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
        
        # Execute the command in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_command)
    
    async def _process_packages(
        self,
        packages: List[str],
        process_func: Callable[[str], Dict[str, Any]],
        progress_message: str = "Processing packages",
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """Process multiple packages in parallel with progress tracking.
        
        Args:
            packages: List of package names to process
            process_func: Function to call for each package
            progress_message: Base message for progress updates
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary with overall results
        """
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
                        
                    # Update progress
                    progress = (processed / total) * 100
                    self._update_progress(
                        message=f"{progress_message}: {pkg} ({processed}/{total})",
                        current=progress,
                        total=100,
                        level_name='secondary'
                    )
                    
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
