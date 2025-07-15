"""
File: smartcash/ui/setup/dependency/operations/operation_manager.py
Deskripsi: Manages package operations with proper state and error handling.
"""
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from enum import Enum, auto

from .base_operation import BaseOperationHandler
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from .install_operation import InstallOperationHandler
from .update_operation import UpdateOperationHandler
from .uninstall_operation import UninstallOperationHandler
from .check_operation import CheckStatusOperationHandler


class OperationType(Enum):
    """Supported operation types."""
    INSTALL = auto()
    UPDATE = auto()
    UNINSTALL = auto()
    CHECK_STATUS = auto()


@dataclass
class OperationContext:
    """Context for an operation execution."""
    operation_type: OperationType
    packages: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    status_callback: Optional[Callable[[str, str], None]] = None
    progress_callback: Optional[Callable[[int, int], None]] = None


class OperationManager:
    """Manages package operations with proper state and error handling."""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        """Initialize the operation manager.
        
        Args:
            ui_components: Dictionary of UI components for operation feedback.
        """
        self.ui_components = ui_components or {}
        self._current_operation: Optional[OperationContext] = None
        self._operation_handlers: Dict[OperationType, Type[BaseOperationHandler]] = {}
        self._operation_in_progress = False  # Simple flag for operation tracking
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup operation handlers."""
        self._operation_handlers = {
            OperationType.INSTALL: InstallOperationHandler,
            OperationType.UPDATE: UpdateOperationHandler,
            OperationType.UNINSTALL: UninstallOperationHandler,
            OperationType.CHECK_STATUS: CheckStatusOperationHandler
        }
    
    def create_operation_context(
        self,
        operation_type: OperationType,
        packages: List[str],
        status_callback: Optional[Callable[[str, str], None]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> OperationContext:
        """Create a new operation context.
        
        Args:
            operation_type: Type of operation to perform.
            packages: List of packages to operate on.
            status_callback: Optional callback for status updates.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            OperationContext: The created operation context.
        """
        requires_confirmation = operation_type in [OperationType.UNINSTALL, OperationType.UPDATE]
        return OperationContext(
            operation_type=operation_type,
            packages=packages,
            requires_confirmation=requires_confirmation,
            status_callback=status_callback,
            progress_callback=progress_callback
        )
    
    def execute_operation(self, context: OperationContext) -> Dict[str, Any]:
        """Execute an operation with the given context.
        
        Args:
            context: The operation context.
            
        Returns:
            Dict[str, Any]: The operation result.
            
        Raises:
            RuntimeError: If another operation is already in progress.
        """
        # Set current operation
        self._current_operation = context
        return self._execute_operation(context)
        
    def _execute_operation(self, context: OperationContext) -> Dict[str, Any]:
        """Internal method to execute an operation.
        
        Args:
            context: The operation context.
            
        Returns:
            Dict[str, Any]: The operation result.
            
        Raises:
            ValueError: If no packages are provided for the operation.
        """
        # Validate packages
        if not context.packages:
            return {
                'success': False,
                'error': 'No packages provided',
                'message': 'No packages specified for operation'
            }
            
        try:
            # Get the appropriate handler class
            handler_class = self._operation_handlers.get(context.operation_type)
            if not handler_class:
                raise ValueError(f"No handler found for operation: {context.operation_type}")
            
            # Create handler instance with UI components and config
            handler = handler_class(
                ui_components=self.ui_components,
                config={}  # Pass any required config here
            )
            
            # Update status
            self._update_status(f"🚀 Starting {context.operation_type.name.lower()}...")
            
            # Execute the operation synchronously
            result = handler.execute_operation(context)
            
            # Update status based on result
            if result.get('success', False):
                self._update_status(f"✅ {context.operation_type.name.capitalize()} completed", "success")
            else:
                error_msg = result.get('error', 'Unknown error')
                self._update_status(f"❌ {context.operation_type.name.capitalize()} failed: {error_msg}", "error")
                
            return result
            
        except Exception as e:
            self._update_status(f"❌ {context.operation_type.name.capitalize()} failed: {str(e)}", "error")
            return {
                'success': False,
                'error': str(e),
                'message': f"Operation failed: {str(e)}"
            }
        finally:
            self._current_operation = None
    
    def _update_status(self, message: str, level: str = "info") -> None:
        """Update operation status.
        
        Args:
            message: Status message.
            level: Message level (info, warning, error, success).
        """
        if self._current_operation and self._current_operation.status_callback:
            self._current_operation.status_callback(message, level)
        elif 'status_label' in self.ui_components:
            self.ui_components['status_label'].value = message
            
    def get_current_operation(self) -> Optional[OperationContext]:
        """Get the current operation context.
        
        Returns:
            Optional[OperationContext]: The current operation context, or None if no operation is in progress.
        """
        return self._current_operation


class DependencyOperationManager(OperationHandler):
    """Operation manager for dependency management that extends OperationHandler."""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None, environment_manager=None, **kwargs):
        """Initialize the dependency operation manager.
        
        Args:
            config: Configuration dictionary
            ui_components: Dictionary of UI components
            environment_manager: Environment manager instance for environment detection
            **kwargs: Additional keyword arguments
        """
        # Extract operation container from ui_components if not provided
        operation_container = kwargs.pop('operation_container', None)
        if operation_container is None and ui_components is not None:
            if 'operation_container' in ui_components:
                operation_container = ui_components['operation_container']
            elif 'containers' in ui_components and 'operation' in ui_components['containers']:
                operation_container = ui_components['containers']['operation']
        
        # Remove logger from kwargs to prevent passing it to OperationHandler.__init__
        kwargs.pop('logger', None)
        
        super().__init__(
            module_name='dependency_operation_manager',
            parent_module='dependency',
            operation_container=operation_container,
            **kwargs
        )
        
        self.config = config
        self.ui_components = ui_components or {}
        self.environment_manager = environment_manager
        self.operation_manager = OperationManager(ui_components=self.ui_components)
        
        # Ensure operation_container is accessible
        if not hasattr(self, 'operation_container') and operation_container:
            self.operation_container = operation_container
    
    def initialize(self):
        """Initialize the dependency operation manager."""
        # Logger is already set up by parent OperationHandler class
        self.log("🔧 Initializing dependency operation manager", 'info')
        
        # Set up operation container if available
        if hasattr(self, 'operation_container') and self.operation_container:
            if hasattr(self.operation_container, 'clear_output'):
                self.operation_container.clear_output()
            elif hasattr(self.operation_container, 'clear_outputs'):
                self.operation_container.clear_outputs()
            
        # Operation handlers are initialized in constructor (no separate initialize method needed)
        
        # Set up UI event handlers if UI components are available
        self._setup_ui_handlers()
        
        self.log("✅ Dependency operation manager initialized", 'info')
        
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'install': self.execute_install,
            'uninstall': self.execute_uninstall, 
            'update': self.execute_update,
            'check_status': self.execute_check_status,
            'install_requirements': self.install_requirements_txt,
            'install_smartcash_yolo_requirements': self.install_smartcash_and_yolo_requirements
        }
    
    def execute_install(self, packages: List[str] = None, progress_callback=None) -> Dict[str, Any]:
        """Execute package installation using real pip install."""
        try:
            from .install_operation import InstallOperationHandler
            
            # Log packages being processed
            if packages:
                self.log(f"Installing specific packages: {packages}", 'info')
            else:
                self.log("Installing selected packages from UI", 'info')
            
            # Create install handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
            
            # Pass progress callback to UI components if provided
            if progress_callback:
                ui_components['progress_callback'] = progress_callback
                
            # Pass packages to config if provided
            if packages:
                self.config['explicit_packages'] = packages
            
            install_handler = InstallOperationHandler(ui_components, self.config)
            
            # Execute the actual installation
            result = install_handler.execute_operation()
            
            # Update operation summary with results
            self._update_operation_summary('install', result)
            
            return result
            
        except Exception as e:
            self.log(f"Error in execute_install: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def execute_uninstall(self, packages: List[str] = None, progress_callback=None) -> Dict[str, Any]:
        """Execute package uninstallation using real pip uninstall."""
        try:
            from .uninstall_operation import UninstallOperationHandler
            
            # Log packages being processed
            if packages:
                self.log(f"Uninstalling specific packages: {packages}", 'info')
            else:
                self.log("Uninstalling selected packages from UI", 'info')
            
            # Create uninstall handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
            
            # Pass progress callback to UI components if provided
            if progress_callback:
                ui_components['progress_callback'] = progress_callback
                
            # Pass packages to config if provided
            if packages:
                self.config['explicit_packages'] = packages
            
            uninstall_handler = UninstallOperationHandler(ui_components, self.config)
            
            # Execute the actual uninstallation
            result = uninstall_handler.execute_operation()
            
            # Update operation summary with results
            self._update_operation_summary('uninstall', result)
            
            return result
            
        except Exception as e:
            self.log(f"Error in execute_uninstall: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def execute_update(self, packages: List[str] = None, progress_callback=None) -> Dict[str, Any]:
        """Execute package update using real pip install --upgrade."""
        try:
            from .update_operation import UpdateOperationHandler
            
            # Log packages being processed
            if packages:
                self.log(f"Updating specific packages: {packages}", 'info')
            else:
                self.log("Updating all installed packages", 'info')
            
            # Create update handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
            
            # Pass progress callback to UI components if provided
            if progress_callback:
                ui_components['progress_callback'] = progress_callback
                
            # Pass packages to config if provided
            if packages:
                self.config['explicit_packages'] = packages
            
            update_handler = UpdateOperationHandler(ui_components, self.config)
            
            # Execute the actual update
            result = update_handler.execute_operation()
            
            # Update operation summary with results
            self._update_operation_summary('update', result)
            
            return result
            
        except Exception as e:
            self.log(f"Error in execute_update: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def execute_check_status(self, packages: List[str] = None, progress_callback=None) -> Dict[str, Any]:
        """Execute package status check using real pip show."""
        try:
            from .check_operation import CheckStatusOperationHandler
            
            # Log packages being checked
            if packages:
                self.log(f"Checking status of specific packages: {packages}", 'info')
            else:
                self.log("Checking status of all configured packages", 'info')
            
            # Create check status handler with UI components including operation container
            ui_components = getattr(self, '_ui_components', {})
            if hasattr(self.operation_container, 'get_ui_components'):
                ui_components.update(self.operation_container.get_ui_components())
            
            # Ensure operation container is available for progress tracking and logging
            ui_components['operation_container'] = self.operation_container
            
            # Pass progress callback to UI components if provided
            if progress_callback:
                ui_components['progress_callback'] = progress_callback
                
            # Pass packages to config if provided
            if packages:
                self.config['explicit_packages'] = packages
            
            check_handler = CheckStatusOperationHandler(ui_components, self.config)
            
            # Execute the actual status check
            result = check_handler.execute_operation()
            
            # Update operation summary with results
            self._update_operation_summary('check_status', result)
            
            return result
            
        except Exception as e:
            self.log(f"Error in execute_check_status: {e}", 'error')
            return {'success': False, 'error': str(e)}
    
    def install_requirements_txt(self, repo_path: str, progress_callback=None) -> Dict[str, Any]:
        """Install requirements.txt from a repository path.
        
        Args:
            repo_path: Path to the repository containing requirements.txt
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing installation results
        """
        try:
            import os
            import subprocess
            
            # Check if requirements.txt exists
            requirements_path = os.path.join(repo_path, 'requirements.txt')
            if not os.path.exists(requirements_path):
                return {
                    'success': False,
                    'error': f'requirements.txt not found in {repo_path}',
                    'message': f'No requirements.txt found in {repo_path}'
                }
            
            self.log(f"🚀 Installing requirements.txt from {repo_path}", 'info')
            
            # Read requirements.txt to get package count for progress
            with open(requirements_path, 'r') as f:
                requirements_content = f.read()
            
            # Parse requirements to count packages
            requirements_lines = [line.strip() for line in requirements_content.split('\n') 
                                if line.strip() and not line.strip().startswith('#')]
            total_packages = len(requirements_lines)
            
            if total_packages == 0:
                return {
                    'success': True,
                    'message': 'No packages to install (empty requirements.txt)',
                    'installed': 0,
                    'total': 0
                }
            
            # Update progress - starting installation
            if progress_callback:
                progress_callback(0, f"Installing {total_packages} packages from requirements.txt")
            
            # Build pip install command
            cmd = ['pip', 'install', '-r', requirements_path]
            
            # Execute pip install command
            self.log(f"Executing: {' '.join(cmd)}", 'info')
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=repo_path,
                text=True
            )
            
            stdout_str = process.stdout or ''
            stderr_str = process.stderr or ''
            
            # Update progress - installation complete
            if progress_callback:
                progress_callback(100, "Requirements.txt installation complete")
            
            if process.returncode == 0:
                self.log(f"✅ Successfully installed requirements.txt from {repo_path}", 'info')
                return {
                    'success': True,
                    'message': f'Successfully installed {total_packages} packages from requirements.txt',
                    'installed': total_packages,
                    'total': total_packages,
                    'stdout': stdout_str,
                    'repo_path': repo_path
                }
            else:
                error_msg = stderr_str or stdout_str or 'Unknown error'
                self.log(f"❌ Failed to install requirements.txt from {repo_path}: {error_msg}", 'error')
                return {
                    'success': False,
                    'error': error_msg,
                    'message': f'Failed to install requirements.txt: {error_msg}',
                    'repo_path': repo_path
                }
                
        except Exception as e:
            error_msg = f"Error installing requirements.txt from {repo_path}: {str(e)}"
            self.log(error_msg, 'error')
            return {
                'success': False,
                'error': str(e),
                'message': error_msg,
                'repo_path': repo_path
            }
    
    def install_smartcash_and_yolo_requirements(self, progress_callback=None) -> Dict[str, Any]:
        """Install requirements.txt from both smartcash and yolov5 repositories.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing installation results
        """
        try:
            # Define repository paths
            smartcash_path = '/content/smartcash'
            yolov5_path = '/content/yolov5'
            
            results = {
                'smartcash': None,
                'yolov5': None,
                'overall_success': False,
                'installed_total': 0,
                'errors': []
            }
            
            # Install smartcash requirements
            if progress_callback:
                progress_callback(0, "Installing SmartCash requirements.txt")
            
            smartcash_result = self.install_requirements_txt(smartcash_path, progress_callback)
            results['smartcash'] = smartcash_result
            
            if smartcash_result['success']:
                results['installed_total'] += smartcash_result.get('installed', 0)
                self.log("✅ SmartCash requirements.txt installed successfully", 'info')
            else:
                results['errors'].append(f"SmartCash: {smartcash_result.get('error', 'Unknown error')}")
                self.log(f"⚠️ SmartCash requirements.txt failed: {smartcash_result.get('error')}", 'warning')
            
            # Install yolov5 requirements
            if progress_callback:
                progress_callback(50, "Installing YOLOv5 requirements.txt")
            
            yolov5_result = self.install_requirements_txt(yolov5_path, progress_callback)
            results['yolov5'] = yolov5_result
            
            if yolov5_result['success']:
                results['installed_total'] += yolov5_result.get('installed', 0)
                self.log("✅ YOLOv5 requirements.txt installed successfully", 'info')
            else:
                results['errors'].append(f"YOLOv5: {yolov5_result.get('error', 'Unknown error')}")
                self.log(f"⚠️ YOLOv5 requirements.txt failed: {yolov5_result.get('error')}", 'warning')
            
            # Update final progress
            if progress_callback:
                progress_callback(100, "Requirements installation complete")
            
            # Determine overall success
            results['overall_success'] = smartcash_result['success'] and yolov5_result['success']
            
            # Create summary message
            if results['overall_success']:
                message = f"✅ Successfully installed requirements from both repositories ({results['installed_total']} packages total)"
            elif smartcash_result['success'] or yolov5_result['success']:
                message = f"⚠️ Partial success: {results['installed_total']} packages installed, but some repositories failed"
            else:
                message = "❌ Failed to install requirements from both repositories"
            
            results['message'] = message
            
            # Update operation summary
            self._update_operation_summary('install_requirements', results)
            
            return results
            
        except Exception as e:
            error_msg = f"Error installing requirements from repositories: {str(e)}"
            self.log(error_msg, 'error')
            return {
                'overall_success': False,
                'error': str(e),
                'message': error_msg,
                'smartcash': None,
                'yolov5': None,
                'installed_total': 0,
                'errors': [str(e)]
            }

    def _update_operation_summary(self, operation_type: str, result: Dict[str, Any]):
        """Update operation summary with operation results.
        
        Args:
            operation_type: Type of operation ('install', 'uninstall', 'update')
            result: Dictionary containing operation results
        """
        try:
            # Find summary container in UI components
            summary_container = None
            if 'containers' in self.ui_components and 'summary' in self.ui_components['containers']:
                summary_container = self.ui_components['containers']['summary']
            
            # If no summary container is found, try to use the operation container
            if summary_container is None and hasattr(self, 'operation_container'):
                summary_container = self.operation_container
            
            if summary_container is None:
                self.log("No summary container available to update", 'info')
                return
            
            # Prepare summary content
            summary = []
            
            # Add operation-specific header
            if operation_type == 'install':
                summary.append("### 📦 Installation Complete")
            elif operation_type == 'uninstall':
                summary.append("### 🗑️ Uninstallation Complete")
            elif operation_type == 'update':
                summary.append("### 🔄 Update Complete")
            else:
                summary.append("### Operation Complete")
            
            # Add installed packages
            if 'installed' in result and result['installed']:
                summary.append("\n✅ Successfully installed packages:")
                # Handle both list and integer formats
                if isinstance(result['installed'], list):
                    for pkg in result['installed']:
                        summary.append(f"- {pkg}")
                else:
                    # If it's an integer, show count
                    summary.append(f"- {result['installed']} packages installed")
            
            # Add failed packages with errors
            if 'failed' in result and result['failed']:
                summary.append("\n❌ Failed to process packages:")
                for pkg, error in result['failed'].items():
                    summary.append(f"- {pkg}: {error}")
            
            # Add skipped packages
            if 'skipped' in result and result['skipped']:
                summary.append("\n⚠️ Skipped packages (already up to date):")
                # Handle both list and integer formats
                if isinstance(result['skipped'], list):
                    for pkg in result['skipped']:
                        summary.append(f"- {pkg}")
                else:
                    # If it's an integer, show count
                    summary.append(f"- {result['skipped']} packages skipped")
            
            # Convert summary to markdown
            summary_markdown = '\n'.join(summary)
            
            # Update the summary container based on its type
            if hasattr(summary_container, 'value'):
                # Handle widgets with value attribute (e.g., HTML, Textarea)
                summary_container.value = summary_markdown
            elif hasattr(summary_container, 'clear_output'):
                # Handle output widgets
                with summary_container:
                    summary_container.clear_output()
                    from IPython.display import display, Markdown
                    display(Markdown(summary_markdown))
            else:
                # Fallback to logging
                self.log("\n" + summary_markdown, 'info')
                
        except Exception as e:
            self.log(f"❌ Failed to update operation summary: {e}", 'error')
            if hasattr(self, 'logger'):
                import traceback
                self.log(f"Error details: {traceback.format_exc()}", 'error')
    
    def _setup_ui_handlers(self):
        """Set up UI event handlers for operation buttons."""
        try:
            if 'widgets' not in self.ui_components:
                return
                
            widgets = self.ui_components['widgets']
            
            # Set up install button handler
            if 'install_button' in widgets:
                def on_install_clicked(_):
                    self.execute_install(self._get_selected_packages())
                widgets['install_button'].on_click(on_install_clicked)
            
            # Set up update button handler
            if 'update_button' in widgets:
                def on_update_clicked(_):
                    self.execute_update(self._get_selected_packages())
                widgets['update_button'].on_click(on_update_clicked)
            
            # Set up uninstall button handler
            if 'uninstall_button' in widgets:
                def on_uninstall_clicked(_):
                    self.execute_uninstall(self._get_selected_packages())
                widgets['uninstall_button'].on_click(on_uninstall_clicked)
                
        except Exception as e:
            self.log(f"❌ Failed to set up UI handlers: {e}", 'error')
    
    def _get_selected_packages(self) -> List[str]:
        """Get the list of selected packages from the UI.
        
        Returns:
            List of selected package names
        """
        try:
            if 'widgets' not in self.ui_components:
                return []
                
            widgets = self.ui_components['widgets']
            selected_packages = []
            
            # Get selected packages from categories
            if 'package_categories' in widgets:
                widget = widgets['package_categories']
                if hasattr(widget, 'selected'):
                    selected_packages.extend(widget.selected)
                elif hasattr(widget, 'value'):
                    selected_packages.extend(widget.value)
            
            # Get custom packages from text area
            if 'custom_packages_input' in widgets:
                custom_packages = widgets['custom_packages_input'].value.strip()
                if custom_packages:
                    selected_packages.extend([pkg.strip() for pkg in custom_packages.split('\n') if pkg.strip()])
            
            return list(set(selected_packages))  # Remove duplicates
            
        except Exception as e:
            self.log(f"❌ Error getting selected packages: {e}", 'error')
            return []
