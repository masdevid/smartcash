"""
File: smartcash/ui/setup/dependency/dependency_uimodule.py
Description: Dependency Module implementation using UIModule pattern.

Simplified dependency management with core functionality:
- Install packages (core + custom)
- Uninstall packages 
- Check status & updates
- Update packages
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from smartcash.ui.core.ui_module import UIModule, SharedMethodRegistry, register_operation_method
from smartcash.ui.core.ui_module_factory import create_template
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.decorators import handle_ui_errors

# Import existing dependency components and handlers
from smartcash.ui.setup.dependency.configs.dependency_config_handler import DependencyConfigHandler
from smartcash.ui.setup.dependency.configs.dependency_defaults import get_default_dependency_config
from smartcash.ui.setup.dependency.operations.operation_manager import DependencyOperationManager

# Global module instance for singleton pattern
_dependency_uimodule: Optional[UIModule] = None

def register_dependency_template() -> None:
    """Register Dependency module template with UIModuleFactory."""
    from smartcash.ui.core.ui_module_factory import UIModuleFactory
    
    template = create_template(
        module_name="dependency",
        parent_module="setup",
        default_config=get_default_dependency_config(),
        required_components=["main_container", "form_container", "action_container", "operation_container"],
        required_operations=["install", "uninstall", "check_status", "update"],
        auto_initialize=False,
        description="Package dependency management module"
    )
    
    try:
        UIModuleFactory.register_template(template, overwrite=True)
        get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("📋 Registered Dependency template")
    except Exception as e:
        get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to register template: {e}")

def register_dependency_shared_methods() -> None:
    """Register Dependency-specific shared methods."""
    
    def get_package_status(package_name: str) -> Dict[str, Any]:
        """Get status of a specific package."""
        try:
            import subprocess
            result = subprocess.run(['pip', 'show', package_name], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return {"installed": True, "package": package_name, "details": result.stdout}
            else:
                return {"installed": False, "package": package_name}
        except Exception as e:
            return {"installed": False, "package": package_name, "error": str(e)}
    
    def list_installed_packages() -> Dict[str, Any]:
        """List all installed packages."""
        try:
            import subprocess
            result = subprocess.run(['pip', 'list', '--format=json'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                import json
                packages = json.loads(result.stdout)
                return {"success": True, "packages": packages}
            else:
                return {"success": False, "error": "Failed to list packages"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Register methods with error handling for re-registration
    methods = [
        ("get_package_status", get_package_status, "Get package installation status"),
        ("list_installed_packages", list_installed_packages, "List all installed packages")
    ]
    
    for name, method, desc in methods:
        try:
            register_operation_method(name, method, description=desc)
        except ValueError:
            SharedMethodRegistry.register_method(name, method, overwrite=True, 
                                               description=desc, category="operations")
    
    get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("🔗 Registered Dependency shared methods")

class DependencyUIModule(UIModule):
    """Dependency-specific UIModule with package management functionality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Dependency UIModule.
        
        Args:
            config: Dependency configuration (optional, uses defaults if not provided)
        """
        # Get default config and merge with provided config
        default_config = get_default_dependency_config()
        if config:
            default_config.update(config)
        
        super().__init__(
            module_name="dependency",
            parent_module="setup", 
            config=default_config,
            auto_initialize=False
        )
        
        # Dependency-specific attributes
        self._operation_manager: Optional[DependencyOperationManager] = None
        self._config_handler: Optional[DependencyConfigHandler] = None
        self._package_status = {}
        
    def initialize(self, config: Dict[str, Any] = None) -> 'DependencyUIModule':
        """Initialize Dependency module with package management.
        
        Args:
            config: Additional configuration to merge
            
        Returns:
            Self for method chaining
        """
        if config:
            self.update_config(**config)
        
        try:
            # Create UI components
            self._create_ui_components()
            
            # Setup operation manager
            self._setup_operation_manager()
            
            # Setup config handler
            self._setup_config_handler()
            
            # Register operations
            self._register_operations()
            
            # Inject shared methods
            SharedMethodRegistry.inject_methods(self, category="operations")
            
            # Setup event handlers for buttons
            self._setup_event_handlers()
            
            # Call parent initialization
            super().initialize()
            
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"✅ Initialized Dependency UIModule")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to initialize Dependency UIModule: {e}")
            raise
        
        return self
    
    def _create_ui_components(self) -> None:
        """Create and register UI components."""
        try:
            # Use the new component structure
            from smartcash.ui.setup.dependency.components import create_dependency_ui_components
            
            ui_components = create_dependency_ui_components(self.get_config())
            
            # Register each component
            for component_type, component in ui_components.items():
                self.register_component(component_type, component)
            
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"📦 Created {len(ui_components)} UI components")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to create UI components: {e}")
            raise
    
    
    def _setup_operation_manager(self) -> None:
        """Setup operation manager for Dependency operations."""
        try:
            # Get operation container instance directly
            operation_container = self.get_component("operation_container")
            
            self._operation_manager = DependencyOperationManager(
                config=self.get_config(),
                operation_container=operation_container,
                ui_components={'operation_container': operation_container}
            )
            
            # Setup UI logging bridge to capture backend service logs
            self._setup_ui_logging_bridge(operation_container)
            
            # Initialize progress tracker display
            self._initialize_progress_display()
            
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("⚙️ Setup operation manager")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to setup operation manager: {e}")
            raise
    
    
    def _setup_config_handler(self) -> None:
        """Setup config handler."""
        try:
            self._config_handler = DependencyConfigHandler()
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("🔧 Setup config handler")
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to setup config handler: {e}")
            raise
    
    def _register_operations(self) -> None:
        """Register Dependency operations."""
        try:
            if not self._operation_manager:
                raise ValueError("Operation manager not initialized")
            
            # Get operations from manager
            operations = self._operation_manager.get_operations()
            
            # Register each operation
            for op_name, op_func in operations.items():
                self.register_operation(op_name, op_func)
            
            # Register convenience methods
            self.register_operation("get_status", self.get_package_status_summary)
            self.register_operation("refresh_status", self.refresh_package_status)
            
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"⚙️ Registered {len(operations)} operations")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to register operations: {e}")
            raise
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            # Connect buttons to handlers (updated button names)
            add_button = self.get_component("add_button")
            if add_button:
                add_button.on_click(self._handle_add_click)
            
            install_button = self.get_component("install_button")
            if install_button:
                install_button.on_click(self._handle_install_click)
            
            check_button = self.get_component("check_button")
            if check_button:
                check_button.on_click(self._handle_check_click)
            
            update_button = self.get_component("update_button")
            if update_button:
                update_button.on_click(self._handle_update_click)
            
            uninstall_button = self.get_component("uninstall_button")
            if uninstall_button:
                uninstall_button.on_click(self._handle_uninstall_click)
            
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("✅ Connected dependency button handlers")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to setup event handlers: {e}")
    
    def _handle_add_click(self, _=None) -> None:
        """Handle add packages button click."""
        try:
            self._log_to_ui("➕ Adding packages to configuration...", "info")
            
            # Get selected packages
            selected_packages = self._get_selected_packages()
            
            # Get custom packages
            custom_packages = self.get_component("custom_packages")
            custom_packages_text = custom_packages.value if custom_packages else ""
            
            # Save to config handler
            if self._config_handler:
                # Update selected packages
                current_config = self._config_handler.config.copy()
                current_config['selected_packages'] = selected_packages
                current_config['custom_packages'] = custom_packages_text
                
                # Save configuration
                self._config_handler.config = current_config
                success = True
                
                if success:
                    self._log_to_ui(f"✅ Packages added to configuration! {len(selected_packages)} packages selected", "success")
                    
                    # Update UI display
                    self._update_ui_after_save()
                else:
                    self._log_to_ui("❌ Failed to add packages to configuration", "error")
            else:
                self._log_to_ui("❌ Config handler not available", "error")
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Add packages failed: {e}")
            self._log_to_ui(f"❌ Add packages error: {str(e)}", "error")
    
    def _update_ui_after_save(self) -> None:
        """Update UI components after saving configuration."""
        try:
            # Refresh package status to show current state
            self.refresh_package_status()
            
            # Update header status
            header_container = self.get_component("header_container")
            if header_container and hasattr(header_container, 'update_status'):
                selected_count = len(self._get_selected_packages())
                header_container.update_status(
                    f"Configuration saved - {selected_count} packages selected",
                    "success"
                )
            
            # Log configuration file location
            config_path = getattr(self._config_handler, 'config_file', "dependency_config.yaml") if self._config_handler else "dependency_config.yaml"
            self._log_to_ui(f"📁 Configuration saved to: {config_path}", "info")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to update UI after save: {e}")
    
    def _handle_install_click(self, _=None) -> None:
        """Handle install button click."""
        try:
            selected_packages = self._get_selected_packages()
            if not selected_packages:
                self._log_to_ui("⚠️ No packages selected for installation", "warning")
                return
            
            self._log_to_ui(f"📥 Installing {len(selected_packages)} packages...", "info")
            
            # Execute install operation with progress tracking
            result = self.execute_install_operation(selected_packages)
            
            if result.get("success", False):
                self._log_to_ui("✅ Installation completed successfully!", "success")
                self.refresh_package_status()
            else:
                error_msg = result.get("message", "Installation failed")
                self._log_to_ui(f"❌ Installation failed: {error_msg}", "error")
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Install click failed: {e}")
            self._log_to_ui(f"❌ Install error: {str(e)}", "error")
    
    def _handle_check_click(self, _=None) -> None:
        """Handle check status button click - automatically check for missing packages."""
        try:
            self._log_to_ui("🔍 Checking package status and finding missing packages...", "info")
            
            # Refresh package status first
            self.refresh_package_status()
            
            # Check for missing packages and suggest them
            self._check_missing_packages()
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Check click failed: {e}")
            self._log_to_ui(f"❌ Check error: {str(e)}", "error")
    
    def _check_missing_packages(self) -> None:
        """Check for missing packages and automatically select them."""
        try:
            missing_packages = []
            
            # Get all packages that should be installed (selected ones)
            selected_packages = self._get_selected_packages()
            
            # Check which ones are actually missing
            for package in selected_packages:
                # Extract package name (before version specifiers)
                pkg_name = package.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
                
                # Check if package is installed
                status = self.get_package_status(pkg_name)
                if not status.get("installed", False):
                    missing_packages.append(package)
            
            # Report findings
            if missing_packages:
                self._log_to_ui(f"⚠️ Found {len(missing_packages)} missing packages:", "warning")
                for pkg in missing_packages[:5]:  # Show first 5
                    self._log_to_ui(f"   - {pkg}", "warning")
                if len(missing_packages) > 5:
                    self._log_to_ui(f"   ... and {len(missing_packages) - 5} more", "warning")
                
                self._log_to_ui("💡 Tip: Click 'Install Selected' to install missing packages", "info")
            else:
                self._log_to_ui("✅ All selected packages are installed!", "success")
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to check missing packages: {e}")
            self._log_to_ui(f"❌ Missing package check error: {str(e)}", "error")
    
    def _handle_update_click(self, _=None) -> None:
        """Handle update button click."""
        try:
            self._log_to_ui("⬆️ Updating packages...", "info")
            
            # Execute update operation
            result = self.execute_update_operation()
            
            if result.get("success", False):
                self._log_to_ui("✅ Update completed successfully!", "success")
                self.refresh_package_status()
            else:
                error_msg = result.get("message", "Update failed")
                self._log_to_ui(f"❌ Update failed: {error_msg}", "error")
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Update click failed: {e}")
            self._log_to_ui(f"❌ Update error: {str(e)}", "error")
    
    def _handle_uninstall_click(self, _=None) -> None:
        """Handle uninstall button click."""
        try:
            selected_packages = self._get_selected_packages()
            if not selected_packages:
                self._log_to_ui("⚠️ No packages selected for uninstallation", "warning")
                return
            
            self._log_to_ui(f"🗑️ Uninstalling {len(selected_packages)} packages...", "info")
            
            # Execute uninstall operation
            result = self.execute_uninstall_operation(selected_packages)
            
            if result.get("success", False):
                self._log_to_ui("✅ Uninstallation completed successfully!", "success")
                self.refresh_package_status()
            else:
                error_msg = result.get("message", "Uninstallation failed")
                self._log_to_ui(f"❌ Uninstallation failed: {error_msg}", "error")
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Uninstall click failed: {e}")
            self._log_to_ui(f"❌ Uninstall error: {str(e)}", "error")
    
    def _get_selected_packages(self) -> List[str]:
        """Get list of selected packages from UI."""
        selected = []
        
        # Get packages from all category checkboxes
        package_checkboxes = self.get_component("package_checkboxes") or {}
        package_categories = self.get_config("package_categories", {})
        
        for category_key, checkboxes in package_checkboxes.items():
            category = package_categories.get(category_key, {})
            packages = category.get("packages", [])
            
            for i, checkbox in enumerate(checkboxes):
                if checkbox.value and i < len(packages):
                    selected.append(packages[i]['pip_name'])
        
        # Get custom packages from text area
        custom_packages = self.get_component("custom_packages")
        if custom_packages and custom_packages.value:
            custom_lines = [line.strip() for line in custom_packages.value.split('\n') if line.strip()]
            selected.extend(custom_lines)
        
        return selected
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """Log message to UI components using operation container's log_accordion."""
        try:
            # Get operation container and use its log_accordion
            operation_container = self.get_component("operation_container")
            if operation_container and hasattr(operation_container, 'log'):
                # Map log levels to LogLevel enum if needed
                from smartcash.ui.components.log_accordion import LogLevel
                level_map = {
                    'info': LogLevel.INFO,
                    'success': LogLevel.INFO,
                    'warning': LogLevel.WARNING,
                    'error': LogLevel.ERROR,
                    'debug': LogLevel.DEBUG
                }
                log_level = level_map.get(level, LogLevel.INFO)
                operation_container.log(message, log_level)
            else:
                # Fallback to logger if operation container not available
                getattr(logger, level, get_module_logger("smartcash.ui.setup.dependency.uimodule").info)(message)
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"UI logging failed: {e}")
    
    def _update_progress(self, progress: int, message: str = "", level: str = "primary") -> None:
        """Update progress tracker using operation container."""
        try:
            operation_container = self.get_component("operation_container")
            if operation_container and hasattr(operation_container, 'update_progress'):
                operation_container.update_progress(progress, message, level)
            else:
                # Fallback to logging
                self._log_to_ui(f"Progress {progress}%: {message}", "info")
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"Progress update failed: {e}")
    
    def execute_install_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute package installation synchronously."""
        if not self._operation_manager:
            return {"success": False, "message": "Operation manager not initialized"}
        
        try:
            # Show progress start
            self._update_progress(0, "Starting installation...")
            
            # Execute install operation synchronously using the operation manager's sync method
            if hasattr(self._operation_manager, 'execute_install_sync'):
                result = self._operation_manager.execute_install_sync(packages)
            else:
                # Fallback to direct pip installation if sync method not available
                result = self._execute_pip_install_sync(packages)
            
            # Update progress based on result
            if result.get("success", False):
                self._update_progress(100, "Installation completed!")
                return {
                    "success": True,
                    "message": result.get("message", "Installation completed"),
                    "data": result
                }
            else:
                self._update_progress(100, "Installation failed!")
                return {
                    "success": False,
                    "message": result.get("error", "Installation failed"),
                    "data": result
                }
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to execute install: {e}")
            self._update_progress(100, f"Installation error: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def execute_uninstall_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute package uninstallation synchronously."""
        if not self._operation_manager:
            return {"success": False, "message": "Operation manager not initialized"}
        
        try:
            # Show progress start
            self._update_progress(0, "Starting uninstallation...")
            
            # Execute uninstall operation synchronously
            if hasattr(self._operation_manager, 'execute_uninstall_sync'):
                result = self._operation_manager.execute_uninstall_sync(packages)
            else:
                # Fallback to direct pip uninstallation if sync method not available
                result = self._execute_pip_uninstall_sync(packages)
            
            # Update progress based on result
            if result.get("success", False):
                self._update_progress(100, "Uninstallation completed!")
                return {
                    "success": True,
                    "message": result.get("message", "Uninstallation completed"),
                    "data": result
                }
            else:
                self._update_progress(100, "Uninstallation failed!")
                return {
                    "success": False,
                    "message": result.get("error", "Uninstallation failed"),
                    "data": result
                }
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to execute uninstall: {e}")
            self._update_progress(100, f"Uninstallation error: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def execute_update_operation(self) -> Dict[str, Any]:
        """Execute package update synchronously."""
        if not self._operation_manager:
            return {"success": False, "message": "Operation manager not initialized"}
        
        try:
            # Show progress start
            self._update_progress(0, "Starting update...")
            
            # Get all installed packages for update
            selected_packages = self._get_selected_packages()
            
            # Execute update operation synchronously
            if hasattr(self._operation_manager, 'execute_update_sync'):
                result = self._operation_manager.execute_update_sync(selected_packages)
            else:
                # Fallback to direct pip update if sync method not available
                result = self._execute_pip_update_sync(selected_packages)
            
            # Update progress based on result
            if result.get("success", False):
                self._update_progress(100, "Update completed!")
                return {
                    "success": True,
                    "message": result.get("message", "Update completed"),
                    "data": result
                }
            else:
                self._update_progress(100, "Update failed!")
                return {
                    "success": False,
                    "message": result.get("error", "Update failed"),
                    "data": result
                }
                
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to execute update: {e}")
            self._update_progress(100, f"Update error: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def get_package_status_summary(self) -> Dict[str, Any]:
        """Get summary of package installation status."""
        return {
            "module": self.full_module_name,
            "package_status": self._package_status,
            "timestamp": datetime.now().isoformat()
        }
    
    def refresh_package_status(self) -> None:
        """Refresh package status and update UI."""
        try:
            # Get status of all packages
            all_packages = self._get_all_packages()
            status_info = []
            
            for package in all_packages:
                status = self.get_package_status(package)
                status_info.append(status)
            
            # Update internal status without logging to UI to avoid extra log_output
            
            self._package_status = {s["package"]: s for s in status_info}
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("✅ Package status refreshed")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to refresh package status: {e}")
    
    def _get_all_packages(self) -> List[str]:
        """Get list of all packages from all categories + custom."""
        all_packages = []
        
        # Get packages from all categories
        package_categories = self.get_config("package_categories", {})
        for category_key, category in package_categories.items():
            if category_key == 'custom_packages':  # Skip custom packages category
                continue
            packages = category.get("packages", [])
            all_packages.extend([pkg['name'] for pkg in packages])
        
        # Custom packages from UI
        custom_packages = self.get_component("custom_packages")
        if custom_packages and custom_packages.value:
            custom_lines = [line.strip() for line in custom_packages.value.split('\n') if line.strip()]
            # Extract package names (before any version specifiers)
            for line in custom_lines:
                pkg_name = line.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
                if pkg_name:
                    all_packages.append(pkg_name)
        
        return all_packages
    
    def get_operation_manager(self) -> Optional[DependencyOperationManager]:
        """Get the operation manager instance."""
        return self._operation_manager
    
    def get_config_handler(self) -> Optional[DependencyConfigHandler]:
        """Get the config handler instance."""
        return self._config_handler
    
    def _setup_ui_logging_bridge(self, operation_container: Any) -> None:
        """Setup UI logging bridge to capture backend service logs."""
        try:
            import logging
            
            # Create custom handler for backend services
            class BackendUILogHandler(logging.Handler):
                def __init__(self, log_func):
                    super().__init__()
                    self.log_func = log_func
                    self.setFormatter(logging.Formatter('%(name)s: %(message)s'))
                
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        level = 'info' if record.levelno == logging.INFO else 'error'
                        self.log_func(msg, level)
                    except Exception:
                        pass  # Silently fail to avoid recursive errors
            
            # Get log function from operation container
            if hasattr(operation_container, 'log_message'):
                log_func = operation_container.log_message
            elif hasattr(operation_container, 'log'):
                log_func = operation_container.log
            else:
                # Fallback to internal logging
                log_func = self._log_to_ui
            
            # Create handler
            ui_handler = BackendUILogHandler(log_func)
            ui_handler.setLevel(logging.INFO)
            
            # Target specific backend service loggers that might log during dependency operations
            target_loggers = [
                'smartcash.dataset',
                'smartcash.model', 
                'smartcash.setup.dependency',
                'smartcash.core',
                'pip',
                'subprocess'
            ]
            
            # Remove existing console handlers and add UI handlers
            for logger_name in target_loggers:
                logger = logging.getLogger(logger_name)
                
                # Remove existing console handlers
                for handler in logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        logger.removeHandler(handler)
                
                # Add UI handler
                logger.addHandler(ui_handler)
            
            # Store handler for cleanup
            if not hasattr(self, '_ui_handlers'):
                self._ui_handlers = []
            self._ui_handlers.append(ui_handler)
            
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("🌉 UI logging bridge setup completed")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to setup UI logging bridge: {e}")
    
    def _initialize_progress_display(self) -> None:
        """Initialize progress tracker display."""
        try:
            operation_container = self.get_component("operation_container")
            if operation_container and hasattr(operation_container, 'progress_tracker'):
                progress_tracker = operation_container.progress_tracker
                if hasattr(progress_tracker, 'initialize') and not getattr(progress_tracker, '_initialized', False):
                    progress_tracker.initialize()
                if hasattr(progress_tracker, 'show'):
                    progress_tracker.show()
                    
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("📊 Progress display initialized")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"Progress display initialization failed: {e}")
    
    def _cleanup_ui_logging_bridge(self) -> None:
        """Cleanup UI logging bridge handlers."""
        try:
            if hasattr(self, '_ui_handlers'):
                import logging
                for handler in self._ui_handlers:
                    # Remove handler from all loggers
                    for logger_name in logging.Logger.manager.loggerDict:
                        logger = logging.getLogger(logger_name)
                        if handler in logger.handlers:
                            logger.removeHandler(handler)
                self._ui_handlers.clear()
                
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("🧹 UI logging bridge cleanup completed")
            
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"UI logging bridge cleanup failed: {e}")
    
    def _execute_pip_install_sync(self, packages: List[str]) -> Dict[str, Any]:
        """Fallback synchronous pip install method."""
        try:
            import subprocess
            import sys
            
            # Update progress
            self._update_progress(25, f"Installing {len(packages)} packages...")
            self._log_to_ui(f"📦 Installing packages: {', '.join(packages)}", "info")
            
            success_count = 0
            failed_packages = []
            
            for i, package in enumerate(packages):
                try:
                    # Update progress for each package
                    progress = 25 + (50 * i // len(packages))
                    self._update_progress(progress, f"Installing {package}...")
                    
                    # Execute pip install
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minute timeout per package
                    )
                    
                    if result.returncode == 0:
                        success_count += 1
                        self._log_to_ui(f"✅ Successfully installed {package}", "info")
                    else:
                        failed_packages.append(package)
                        self._log_to_ui(f"❌ Failed to install {package}: {result.stderr.strip()}", "error")
                        
                except subprocess.TimeoutExpired:
                    failed_packages.append(package)
                    self._log_to_ui(f"⏰ Timeout installing {package}", "error")
                except Exception as e:
                    failed_packages.append(package)
                    self._log_to_ui(f"❌ Error installing {package}: {str(e)}", "error")
            
            # Final result
            if failed_packages:
                message = f"Installed {success_count}/{len(packages)} packages. Failed: {', '.join(failed_packages)}"
                return {"success": False, "message": message, "failed_packages": failed_packages}
            else:
                message = f"Successfully installed all {success_count} packages"
                return {"success": True, "message": message}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_pip_uninstall_sync(self, packages: List[str]) -> Dict[str, Any]:
        """Fallback synchronous pip uninstall method."""
        try:
            import subprocess
            import sys
            
            # Update progress
            self._update_progress(25, f"Uninstalling {len(packages)} packages...")
            self._log_to_ui(f"🗑️ Uninstalling packages: {', '.join(packages)}", "info")
            
            success_count = 0
            failed_packages = []
            
            for i, package in enumerate(packages):
                try:
                    # Update progress for each package
                    progress = 25 + (50 * i // len(packages))
                    self._update_progress(progress, f"Uninstalling {package}...")
                    
                    # Execute pip uninstall
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "uninstall", package, "-y"],
                        capture_output=True,
                        text=True,
                        timeout=60  # 1 minute timeout per package
                    )
                    
                    if result.returncode == 0:
                        success_count += 1
                        self._log_to_ui(f"✅ Successfully uninstalled {package}", "info")
                    else:
                        failed_packages.append(package)
                        self._log_to_ui(f"❌ Failed to uninstall {package}: {result.stderr.strip()}", "error")
                        
                except subprocess.TimeoutExpired:
                    failed_packages.append(package)
                    self._log_to_ui(f"⏰ Timeout uninstalling {package}", "error")
                except Exception as e:
                    failed_packages.append(package)
                    self._log_to_ui(f"❌ Error uninstalling {package}: {str(e)}", "error")
            
            # Final result
            if failed_packages:
                message = f"Uninstalled {success_count}/{len(packages)} packages. Failed: {', '.join(failed_packages)}"
                return {"success": False, "message": message, "failed_packages": failed_packages}
            else:
                message = f"Successfully uninstalled all {success_count} packages"
                return {"success": True, "message": message}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _execute_pip_update_sync(self, packages: List[str]) -> Dict[str, Any]:
        """Fallback synchronous pip update method."""
        try:
            import subprocess
            import sys
            
            # Update progress
            self._update_progress(25, f"Updating {len(packages)} packages...")
            self._log_to_ui(f"⬆️ Updating packages: {', '.join(packages)}", "info")
            
            success_count = 0
            failed_packages = []
            
            for i, package in enumerate(packages):
                try:
                    # Update progress for each package
                    progress = 25 + (50 * i // len(packages))
                    self._update_progress(progress, f"Updating {package}...")
                    
                    # Execute pip install --upgrade
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", package],
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minute timeout per package
                    )
                    
                    if result.returncode == 0:
                        success_count += 1
                        self._log_to_ui(f"✅ Successfully updated {package}", "info")
                    else:
                        failed_packages.append(package)
                        self._log_to_ui(f"❌ Failed to update {package}: {result.stderr.strip()}", "error")
                        
                except subprocess.TimeoutExpired:
                    failed_packages.append(package)
                    self._log_to_ui(f"⏰ Timeout updating {package}", "error")
                except Exception as e:
                    failed_packages.append(package)
                    self._log_to_ui(f"❌ Error updating {package}: {str(e)}", "error")
            
            # Final result
            if failed_packages:
                message = f"Updated {success_count}/{len(packages)} packages. Failed: {', '.join(failed_packages)}"
                return {"success": False, "message": message, "failed_packages": failed_packages}
            else:
                message = f"Successfully updated all {success_count} packages"
                return {"success": True, "message": message}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._cleanup_ui_logging_bridge()
            super().cleanup()
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"Cleanup error: {e}")

def create_dependency_uimodule(config: Dict[str, Any] = None, 
                              auto_initialize: bool = True,
                              force_new: bool = False) -> DependencyUIModule:
    """Create Dependency UIModule using factory pattern.
    
    Args:
        config: Dependency configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        force_new: Force creation of new instance
        
    Returns:
        DependencyUIModule instance
    """
    global _dependency_uimodule
    
    # Return existing instance if available and not forcing new
    if not force_new and _dependency_uimodule is not None:
        if config:
            _dependency_uimodule.update_config(**config)
        return _dependency_uimodule
    
    try:
        # Ensure template is registered
        register_dependency_template()
        
        # Ensure shared methods are registered
        register_dependency_shared_methods()
        
        # Create new module instance
        module = DependencyUIModule(config)
        
        # Initialize if requested
        if auto_initialize:
            module.initialize()
        
        # Store global reference
        _dependency_uimodule = module
        
        get_module_logger("smartcash.ui.setup.dependency.uimodule").debug(f"🏭 Created Dependency UIModule")
        return module
        
    except Exception as e:
        get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"❌ Failed to create Dependency UIModule: {e}")
        raise

def get_dependency_uimodule(create_if_missing: bool = True, **kwargs) -> Optional[DependencyUIModule]:
    """Get existing Dependency UIModule instance.
    
    Args:
        create_if_missing: Create new instance if none exists
        **kwargs: Arguments for create_dependency_uimodule if creating
        
    Returns:
        DependencyUIModule instance or None
    """
    global _dependency_uimodule
    
    if _dependency_uimodule is None and create_if_missing:
        _dependency_uimodule = create_dependency_uimodule(**kwargs)
    
    return _dependency_uimodule

def reset_dependency_uimodule() -> None:
    """Reset global Dependency UIModule instance."""
    global _dependency_uimodule
    
    if _dependency_uimodule is not None:
        try:
            _dependency_uimodule.cleanup()
        except Exception as e:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"Error during cleanup: {e}")
        finally:
            _dependency_uimodule = None
    
    get_module_logger("smartcash.ui.setup.dependency.uimodule").debug("🔄 Reset global Dependency UIModule instance")

# === Backward Compatibility Layer ===

@handle_ui_errors(return_type=None)
def initialize_dependency_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """Initialize Dependency UI using new UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI immediately (True) or return components (False)
        **kwargs: Additional keyword arguments for module creation
        
    Returns:
        None if display=True, dict of components if display=False
    """
    try:
        from IPython.display import display as ipython_display
        
        # Get the module and UI components
        module = create_dependency_uimodule(config, auto_initialize=True, **kwargs)
        ui_components = {
            component_type: module.get_component(component_type)
            for component_type in module.list_components()
        }
        
        main_ui = ui_components.get('main_container') or ui_components.get('ui')
        
        # Setup UI logging bridge to capture backend service logs
        operation_container = ui_components.get('operation_container')
        if operation_container and hasattr(module, '_setup_ui_logging_bridge'):
            module._setup_ui_logging_bridge(operation_container)
        
        # Initialize progress display
        if hasattr(module, '_initialize_progress_display'):
            module._initialize_progress_display()
        
        if display and main_ui:
            ipython_display(main_ui)
            return None
        
        # Return components without displaying
        result = {
            'success': True,
            'module': module,
            'ui_components': ui_components,
            'main_ui': main_ui
        }
        
        return result
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'module': None,
            'ui_components': {},
            'main_ui': None
        }
        
        if display:
            get_module_logger("smartcash.ui.setup.dependency.uimodule").error(f"Failed to initialize dependency UI: {e}")
            return None
        
        return error_result

@handle_ui_errors(return_type=dict)
def get_dependency_components(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get Dependency components using new UIModule pattern."""
    module = create_dependency_uimodule(config, auto_initialize=True)
    return {
        component_type: module.get_component(component_type)
        for component_type in module.list_components()
    }

@handle_ui_errors(return_type=None)
def display_dependency_ui(config: Dict[str, Any] = None) -> None:
    """Display Dependency UI using new UIModule pattern."""
    initialize_dependency_ui(config)