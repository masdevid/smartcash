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
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Import existing dependency components and handlers
from smartcash.ui.setup.dependency.configs.dependency_config_handler import DependencyConfigHandler
from smartcash.ui.setup.dependency.configs.dependency_defaults import get_default_dependency_config
from smartcash.ui.setup.dependency.operations.operation_manager import DependencyOperationManager

logger = get_module_logger("smartcash.ui.setup.dependency.uimodule")

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
        logger.debug("📋 Registered Dependency template")
    except Exception as e:
        logger.error(f"❌ Failed to register template: {e}")

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
    
    logger.debug("🔗 Registered Dependency shared methods")

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
            
            logger.debug(f"✅ Initialized Dependency UIModule")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Dependency UIModule: {e}")
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
            
            logger.debug(f"📦 Created {len(ui_components)} UI components")
            
        except Exception as e:
            logger.error(f"❌ Failed to create UI components: {e}")
            raise
    
    
    def _setup_operation_manager(self) -> None:
        """Setup operation manager for Dependency operations."""
        try:
            # Create wrapper for operation container
            operation_container_wrapper = self._create_operation_container_wrapper()
            
            # Get operation container component
            operation_container = self.get_component("operation_container")
            
            self._operation_manager = DependencyOperationManager(
                config=self.get_config(),
                operation_container=operation_container_wrapper,
                ui_components={'operation_container': operation_container}
            )
            
            logger.debug("⚙️ Setup operation manager")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup operation manager: {e}")
            raise
    
    def _create_operation_container_wrapper(self):
        """Create a wrapper object that provides operation container methods."""
        class OperationContainerWrapper:
            def __init__(self, module):
                self.module = module
            
            def update_progress(self, progress, message="", level="primary", **_):
                """Update progress bar."""
                try:
                    progress_tracker = self.module.get_component("progress_tracker")
                    if progress_tracker and hasattr(progress_tracker, 'set_progress'):
                        progress_tracker.set_progress(progress, message, level)
                except Exception as e:
                    logger.debug(f"Progress update failed: {e}")
            
            def log_message(self, message, level='info'):
                """Log message to UI log output."""
                try:
                    # Try log_accordion first
                    log_accordion = self.module.get_component("log_accordion")
                    if log_accordion and hasattr(log_accordion, 'log'):
                        log_accordion.log(message, level)
                        return
                    
                    # Fallback to log_output
                    log_output = self.module.get_component("log_output")
                    if log_output and hasattr(log_output, 'log'):
                        log_output.log(message, level)
                        return
                        
                    # Last resort: use logger
                    getattr(logger, level, logger.info)(message)
                except Exception as e:
                    logger.debug(f"Log message failed: {e}")
        
        return OperationContainerWrapper(self)
    
    def _setup_config_handler(self) -> None:
        """Setup config handler."""
        try:
            self._config_handler = DependencyConfigHandler()
            logger.debug("🔧 Setup config handler")
        except Exception as e:
            logger.error(f"❌ Failed to setup config handler: {e}")
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
            
            logger.debug(f"⚙️ Registered {len(operations)} operations")
            
        except Exception as e:
            logger.error(f"❌ Failed to register operations: {e}")
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
            
            logger.debug("✅ Connected dependency button handlers")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup event handlers: {e}")
    
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
            logger.error(f"❌ Add packages failed: {e}")
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
            logger.error(f"❌ Failed to update UI after save: {e}")
    
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
            logger.error(f"❌ Install click failed: {e}")
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
            logger.error(f"❌ Check click failed: {e}")
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
            logger.error(f"❌ Failed to check missing packages: {e}")
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
            logger.error(f"❌ Update click failed: {e}")
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
            logger.error(f"❌ Uninstall click failed: {e}")
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
        """Log message to UI components."""
        try:
            # Log to operation container
            log_accordion = self.get_component("log_accordion")
            if log_accordion and hasattr(log_accordion, 'log'):
                log_accordion.log(message, level)
            
            # Also log to status output
            status_output = self.get_component("status_output")
            if status_output:
                with status_output:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        except Exception as e:
            logger.debug(f"UI logging failed: {e}")
    
    def _update_progress(self, progress: int, message: str = "", level: str = "primary") -> None:
        """Update progress tracker."""
        try:
            progress_tracker = self.get_component("progress_tracker")
            if progress_tracker and hasattr(progress_tracker, 'set_progress'):
                progress_tracker.set_progress(progress, message, level)
            else:
                # Fallback to logging
                self._log_to_ui(f"Progress {progress}%: {message}", "info")
        except Exception as e:
            logger.debug(f"Progress update failed: {e}")
    
    def execute_install_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute package installation."""
        if not self._operation_manager:
            return {"success": False, "message": "Operation manager not initialized"}
        
        try:
            # Show progress start
            self._update_progress(0, "Starting installation...")
            
            # Execute install operation - this should be async but we'll handle sync
            import asyncio
            try:
                # Try to run in existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task to run later
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self._operation_manager.execute_install(packages)
                        )
                        result = future.result(timeout=300)  # 5 minute timeout
                else:
                    result = asyncio.run(self._operation_manager.execute_install(packages))
            except RuntimeError:
                # No event loop running, create new one
                result = asyncio.run(self._operation_manager.execute_install(packages))
            
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
            logger.error(f"❌ Failed to execute install: {e}")
            self._update_progress(100, f"Installation error: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def execute_uninstall_operation(self, packages: List[str]) -> Dict[str, Any]:
        """Execute package uninstallation."""
        if not self._operation_manager:
            return {"success": False, "message": "Operation manager not initialized"}
        
        try:
            # Show progress start
            self._update_progress(0, "Starting uninstallation...")
            
            # Execute uninstall operation
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self._operation_manager.execute_uninstall(packages)
                        )
                        result = future.result(timeout=300)
                else:
                    result = asyncio.run(self._operation_manager.execute_uninstall(packages))
            except RuntimeError:
                result = asyncio.run(self._operation_manager.execute_uninstall(packages))
            
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
            logger.error(f"❌ Failed to execute uninstall: {e}")
            self._update_progress(100, f"Uninstallation error: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def execute_update_operation(self) -> Dict[str, Any]:
        """Execute package update."""
        if not self._operation_manager:
            return {"success": False, "message": "Operation manager not initialized"}
        
        try:
            # Show progress start
            self._update_progress(0, "Starting update...")
            
            # Get all installed packages for update
            selected_packages = self._get_selected_packages()
            
            # Execute update operation
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self._operation_manager.execute_update(selected_packages)
                        )
                        result = future.result(timeout=300)
                else:
                    result = asyncio.run(self._operation_manager.execute_update(selected_packages))
            except RuntimeError:
                result = asyncio.run(self._operation_manager.execute_update(selected_packages))
            
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
            logger.error(f"❌ Failed to execute update: {e}")
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
            
            # Update status display
            status_output = self.get_component("status_output")
            if status_output:
                with status_output:
                    status_output.clear_output()
                    print("📊 Package Status:")
                    for status in status_info:
                        icon = "✅" if status.get("installed", False) else "❌"
                        package_name = status.get("package", "unknown")
                        print(f"  {icon} {package_name}")
            
            self._package_status = {s["package"]: s for s in status_info}
            logger.debug("✅ Package status refreshed")
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh package status: {e}")
    
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
        
        logger.debug(f"🏭 Created Dependency UIModule")
        return module
        
    except Exception as e:
        logger.error(f"❌ Failed to create Dependency UIModule: {e}")
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
            logger.error(f"Error during cleanup: {e}")
        finally:
            _dependency_uimodule = None
    
    logger.debug("🔄 Reset global Dependency UIModule instance")

# === Backward Compatibility Layer ===

@handle_ui_errors(return_type=None)
def initialize_dependency_ui(config: Dict[str, Any] = None) -> None:
    """Initialize Dependency UI using new UIModule pattern."""
    from IPython.display import display
    
    module = create_dependency_uimodule(config, auto_initialize=True)
    main_container = module.get_component('main_container')
    if main_container:
        display(main_container)

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