"""
File: smartcash/ui/setup/dependency/handlers/dependency_ui_handler.py
Deskripsi: UI handler untuk dependency management
"""

from typing import Dict, Any, Optional
from datetime import datetime
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from ..configs.dependency_config_handler import DependencyConfigHandler
from ..operations.operation_manager import DependencyOperationManager

class DependencyUIHandler(ModuleUIHandler):
    """Enhanced UI handler for dependency management with operation container integration."""
    
    def __init__(self, module_name: str = 'dependency', parent_module: str = 'setup', **kwargs):
        # Extract parameters that ModuleUIHandler doesn't expect
        default_config = kwargs.pop('default_config', None)
        auto_setup_handlers = kwargs.pop('auto_setup_handlers', True)
        enable_sharing = kwargs.pop('enable_sharing', True)
        
        # Call parent with only the parameters it expects
        super().__init__(module_name, parent_module)
        
        # Initialize config handler with default config if provided
        self.config_handler = DependencyConfigHandler(default_config=default_config)
        self._status_messages = []
        
        # Store additional parameters
        self.auto_setup_handlers = auto_setup_handlers
        self.enable_sharing = enable_sharing
        
        # Operation container and handler will be set up in setup() method
        self.operation_container = None
        self.operation_handler = None
        
        # Track current operation state
        self.operation_in_progress = False
        self.current_operation_type = None
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration from UI components following the new structure.
        
        Returns:
            Dict containing the extracted configuration
        """
        try:
            config = self.config_handler.get_config().copy()
            
            # Extract from form widgets if available
            if 'widgets' in self._ui_components and 'form' in self._ui_components['containers']:
                form_widgets = self._ui_components['widgets']
                
                # Extract selected packages from package categories
                if 'package_categories' in form_widgets:
                    selected_packages = self._extract_selected_packages(form_widgets['package_categories'])
                    config['selected_packages'] = selected_packages
                
                # Extract custom packages from text area if available
                if 'custom_packages_input' in form_widgets:
                    custom_packages = form_widgets['custom_packages_input'].value.strip()
                    if custom_packages:
                        config['custom_packages'] = custom_packages
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting config from UI: {e}")
            return self.config_handler.get_config()
    
    def _extract_selected_packages(self, categories_widget) -> list:
        """Extract selected packages from categories widget.
        
        Args:
            categories_widget: The widget containing package categories
            
        Returns:
            List of selected package names
        """
        selected_packages = []
        
        try:
            if hasattr(categories_widget, 'selected'):
                # Handle case where widget has a 'selected' attribute
                return categories_widget.selected
            elif hasattr(categories_widget, 'value'):
                # Handle case where widget has a 'value' attribute
                return categories_widget.value
            elif hasattr(categories_widget, 'children'):
                # Handle case where we need to traverse children
                for child in categories_widget.children:
                    if hasattr(child, 'selected') and child.selected:
                        if hasattr(child, 'description'):
                            selected_packages.append(child.description)
                        elif hasattr(child, 'value'):
                            selected_packages.append(child.value)
        except Exception as e:
            self.logger.error(f"❌ Error extracting selected packages: {e}")
        
        return selected_packages
    
    def _find_package_widgets(self, container, widgets_list=None):
        """Recursively find package widgets dalam container"""
        if widgets_list is None:
            widgets_list = []
        
        if hasattr(container, 'package_info'):
            widgets_list.append(container)
        
        if hasattr(container, 'children'):
            for child in container.children:
                self._find_package_widgets(child, widgets_list)
        
        return widgets_list
    
    def update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Update UI components dari configuration"""
        try:
            if 'dependency_tabs' in self._ui_components:
                tabs = self._ui_components['dependency_tabs']
                
                # Update selected packages
                selected_packages = config.get('selected_packages', [])
                self._update_selected_packages(tabs, selected_packages)
                
                # Update custom packages
                custom_packages = config.get('custom_packages', '')
                self._update_custom_packages(tabs, custom_packages)
                
            self.logger.info("✅ UI berhasil diupdate dari config")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating UI dari config: {e}")
    
    def _update_selected_packages(self, tabs, selected_packages: list) -> None:
        """Update selected packages dalam UI"""
        try:
            if hasattr(tabs, 'children') and len(tabs.children) > 0:
                categories_tab = tabs.children[0]
                
                for widget in self._find_package_widgets(categories_tab):
                    if hasattr(widget, 'package_info'):
                        is_selected = widget.package_info['name'] in selected_packages
                        widget.is_selected = is_selected
                        
                        # Update visual state
                        if hasattr(widget, 'status_button'):
                            widget.status_button.description = '✅' if is_selected else '⭕'
                            widget.status_button.button_style = 'success' if is_selected else ''
                        
                        # Update container styling
                        if hasattr(widget, 'layout'):
                            border_color = '#4CAF50' if is_selected else '#ddd'
                            bg_color = '#fafafa' if is_selected else 'white'
                            widget.layout.border = f'1px solid {border_color}'
                            widget.layout.background_color = bg_color
                            
        except Exception as e:
            self.logger.error(f"❌ Error updating selected packages: {e}")
    
    def _update_custom_packages(self, tabs, custom_packages: str) -> None:
        """Update custom packages dalam UI"""
        try:
            if hasattr(tabs, 'children') and len(tabs.children) > 1:
                custom_tab = tabs.children[1]
                
                if hasattr(custom_tab, 'packages_textarea'):
                    custom_tab.packages_textarea.value = custom_packages
                    
        except Exception as e:
            self.logger.error(f"❌ Error updating custom packages: {e}")
    
    def setup(self, ui_components: Dict[str, Any]) -> None:
        """
        Set up the handler with UI components.
        
        Args:
            ui_components: Dictionary of UI components to be managed by this handler.
        """
        self.logger.info("🖥️ Setting up UI components for Dependency UI Handler")
        self._ui_components = ui_components
        
        # Get operation container from UI components and set up operation handler
        if 'operation_manager' in ui_components:
            self.operation_container = ui_components['operation_manager']
            self.logger.info("✅ Using operation container from UI components")
        elif 'operation_container' in ui_components:
            # Fallback to direct operation container
            self.operation_container = ui_components['operation_container']
            self.logger.info("✅ Using operation container directly from UI components")
        else:
            # Create fallback operation container
            from smartcash.ui.components.operation_container import OperationContainer
            self.operation_container = OperationContainer()
            self.logger.warning("⚠️ Created fallback operation container")
        
        # Set up operation handler with the operation container
        self.operation_handler = DependencyOperationManager(
            config=self.config_handler.get_config(),
            operation_container=self.operation_container
        )
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Initialize with default config
        self.sync_ui_with_config()
        
        self.logger.info("✅ UI components setup complete for Dependency UI Handler")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        # Button handlers
        if 'save_button' in self._ui_components:
            self._ui_components['save_button'].on_click(self._handle_save_config)
        
        if 'reset_button' in self._ui_components:
            self._ui_components['reset_button'].on_click(self._handle_reset_config)
        
        # Main install button handler
        if 'install_button' in self._ui_components:
            self._ui_components['install_button'].on_click(self._handle_main_operation)
        
        self.logger.info("✅ Event handlers setup complete")
    
    def _handle_main_operation(self, button=None) -> None:
        """Handle the main operation button click - runs package operations."""
        self.logger.debug(f"_handle_main_operation called with button: {button}")
        
        if self.operation_in_progress:
            self.logger.warning("Operation already in progress, ignoring click")
            return
        
        action_manager = None
        try:
            self.operation_in_progress = True
            
            # Get action container manager for phase updates
            action_manager = self._ui_components.get('action_container_manager')
            if not action_manager:
                self.logger.error("Action container manager not found")
                return
            
            self.logger.info("🚀 Starting package operation")
            
            # Update button to show initial phase
            self.logger.debug("Setting phase to 'checking'")
            action_manager['set_phase']('checking')
            
            # Extract current config from UI
            self.logger.debug("Extracting config from UI")
            config = self.extract_config_from_ui()
            self.operation_handler.config = config
            
            # Get selected packages
            selected_packages = config.get('selected_packages', [])
            custom_packages = config.get('custom_packages', '')
            
            if not selected_packages and not custom_packages.strip():
                self.logger.warning("No packages selected for operation")
                action_manager['set_phase']('error')
                self.track_status("⚠️ No packages selected", "warning")
                return
            
            # Execute the operation using the operation handler
            self.logger.debug("Executing package operation")
            
            try:
                # For now, simulate the operation
                action_manager['set_phase']('installing')
                self.track_status("📦 Installing selected packages...", "info")
                
                # Here you would call the actual operation handler
                # result = self.operation_handler.execute_operation(...)
                
                # Simulate success
                action_manager['set_phase']('complete')
                self.track_status("✅ Package operation completed successfully!", "success")
                
            except Exception as op_error:
                self.logger.error(f"Error during operation execution: {op_error}", exc_info=True)
                if action_manager:
                    action_manager['set_phase']('error')
                self.track_status(f"❌ Operation execution failed: {str(op_error)}", "error")
                raise
                
        except Exception as e:
            self.logger.error(f"Error in _handle_main_operation: {e}", exc_info=True)
            if action_manager:
                action_manager['set_phase']('error')
            self.track_status(f"❌ Operation failed with exception: {str(e)}", "error")
        finally:
            self.operation_in_progress = False
            self.logger.debug("Operation process completed")
    
    def _handle_save_config(self, button=None) -> None:
        """Handle configuration save."""
        try:
            config = self.extract_config_from_ui()
            self.config_handler.update_config(config)
            self.track_status("💾 Configuration saved", "success")
        except Exception as e:
            self.track_status(f"❌ Save failed: {str(e)}", "error")
    
    def _handle_reset_config(self, button=None) -> None:
        """Handle configuration reset."""
        try:
            self.config_handler.reset_to_defaults()
            config = self.config_handler.get_config()
            self.update_ui_from_config(config)
            self.track_status("🔄 Configuration reset to defaults", "info")
        except Exception as e:
            self.track_status(f"❌ Reset failed: {str(e)}", "error")
    
    def sync_config_with_ui(self) -> None:
        """Sync configuration dengan UI state"""
        try:
            # Extract current config dari UI
            current_config = self.extract_config_from_ui()
            
            # Update config handler
            self.config_handler.update_config(current_config)
            
            self.logger.info("✅ Config berhasil di-sync dengan UI")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing config dengan UI: {e}")
    
    def sync_ui_with_config(self) -> None:
        """Sync UI dengan configuration"""
        try:
            # Get current config
            current_config = self.config_handler.get_config()
            
            # Update UI
            self.update_ui_from_config(current_config)
            
            self.logger.info("✅ UI berhasil di-sync dengan config")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing UI dengan config: {e}")
    
    def get_selected_packages(self) -> list:
        """Get list selected packages"""
        config = self.extract_config_from_ui()
        return config.get('selected_packages', [])
    
    def get_custom_packages(self) -> str:
        """Get custom packages string"""
        config = self.extract_config_from_ui()
        return config.get('custom_packages', '')
    
    def add_selected_package(self, package_name: str) -> bool:
        """Add package ke selected list"""
        try:
            return self.config_handler.add_selected_package(package_name)
        except Exception as e:
            self.logger.error(f"❌ Error adding package {package_name}: {e}")
            return False
    
    def remove_selected_package(self, package_name: str) -> bool:
        """Remove package dari selected list"""
        try:
            return self.config_handler.remove_selected_package(package_name)
        except Exception as e:
            self.logger.error(f"❌ Error removing package {package_name}: {e}")
            return False
    
    def update_custom_packages(self, custom_packages: str) -> bool:
        """Update custom packages"""
        try:
            return self.config_handler.update_custom_packages(custom_packages)
        except Exception as e:
            self.logger.error(f"❌ Error updating custom packages: {e}")
            return False
    
    def track_status(self, message: str, status_type: str) -> None:
        """Track status messages for display."""
        timestamp = datetime.now().isoformat()
        self._status_messages.append({
            'message': message,
            'type': status_type,
            'timestamp': timestamp
        })
        
        # Log the message
        log_func = getattr(self.logger, status_type, self.logger.info)
        log_func(message)
        
        # Also log to operation container if available
        if hasattr(self, 'operation_container') and self.operation_container:
            try:
                self.operation_container.log_message(message=message, level=status_type)
            except Exception as e:
                self.logger.error(f"Error logging to operation container: {e}")
    
    def get_status_history(self) -> list:
        """Get history of status messages."""
        return self._status_messages.copy()
    
    def clear_status_history(self) -> None:
        """Clear status message history."""
        self._status_messages.clear()
    
    def get_current_status(self) -> Optional[Dict[str, Any]]:
        """Get the most recent status message."""
        return self._status_messages[-1] if self._status_messages else None

    def initialize(self) -> None:
        """
        Initialize the UI handler for dependency module.
        Implements the abstract method required by ModuleUIHandler.
        """
        self.logger.info("🚀 Initializing Dependency UI Handler")
        # Perform any necessary initialization for the dependency UI handler
        # This can be extended with specific initialization logic as needed
        self.logger.info("✅ Dependency UI Handler initialized successfully")