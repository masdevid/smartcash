"""
File: smartcash/ui/setup/colab/handlers/colab_ui_handler.py  
Description: UI handler for colab module with single button sequential operations
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime

from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from smartcash.ui.components.operation_container import OperationContainer
from ..configs.colab_config_handler import ColabConfigHandler
from ..operations.operation_manager import ColabOperationManager


class ColabUIHandler(ModuleUIHandler):
    """UI handler for colab module with single button sequential operations."""
    
    def __init__(self, module_name: str = 'colab', parent_module: str = 'setup', **kwargs):
        """Initialize colab UI handler.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            **kwargs: Additional arguments
        """
        super().__init__(module_name, parent_module)
        
        self.config_handler = ColabConfigHandler()
        self._status_messages = []
        
        # Operation container and handler will be set up in setup() method
        self.operation_container = None
        self.operation_handler = None
        
        # Track current setup state
        self.setup_in_progress = False
        self.current_stage_index = 0
        
        # Setup stage sequence: INIT → DRIVE → SYMLINK → FOLDERS → CONFIG → ENV → VERIFY → COMPLETE
        self.setup_stages = [
            ('init', '⚙️ Initializing...'),
            ('drive', '📁 Mounting Drive...'),
            ('symlink', '🔗 Creating Symlinks...'),
            ('folders', '📂 Creating Folders...'),
            ('config', '⚙️ Syncing Config...'),
            ('env', '🌍 Setting Environment...'),
            ('verify', '🔍 Verifying Setup...'),
            ('complete', '✅ Environment Ready!')
        ]
    
    def setup(self, ui_components: Dict[str, Any]) -> None:
        """Set up the handler with UI components.
        
        Args:
            ui_components: Dictionary of UI components to be managed by this handler
        """
        self.logger.info("🖥️ Setting up UI components for Colab UI Handler")
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
        self.operation_handler = ColabOperationManager(
            config=self.config_handler.get_config(),
            operation_container=self.operation_container
        )
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Initialize with default config
        self.sync_ui_with_config()
        
        self.logger.info("✅ UI components setup complete for Colab UI Handler")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        # Configuration change handlers
        form_widgets = [
            'environment_type_dropdown', 'auto_mount_drive_checkbox', 
            'project_name_text', 'gpu_enabled_checkbox', 'gpu_type_dropdown',
            'setup_stages_select', 'auto_start_checkbox', 'stop_on_error_checkbox',
            'max_retries_int', 'show_advanced_checkbox'
        ]
        
        for widget_name in form_widgets:
            if widget_name in self._ui_components:
                widget = self._ui_components[widget_name]
                widget.observe(
                    lambda change, w=widget_name: self._on_form_change(w, change), 
                    names='value'
                )
        
        # Button handlers
        if 'save_button' in self._ui_components:
            self._ui_components['save_button'].on_click(self._handle_save_config)
        
        if 'reset_button' in self._ui_components:
            self._ui_components['reset_button'].on_click(self._handle_reset_config)
        
        # Main setup button handler - this is the key change!
        if 'setup_button' in self._ui_components:
            self._ui_components['setup_button'].on_click(self._handle_main_setup)
        
        self.logger.info("✅ Event handlers setup complete")
    
    def _handle_main_setup(self, button=None) -> None:
        """Handle the main setup button click - runs complete sequential setup.
        
        This is the single button that executes: INIT → DRIVE → SYMLINK → FOLDERS → CONFIG → ENV → VERIFY → COMPLETE
        """
        self.logger.debug(f"_handle_main_setup called with button: {button}")
        
        if self.setup_in_progress:
            self.logger.warning("Setup already in progress, ignoring click")
            return
        
        action_manager = None
        try:
            self.setup_in_progress = True
            self.current_stage_index = 0
            
            # Get action container manager for phase updates
            action_manager = self._ui_components.get('action_container_manager')
            if not action_manager:
                self.logger.error("Action container manager not found")
                return
            
            self.logger.info("🚀 Starting complete environment setup")
            
            # Update button to show initial stage
            self.logger.debug("Setting phase to 'init'")
            action_manager['set_phase']('init')
            
            # Extract current config from UI
            self.logger.debug("Extracting config from UI")
            config = self.extract_config_from_ui()
            self.operation_handler.config = config
            
            # Execute the full setup operation using the operation handler
            self.logger.debug("Executing full setup operation")
            
            # Debug logging for operation handler and operation
            self.logger.debug(f"Operation handler: {self.operation_handler}")
            self.logger.debug(f"Operation handler has _full_setup_operation: {hasattr(self.operation_handler, '_full_setup_operation')}")
            if hasattr(self.operation_handler, '_full_setup_operation'):
                op = self.operation_handler._full_setup_operation
                self.logger.debug(f"_full_setup_operation: {op}")
                self.logger.debug(f"_full_setup_operation callable: {callable(op)}")
            
            try:
                operation = self.operation_handler._full_setup_operation
                self.logger.debug(f"Operation to execute: {operation}")
                self.logger.debug(f"Operation type: {type(operation)}")
                self.logger.debug(f"Operation callable: {callable(operation)}")
                
                result = self.operation_handler.execute_operation(
                    operation,
                    operation_name="Environment Setup",
                    show_progress=True,
                    allow_cancel=True
                )
                self.logger.debug(f"Operation result: {result}")
                
                if hasattr(result, 'status') and hasattr(result.status, 'value'):
                    if result.status.value == 'completed':
                        self.logger.debug("Operation completed successfully")
                        action_manager['set_phase']('complete')
                        self.track_status("✅ Environment setup completed successfully!", "success")
                    else:
                        error_msg = str(result.error) if hasattr(result, 'error') and result.error else \
                                  result.message if hasattr(result, 'message') else 'Unknown error'
                        self.logger.error(f"Operation failed: {error_msg}")
                        action_manager['set_phase']('error')
                        self.track_status(f"❌ Setup failed: {error_msg}", "error")
                else:
                    self.logger.error(f"Unexpected result format: {result}")
                    action_manager['set_phase']('error')
                    self.track_status("❌ Setup failed with unexpected result format", "error")
                    
            except Exception as op_error:
                self.logger.error(f"Error during operation execution: {op_error}", exc_info=True)
                if action_manager:
                    action_manager['set_phase']('error')
                self.track_status(f"❌ Operation execution failed: {str(op_error)}", "error")
                raise
                
        except Exception as e:
            self.logger.error(f"Error in _handle_main_setup: {e}", exc_info=True)
            if action_manager:
                action_manager['set_phase']('error')
            self.track_status(f"❌ Setup failed with exception: {str(e)}", "error")
        finally:
            self.setup_in_progress = False
            self.logger.debug("Setup process completed")
    
    def _on_form_change(self, widget_name: str, change) -> None:
        """Handle form widget changes."""
        try:
            # Extract current config
            current_config = self.extract_config_from_ui()
            
            # Update config handler
            self.config_handler.update_config(current_config)
            
            # Update status summary
            if 'status_summary' in self._ui_components:
                self._update_status_summary(current_config)
            
            # Handle specific widget changes
            if widget_name == 'environment_type_dropdown':
                self._on_environment_type_change(change.get('new'))
            elif widget_name == 'gpu_enabled_checkbox':
                self._on_gpu_enabled_change(change.get('new'))
            elif widget_name == 'project_name_text':
                self._on_project_name_change(change.get('new'))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Form change handling error: {e}")
    
    def _on_environment_type_change(self, env_type: str) -> None:
        """Handle environment type change."""
        try:
            # Update UI based on environment capabilities
            env_info = self.config_handler.get_available_environments().get(env_type, {})
            
            # Update mount drive visibility
            if 'auto_mount_drive_checkbox' in self._ui_components:
                mount_required = env_info.get('mount_required', False)
                self._ui_components['auto_mount_drive_checkbox'].disabled = not mount_required
                if not mount_required:
                    self._ui_components['auto_mount_drive_checkbox'].value = False
            
            # Update GPU availability
            if 'gpu_enabled_checkbox' in self._ui_components:
                supports_gpu = env_info.get('supports_gpu', False)
                self._ui_components['gpu_enabled_checkbox'].disabled = not supports_gpu
                if not supports_gpu:
                    self._ui_components['gpu_enabled_checkbox'].value = False
            
            self.track_status(f"Environment changed to: {env_type}", "info")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Environment type change error: {e}")
    
    def _on_gpu_enabled_change(self, gpu_enabled: bool) -> None:
        """Handle GPU enabled change."""
        try:
            # Update GPU type dropdown visibility
            if 'gpu_type_dropdown' in self._ui_components:
                self._ui_components['gpu_type_dropdown'].disabled = not gpu_enabled
                if not gpu_enabled:
                    self._ui_components['gpu_type_dropdown'].value = 'none'
            
            status = "enabled" if gpu_enabled else "disabled"
            self.track_status(f"GPU {status}", "info")
            
        except Exception as e:
            self.logger.warning(f"⚠️ GPU enabled change error: {e}")
    
    def _on_project_name_change(self, project_name: str) -> None:
        """Handle project name change."""
        try:
            if project_name and project_name.strip():
                self.config_handler.set_project_name(project_name.strip())
                self.track_status(f"Project name updated: {project_name}", "info")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Project name change error: {e}")
    
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
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration from UI components."""
        try:
            config = self.config_handler.get_config().copy()
            
            # Extract from environment form components
            if 'environment_type_dropdown' in self._ui_components:
                env_type = self._ui_components['environment_type_dropdown'].value
                config['environment']['type'] = env_type
            
            if 'auto_mount_drive_checkbox' in self._ui_components:
                auto_mount = self._ui_components['auto_mount_drive_checkbox'].value
                config['environment']['auto_mount_drive'] = auto_mount
            
            if 'project_name_text' in self._ui_components:
                project_name = self._ui_components['project_name_text'].value.strip()
                if project_name:
                    config['environment']['project_name'] = project_name
            
            if 'gpu_enabled_checkbox' in self._ui_components:
                gpu_enabled = self._ui_components['gpu_enabled_checkbox'].value
                config['environment']['gpu_enabled'] = gpu_enabled
            
            if 'gpu_type_dropdown' in self._ui_components:
                gpu_type = self._ui_components['gpu_type_dropdown'].value
                config['environment']['gpu_type'] = gpu_type
            
            # Extract from setup form components
            if 'setup_stages_select' in self._ui_components:
                stages = list(self._ui_components['setup_stages_select'].value)
                config['setup']['stages'] = stages
            
            if 'auto_start_checkbox' in self._ui_components:
                auto_start = self._ui_components['auto_start_checkbox'].value
                config['setup']['auto_start'] = auto_start
            
            if 'stop_on_error_checkbox' in self._ui_components:
                stop_on_error = self._ui_components['stop_on_error_checkbox'].value
                config['setup']['stop_on_error'] = stop_on_error
            
            if 'max_retries_int' in self._ui_components:
                max_retries = self._ui_components['max_retries_int'].value
                config['setup']['max_retries'] = max_retries
            
            # Extract from UI options
            if 'show_advanced_checkbox' in self._ui_components:
                show_advanced = self._ui_components['show_advanced_checkbox'].value
                config['ui']['show_advanced_options'] = show_advanced
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting config from UI: {e}")
            return self.config_handler.get_config()
    
    def update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Update UI components from configuration."""
        try:
            env_config = config.get('environment', {})
            setup_config = config.get('setup', {})
            ui_config = config.get('ui', {})
            
            # Update environment components
            if 'environment_type_dropdown' in self._ui_components:
                env_type = env_config.get('type', 'colab')
                self._ui_components['environment_type_dropdown'].value = env_type
            
            if 'auto_mount_drive_checkbox' in self._ui_components:
                auto_mount = env_config.get('auto_mount_drive', True)
                self._ui_components['auto_mount_drive_checkbox'].value = auto_mount
            
            if 'project_name_text' in self._ui_components:
                project_name = env_config.get('project_name', 'SmartCash')
                self._ui_components['project_name_text'].value = project_name
            
            if 'gpu_enabled_checkbox' in self._ui_components:
                gpu_enabled = env_config.get('gpu_enabled', False)
                self._ui_components['gpu_enabled_checkbox'].value = gpu_enabled
            
            if 'gpu_type_dropdown' in self._ui_components:
                gpu_type = env_config.get('gpu_type', 'none')
                self._ui_components['gpu_type_dropdown'].value = gpu_type
            
            # Update setup components
            if 'setup_stages_select' in self._ui_components:
                stages = setup_config.get('stages', [])
                self._ui_components['setup_stages_select'].value = tuple(stages)
            
            if 'auto_start_checkbox' in self._ui_components:
                auto_start = setup_config.get('auto_start', False)
                self._ui_components['auto_start_checkbox'].value = auto_start
            
            if 'stop_on_error_checkbox' in self._ui_components:
                stop_on_error = setup_config.get('stop_on_error', True)
                self._ui_components['stop_on_error_checkbox'].value = stop_on_error
            
            if 'max_retries_int' in self._ui_components:
                max_retries = setup_config.get('max_retries', 3)
                self._ui_components['max_retries_int'].value = max_retries
            
            # Update UI options
            if 'show_advanced_checkbox' in self._ui_components:
                show_advanced = ui_config.get('show_advanced_options', False)
                self._ui_components['show_advanced_checkbox'].value = show_advanced
            
            # Update status summary if available
            if 'status_summary' in self._ui_components:
                self._update_status_summary(config)
            
            self.logger.info("✅ UI successfully updated from config")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating UI from config: {e}")
    
    def _update_status_summary(self, config: Dict[str, Any]) -> None:
        """Update status summary widget."""
        try:
            from ..components.setup_summary import update_setup_summary
            update_setup_summary(self._ui_components['status_summary'], config)
        except ImportError:
            # Fallback if setup_summary module is not available
            self.logger.warning("⚠️ Setup summary module not available")
    
    def sync_config_with_ui(self) -> None:
        """Sync configuration with UI state."""
        try:
            # Extract current config from UI
            current_config = self.extract_config_from_ui()
            
            # Update config handler
            self.config_handler.update_config(current_config)
            
            self.logger.info("✅ Config successfully synced with UI")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing config with UI: {e}")
    
    def sync_ui_with_config(self) -> None:
        """Sync UI with configuration."""
        try:
            # Get current config
            current_config = self.config_handler.get_config()
            
            # Update UI
            self.update_ui_from_config(current_config)
            
            self.logger.info("✅ UI successfully synced with config")
            
        except Exception as e:
            self.logger.error(f"❌ Error syncing UI with config: {e}")
    
    def initialize(self) -> None:
        """Initialize the UI handler for colab module."""
        self.logger.info("🚀 Initializing Colab UI Handler")
        # Perform any necessary initialization for the colab UI handler
        self.logger.info("✅ Colab UI Handler initialized successfully")
    
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