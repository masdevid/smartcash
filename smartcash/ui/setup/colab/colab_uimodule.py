"""
File: smartcash/ui/setup/colab/colab_uimodule.py
Description: Colab Module implementation using BaseUIModule pattern with operation checklist compliance.
"""

from typing import Dict, Any, Optional
import os
import sys

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs
from smartcash.ui.logger import get_module_logger

# Environment management imports
from smartcash.common.environment import get_environment_manager, EnvironmentManager
from smartcash.common.constants.paths import get_paths_for_environment

# Colab module imports
from smartcash.ui.setup.colab.components.colab_ui import create_colab_ui
from smartcash.ui.setup.colab.configs.colab_config_handler import ColabConfigHandler
from smartcash.ui.setup.colab.configs.colab_defaults import get_default_colab_config
from smartcash.ui.setup.colab.operations.colab_factory import (
    ColabOperationFactory,
    init_environment,
    mount_drive,
    create_symlinks,
    create_folders,
    sync_config,
    setup_environment,
    verify_setup,
    execute_full_setup,
    detect_environment
)


class ColabUIModule(BaseUIModule):
    """
    Colab Module implementation using BaseUIModule pattern.
    
    Features:
    - 🌟 Google Colab environment detection and setup
    - 🔧 Sequential operations: INIT → DRIVE → SYMLINK → FOLDERS → CONFIG → ENV → VERIFY
    - 📊 Real-time progress tracking and UI-integrated logging
    - 💾 No-persistence configuration (Colab-specific requirement)
    - 🔄 Enhanced factory-based initialization functions
    - ✅ Full compliance with OPERATION_CHECKLISTS.md requirements
    """
    
    def __init__(self, **kwargs):
        """Initialize Colab UI module."""
        super().__init__(
            module_name='colab',
            parent_module='setup',
            **kwargs
        )
        
        # Set required components for validation (Operation Checklist 1.2)
        self._required_components = [
            'main_container',
            'header_container', 
            'action_container',
            'operation_container'
        ]
        
        # Colab-specific attributes
        self._environment_manager: Optional[EnvironmentManager] = None
        self._environment_detected = False
        self._is_colab_environment = False
        self._environment_paths = {}
        
        # Initialize log buffer for pre-operation-container logs
        self._log_buffer = []
        
        # Initialize operation factory
        self._operation_factory: Optional[ColabOperationFactory] = None
        
        self.logger.debug("✅ ColabUIModule initialized")
    
    def _initialize_progress_display(self) -> None:
        """Initialize progress display components."""
        try:
            # Ensure progress visibility for colab operations
            if hasattr(self, '_ensure_progress_visibility'):
                self._ensure_progress_visibility()
            
            # Initialize progress bars if needed
            if hasattr(self, '_ui_components') and self._ui_components:
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'initialize'):
                    progress_tracker.initialize()
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to initialize progress display: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Colab module (BaseUIModule requirement)."""
        return get_default_colab_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> ColabConfigHandler:
        """Create config handler instance for Colab module (BaseUIModule requirement)."""
        handler = ColabConfigHandler(logger=self.logger)
        if config:
            handler.update_config(config)
        return handler
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for Colab module (BaseUIModule requirement)."""
        try:
            self.logger.debug("Creating Colab UI components...")
            ui_components = create_colab_ui(config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            self.logger.debug(f"✅ Created {len(ui_components)} UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Colab module with environment detection.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Detect environment first (Operation Checklist 8.3)
            self._detect_environment()
            
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = super().initialize()
            
            if success:
                # Setup operation container reference for backwards compatibility
                self._setup_operation_container()
                
                # Initialize operation factory
                self._initialize_operation_factory()
                
                # Setup button phases for sequential operations
                self._setup_button_phases()
                
                # Re-setup button handlers after UI components are fully initialized
                self._setup_colab_button_handlers()
                
                # Flush any buffered logs to operation container
                self._flush_log_buffer()
                
                # Log environment detection results (Operation Checklist 3.2)
                env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
                self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
                
                if self._is_colab_environment:
                    self.log("✅ Berjalan di Google Colab - semua fitur tersedia", 'success')
                else:
                    self.log("⚠️ Tidak berjalan di Google Colab - beberapa fitur mungkin terbatas", 'warning')
                
                # Update status panel (Operation Checklist 7.1) - use direct log to avoid console output
                self.log("📊 Status: Siap untuk pengaturan lingkungan Colab", 'info')
                self.update_operation_status("Siap untuk pengaturan lingkungan Colab", "info")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Colab module: {e}")
            return False
    
    def _setup_operation_container(self) -> None:
        """Setup operation container reference for backwards compatibility."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                self._operation_container = operation_container
                
                # Set up proper logging bridge for the operation container
                if operation_container:
                    # If it's a dict, extract the actual container object
                    if isinstance(operation_container, dict):
                        container_obj = operation_container.get('container')
                        if container_obj:
                            self._operation_container = container_obj
                            self.logger.debug("✅ Operation container reference set up from dict")
                        else:
                            self.logger.debug("✅ Operation container reference set up as dict")
                    else:
                        self.logger.debug("✅ Operation container reference set up as object")
                else:
                    self.logger.warning("⚠️ Operation container not found in UI components")
        except Exception as e:
            self.logger.error(f"Failed to setup operation container: {e}")
    
    def _setup_colab_button_handlers(self) -> None:
        """Setup Colab-specific button handlers after UI components are initialized."""
        try:
            if not hasattr(self, '_ui_components') or not self._ui_components:
                self.logger.warning("⚠️ UI components not available for button handler setup")
                return
            
            # Get the setup button from UI components 
            setup_button = None
            button_candidates = [
                'setup_button',
                'primary_button', 
                'colab_setup'
            ]
            
            for button_id in button_candidates:
                button = self._ui_components.get(button_id)
                if button and hasattr(button, 'on_click'):
                    setup_button = button
                    self.logger.debug(f"✅ Found setup button: {button_id}")
                    break
            
            if not setup_button:
                self.logger.warning("⚠️ No setup button found in UI components")
                # Try to find it in action container
                action_container = self._ui_components.get('action_container')
                if action_container and isinstance(action_container, dict):
                    buttons = action_container.get('buttons', {})
                    setup_button = buttons.get('colab_setup')
                    if setup_button:
                        self.logger.debug("✅ Found setup button in action container")
            
            if setup_button and hasattr(setup_button, 'on_click'):
                # Register the handler with debug wrapper
                def debug_handler(button):
                    self.logger.info("🔥 BUTTON CLICKED! Processing setup button...")
                    try:
                        result = self._handle_setup_button(button)
                        return result
                    except Exception as e:
                        self.logger.error(f"❌ Button handler failed: {e}", exc_info=True)
                        raise
                
                # Use on_click to register the handler
                setup_button.on_click(debug_handler)
                self.logger.info("✅ Colab setup button handler registered successfully with debug wrapper")
                
                # Verify registration by checking if we can call the handler
                try:
                    # Test if the handler was registered by checking widget attributes
                    if hasattr(setup_button, '_click_handlers'):
                        # Handle different types of click handlers
                        click_handlers = setup_button._click_handlers
                        if hasattr(click_handlers, '__len__'):
                            handler_count = len(click_handlers)
                            self.logger.info(f"📊 Button now has {handler_count} click handlers")
                        elif hasattr(click_handlers, '_callbacks'):
                            # CallbackDispatcher case
                            callback_count = len(click_handlers._callbacks) if hasattr(click_handlers._callbacks, '__len__') else 'unknown'
                            self.logger.info(f"📊 Button has CallbackDispatcher with {callback_count} callbacks")
                        else:
                            self.logger.info(f"📊 Button has click handlers of type: {type(click_handlers)}")
                    elif hasattr(setup_button, '_model_id'):
                        self.logger.info(f"📊 Button widget model ID: {setup_button._model_id}")
                    else:
                        self.logger.info("📊 Button widget registered (no _click_handlers attribute)")
                except Exception as e:
                    self.logger.debug(f"Could not check handler registration: {e}")
                
                # Update button state
                if hasattr(setup_button, 'description'):
                    self.logger.debug(f"Setup button description: {setup_button.description}")
                    
            else:
                self.logger.error("❌ Setup button not found or missing on_click method")
                
                # Debug: List all available UI components
                ui_keys = list(self._ui_components.keys())
                self.logger.debug(f"Available UI components: {ui_keys}")
                
                # Check for buttons with on_click methods
                buttons_with_onclick = []
                for key, component in self._ui_components.items():
                    if hasattr(component, 'on_click'):
                        buttons_with_onclick.append(key)
                
                if buttons_with_onclick:
                    self.logger.debug(f"Components with on_click: {buttons_with_onclick}")
                else:
                    self.logger.warning("No components with on_click method found")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup Colab button handlers: {e}", exc_info=True)
    
    def _initialize_operation_factory(self) -> None:
        """Initialize the operation factory with current config."""
        try:
            config = self.get_current_config() if hasattr(self, 'get_current_config') else {}
            operation_container = getattr(self, '_operation_container', None)
            self._operation_factory = ColabOperationFactory(config, operation_container)
            self.logger.debug("✅ Operation factory initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize operation factory: {e}")
    
    def _detect_environment(self) -> None:
        """Detect environment using standardized EnvironmentManager (Operation Checklist 8.3)."""
        try:
            # Use standardized environment manager
            self._environment_manager = get_environment_manager(logger=self.logger)
            self._is_colab_environment = self._environment_manager.is_colab
            self._environment_detected = True
            
            # Get appropriate paths for current environment
            self._environment_paths = get_paths_for_environment(
                is_colab=self._is_colab_environment,
                is_drive_mounted=self._environment_manager.is_drive_mounted if self._is_colab_environment else False
            )
            
            env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
            self.logger.debug(f"✅ Environment detected: {env_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to detect environment: {e}")
            # Fallback to simple detection
            env_result = detect_environment()
            self._is_colab_environment = env_result.get('is_colab', False)
            self._environment_detected = True
    
    def _setup_button_phases(self) -> None:
        """Setup button phases for sequential colab operations."""
        try:
            # Define sequential phases for colab setup matching constants
            phases = [
                {'id': 'init', 'text': '🔧 Inisialisasi', 'description': 'Menginisialisasi environment'},
                {'id': 'drive', 'text': '📁 Mount Drive', 'description': 'Memasang Google Drive'},
                {'id': 'symlink', 'text': '🔗 Symlink', 'description': 'Membuat symbolic links'},
                {'id': 'folders', 'text': '📂 Folders', 'description': 'Membuat direktori'},
                {'id': 'config', 'text': '⚙️ Config', 'description': 'Sinkronisasi konfigurasi'},
                {'id': 'env', 'text': '🌍 Environment', 'description': 'Setup environment'},
                {'id': 'verify', 'text': '🔍 Verifikasi', 'description': 'Memverifikasi pengaturan'},
                {'id': 'complete', 'text': '🎉 Selesai', 'description': 'Setup selesai'}
            ]
            
            # Set phases to UI components if available
            if hasattr(self, '_ui_components') and self._ui_components:
                # Check if _ui_components is a dictionary and has set_phases method
                if isinstance(self._ui_components, dict) and 'set_phases' in self._ui_components:
                    self._ui_components['set_phases'](phases)
                    self.logger.debug("✅ Button phases configured")
                elif hasattr(self, 'set_phases'):
                    self.set_phases(phases)
                    self.logger.debug("✅ Button phases configured via method")
                else:
                    self.logger.debug("⚠️ Button phases not configured - set_phases method not found")
                    
        except Exception as e:
            self.logger.error(f"Failed to setup button phases: {e}")
    
    def _register_default_operations(self) -> None:
        """Register default operations for Colab module (Operation Checklist 9.1)."""
        # Call parent method first
        super()._register_default_operations()
        
        # Note: Colab operations are now handled directly through the primary button
        # with phase states, eliminating the need for separate operation registrations
        
        # Note: Dynamic button handler registration is now handled by BaseUIModule
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Colab module-specific button handlers."""
        # Call parent method to get base handlers (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add Colab-specific handlers - only for buttons that actually exist in UI
        colab_handlers = {
            'setup_button': self._handle_setup_button,
            'primary_button': self._handle_setup_button,
            'colab_setup': self._handle_setup_button
        }
        
        handlers.update(colab_handlers)
        return handlers
    
    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container."""
        try:
            if not self._log_buffer:
                return
                
            # Ensure operation container is available
            if not hasattr(self, '_operation_container') or not self._operation_container:
                self.logger.warning("⚠️ Operation container not available for log buffer flush")
                return
                
            # Display all buffered logs to operation container
            for log_entry in self._log_buffer:
                message, level = log_entry
                self.log(message, level)
            
            # Clear the buffer
            self._log_buffer.clear()
            self.logger.debug(f"✅ Flushed {len(self._log_buffer)} buffered logs to operation container")
            
        except Exception as e:
            self.logger.debug(f"Failed to flush log buffer: {e}")
    
    # ==================== HELPER METHODS ====================
    
    def _execute_factory_operation(self, factory_func, operation_name: str, start_message: str, 
                                   success_message: str, error_message: str, validation_message: str = None,
                                   validation_condition: bool = True, post_success_callback = None, button=None) -> Dict[str, Any]:
        """Execute factory operation with standardized pattern."""
        def validate_operation():
            if validation_message and not validation_condition:
                return {'valid': False, 'message': validation_message}
            if validation_message and not self._environment_detected:
                return {'valid': False, 'message': validation_message}
            return {'valid': True}
        
        def execute_operation():
            self.log(start_message, 'info')
            
            config = self.get_current_config() if hasattr(self, 'get_current_config') else {}
            result = factory_func(
                config=config,
                operation_container=getattr(self, '_operation_container', None),
                progress_callback=lambda p, m: self.update_progress(p, m)
            )
            
            if result.get('success'):
                self.log(f"✅ {operation_name} berhasil diselesaikan", 'success')
                if post_success_callback:
                    post_success_callback(result)
            
            return result
        
        return self._execute_operation_with_wrapper(
            operation_name=operation_name,
            operation_func=execute_operation,
            button=button,
            validation_func=validate_operation,
            success_message=success_message,
            error_message=error_message
        )
    
    # ==================== OPERATION HANDLERS ====================
    
    def _handle_setup_button(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle setup button click - execute full setup in one click."""
        try:
            self.log("🚀 Memulai setup lengkap lingkungan Colab...", 'info')
            
            # Execute all phases sequentially
            phases = [
                ('init', self._execute_init_phase),
                ('drive', self._execute_drive_phase),
                ('symlink', self._execute_symlink_phase),
                ('folders', self._execute_folders_phase),
                ('config', self._execute_config_phase),
                ('env', self._execute_env_phase),
                ('verify', self._execute_verify_phase)
            ]
            
            for phase_name, phase_func in phases:
                self.log(f"🔄 Menjalankan fase: {phase_name}", 'info')
                result = phase_func(button)
                
                if not result.get('success'):
                    error_msg = f"❌ Fase {phase_name} gagal: {result.get('message', 'Unknown error')}"
                    self.log(error_msg, 'error')
                    return {'success': False, 'message': error_msg}
                
                self.log(f"✅ Fase {phase_name} berhasil", 'success')
            
            # All phases completed successfully
            self._set_phase('complete')
            success_msg = "🎉 Setup Colab lengkap berhasil diselesaikan!"
            self.log(success_msg, 'success')
            return {'success': True, 'message': success_msg}
                
        except Exception as e:
            error_msg = f"Error in setup button handler: {e}"
            self.logger.error(error_msg)
            self._set_phase('error')
            return {'success': False, 'message': error_msg}
    
    def _get_current_phase(self) -> str:
        """Get current phase from button state."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                setup_button = self._ui_components.get('setup_button')
                if setup_button and hasattr(setup_button, 'current_phase'):
                    return setup_button.current_phase
            return 'init'  # Default starting phase
        except Exception:
            return 'init'
    
    def _set_phase(self, phase: str) -> None:
        """Set the current phase and update button state."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                set_phase_func = self._ui_components.get('set_phase')
                if set_phase_func and callable(set_phase_func):
                    set_phase_func(phase)
                    self.logger.debug(f"✅ Phase set to: {phase}")
        except Exception as e:
            self.logger.error(f"Failed to set phase to {phase}: {e}")
    
    def _advance_to_next_phase(self, next_phase: str) -> None:
        """Advance to the next phase."""
        self._set_phase(next_phase)
        self.log(f"📍 Beralih ke fase: {next_phase}", 'info')
    
    # ==================== PHASE-SPECIFIC EXECUTION METHODS ====================
    
    def _execute_phase(self, phase_config: Dict[str, Any], button=None) -> Dict[str, Any]:
        """Execute any phase using configuration."""
        phase_name = phase_config['name']
        factory_func = phase_config['factory_func']
        next_phase = phase_config.get('next_phase')
        
        def validate_phase():
            return {'valid': True}
        
        def execute_phase():
            self.log(phase_config['start_message'], 'info')
            self._set_phase(phase_name)
            
            config = self.get_current_config() if hasattr(self, 'get_current_config') else {}
            result = factory_func(
                config=config,
                operation_container=getattr(self, '_operation_container', None),
                progress_callback=lambda p, m: self.update_progress(p, m)
            )
            
            if result.get('success') and next_phase:
                self.log(phase_config['success_log'], 'success')
                self._advance_to_next_phase(next_phase)
            
            return result
        
        return self._execute_operation_with_wrapper(
            operation_name=phase_config['operation_name'],
            operation_func=execute_phase,
            button=button,
            validation_func=validate_phase,
            success_message=phase_config['success_message'],
            error_message=phase_config['error_message']
        )
    
    def _execute_init_phase(self, button=None) -> Dict[str, Any]:
        """Execute initialization phase."""
        return self._execute_phase({
            'name': 'init',
            'factory_func': init_environment,
            'next_phase': 'drive',
            'start_message': "🔧 Fase Inisialisasi: Mempersiapkan lingkungan...",
            'success_log': "✅ Inisialisasi berhasil",
            'operation_name': "Inisialisasi Lingkungan",
            'success_message': "Inisialisasi berhasil - melanjutkan ke fase Drive",
            'error_message': "Kesalahan inisialisasi"
        }, button)
    
    def _execute_drive_phase(self, button=None) -> Dict[str, Any]:
        """Execute Google Drive mounting phase."""
        return self._execute_phase({
            'name': 'drive',
            'factory_func': mount_drive,
            'next_phase': 'symlink',
            'start_message': "📁 Fase Drive: Mounting Google Drive...",
            'success_log': "✅ Drive berhasil dipasang",
            'operation_name': "Mount Google Drive",
            'success_message': "Drive mounted - melanjutkan ke fase Symlink",
            'error_message': "Kesalahan mount drive"
        }, button)
    
    def _execute_symlink_phase(self, button=None) -> Dict[str, Any]:
        """Execute symlink creation phase."""
        return self._execute_phase({
            'name': 'symlink',
            'factory_func': create_symlinks,
            'next_phase': 'folders',
            'start_message': "🔗 Fase Symlink: Menyiapkan symbolic links...",
            'success_log': "✅ Symlink berhasil dibuat",
            'operation_name': "Membuat Symlinks",
            'success_message': "Symlinks dibuat - melanjutkan ke fase Folders",
            'error_message': "Kesalahan pembuatan symlink"
        }, button)
    
    def _execute_folders_phase(self, button=None) -> Dict[str, Any]:
        """Execute folder creation phase."""
        return self._execute_phase({
            'name': 'folders',
            'factory_func': create_folders,
            'next_phase': 'config',
            'start_message': "📂 Fase Folders: Membuat direktori yang diperlukan...",
            'success_log': "✅ Direktori berhasil dibuat",
            'operation_name': "Membuat Direktori",
            'success_message': "Direktori dibuat - melanjutkan ke fase Config",
            'error_message': "Kesalahan pembuatan direktori"
        }, button)
    
    def _execute_config_phase(self, button=None) -> Dict[str, Any]:
        """Execute configuration phase."""
        return self._execute_phase({
            'name': 'config',
            'factory_func': sync_config,
            'next_phase': 'env',
            'start_message': "⚙️ Fase Config: Menyiapkan konfigurasi...",
            'success_log': "✅ Konfigurasi berhasil disiapkan",
            'operation_name': "Setup Konfigurasi",
            'success_message': "Konfigurasi selesai - melanjutkan ke fase Environment",
            'error_message': "Kesalahan setup konfigurasi"
        }, button)
    
    def _execute_env_phase(self, button=None) -> Dict[str, Any]:
        """Execute environment setup phase."""
        return self._execute_phase({
            'name': 'env',
            'factory_func': setup_environment,
            'next_phase': 'verify',
            'start_message': "🌍 Fase Environment: Menyiapkan environment variables...",
            'success_log': "✅ Environment berhasil disiapkan",
            'operation_name': "Setup Environment",
            'success_message': "Environment selesai - melanjutkan ke fase Verifikasi",
            'error_message': "Kesalahan setup environment"
        }, button)
    
    def _execute_verify_phase(self, button=None) -> Dict[str, Any]:
        """Execute verification phase."""
        return self._execute_phase({
            'name': 'verify',
            'factory_func': verify_setup,
            'next_phase': 'complete',
            'start_message': "🔍 Fase Verify: Memverifikasi setup lengkap...",
            'success_log': "🎉 Semua fase setup Colab telah selesai!",
            'operation_name': "Verifikasi Setup",
            'success_message': "🎉 Setup Colab selesai sempurna!",
            'error_message': "Kesalahan verifikasi setup"
        }, button)
    
    def _handle_full_setup(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle full Colab setup operation using factory pattern."""
        return self._execute_factory_operation(
            factory_func=execute_full_setup,
            operation_name="Pengaturan Lengkap Colab",
            start_message="🚀 Memulai pengaturan lengkap lingkungan Colab...",
            success_message="Pengaturan lengkap Colab berhasil diselesaikan",
            error_message="Kesalahan pengaturan lengkap Colab",
            validation_message="Lingkungan belum terdeteksi, silakan coba lagi",
            button=button
        )
    
    def _handle_init_environment(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle environment initialization using factory pattern."""
        return self._execute_factory_operation(
            factory_func=init_environment,
            operation_name="Inisialisasi Lingkungan",
            start_message="🔧 Menginisialisasi lingkungan...",
            success_message="Inisialisasi lingkungan berhasil diselesaikan",
            error_message="Kesalahan inisialisasi lingkungan",
            validation_message="Lingkungan belum terdeteksi, silakan coba lagi",
            button=button
        )
    
    def _handle_mount_drive(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle Google Drive mounting using factory pattern."""
        def post_success_callback(result):  # noqa: ARG001
            if self._environment_manager:
                self._environment_paths = get_paths_for_environment(is_colab=True, is_drive_mounted=True)
        
        return self._execute_factory_operation(
            factory_func=mount_drive,
            operation_name="Mount Google Drive",
            start_message="📁 Memasang Google Drive...",
            success_message="Google Drive berhasil dipasang",
            error_message="Kesalahan mount Google Drive",
            validation_message="Mount Google Drive hanya tersedia di lingkungan Colab",
            validation_condition=self._is_colab_environment,
            post_success_callback=post_success_callback,
            button=button
        )
    
    def _handle_verify_setup(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle setup verification using factory pattern."""
        return self._execute_factory_operation(
            factory_func=verify_setup,
            operation_name="Verifikasi Pengaturan",
            start_message="🔍 Memverifikasi pengaturan lingkungan...",
            success_message="Verifikasi pengaturan berhasil diselesaikan",
            error_message="Kesalahan verifikasi pengaturan",
            validation_message="Lingkungan belum terdeteksi, silakan coba lagi",
            button=button
        )
    
    def _handle_detect_environment(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle environment detection using factory pattern."""
        def custom_detect():
            env_result = detect_environment()
            self._is_colab_environment = env_result.get('is_colab', False)
            self._environment_detected = True
            
            import sys
            env_info = {
                'is_colab': self._is_colab_environment,
                'runtime_type': env_result.get('runtime_type', 'unknown'),
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            }
            
            env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
            self.log(f"🌍 Lingkungan: {env_type}", 'info')
            self.log(f"🐍 Python: {env_info['python_version']}", 'info')
            self.log(f"💻 Platform: {env_info['platform']}", 'info')
            
            return {'success': True, 'message': f'Lingkungan terdeteksi: {env_type}', 'environment': env_info}
        
        return self._execute_operation_with_wrapper(
            operation_name="Deteksi Lingkungan",
            operation_func=custom_detect,
            button=button,
            validation_func=lambda: {'valid': True},
            success_message="Deteksi lingkungan berhasil diselesaikan",
            error_message="Kesalahan deteksi lingkungan"
        )
    
    # ==================== COLAB-SPECIFIC METHODS ====================
    
    def get_colab_status(self) -> Dict[str, Any]:
        """
        Get current Colab environment status (Operation Checklist 9.2).
        
        Returns:
            Status information dictionary
        """
        try:
            status = {
                'initialized': self._is_initialized,
                'module_name': self.module_name,
                'environment_detected': self._environment_detected,
                'is_colab': self._is_colab_environment,
                'config_loaded': self._config_handler is not None,
                'ui_created': bool(self._ui_components)
            }
            
            # Add environment-specific information
            if self._is_colab_environment and self._environment_manager:
                try:
                    # Use EnvironmentManager for drive status
                    status['drive_mounted'] = self._environment_manager.is_drive_mounted
                    status['drive_path'] = str(self._environment_manager.drive_path) if self._environment_manager.drive_path else None
                    status['base_directory'] = str(self._environment_manager.base_dir)
                    status['data_directory'] = str(self._environment_manager.get_dataset_path())
                except Exception as e:
                    self.logger.warning(f"Failed to get drive status: {e}")
                    status['drive_mounted'] = False
            
            return status
            
        except Exception as e:
            return {'error': f'Pemeriksaan status gagal: {str(e)}'}
    
    def is_colab_environment(self) -> bool:
        """Check if running in Google Colab environment."""
        return self._is_colab_environment
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get detailed environment information."""
        # Use factory method for environment detection
        env_result = detect_environment()
        
        base_info = {
            'is_colab': env_result.get('is_colab', self._is_colab_environment),
            'runtime_type': env_result.get('runtime_type', 'unknown'),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'working_directory': os.getcwd(),
            'environment_detected': self._environment_detected,
            'paths': self._environment_paths
        }
        
        # Add EnvironmentManager system info if available
        if self._environment_manager:
            try:
                system_info = self._environment_manager.get_system_info()
                base_info.update({
                    'base_directory': system_info.get('base_directory'),
                    'data_directory': system_info.get('data_directory'),
                    'drive_mounted': system_info.get('drive_mounted'),
                    'drive_path': system_info.get('drive_path'),
                    'cuda_available': system_info.get('cuda_available'),
                    'total_memory_gb': system_info.get('total_memory_gb'),
                    'available_memory_gb': system_info.get('available_memory_gb')
                })
            except Exception as e:
                self.logger.warning(f"Failed to get system info from EnvironmentManager: {e}")
        
        return base_info


# ==================== FACTORY FUNCTIONS ====================

# Create standardized display function using enhanced factory
from smartcash.ui.core.enhanced_ui_module_factory import EnhancedUIModuleFactory

def initialize_colab_ui(config: Optional[Dict[str, Any]] = None, 
                       show_display: bool = True, 
                       **kwargs) -> Optional[Dict[str, Any]]:
    """Initialize and optionally display the Colab UI module."""
    # Filter out conflicting display-related parameters from kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['display', 'show_display']}
    
    # Determine final display value - prioritize explicit 'display' parameter
    if 'display' in kwargs:
        final_display = kwargs['display']
    else:
        final_display = show_display
    
    return EnhancedUIModuleFactory.create_and_display(
        module_class=ColabUIModule,
        config=config,
        display=final_display,
        **filtered_kwargs
    )

def get_colab_components(config: Optional[Dict[str, Any]] = None, 
                        **kwargs) -> Optional[Dict[str, Any]]:
    """Get Colab UI components without displaying."""
    return initialize_colab_ui(config=config, show_display=False, **kwargs)


# ==================== CONVENIENCE FUNCTIONS ====================

# Global module instance for singleton pattern
_colab_module_instance: Optional[ColabUIModule] = None

def create_colab_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> ColabUIModule:
    """
    Create a new Colab UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        ColabUIModule instance
    """
    module = ColabUIModule()
    
    if auto_initialize:
        module.initialize(config, **kwargs)
    
    return module


def get_colab_uimodule() -> Optional[ColabUIModule]:
    """Get the current Colab UIModule instance."""
    global _colab_module_instance
    return _colab_module_instance


def reset_colab_uimodule() -> None:
    """Reset the global Colab UIModule instance."""
    global _colab_module_instance
    if _colab_module_instance:
        try:
            _colab_module_instance.cleanup()
        except:
            pass
    _colab_module_instance = None

def detect_colab_environment() -> Dict[str, Any]:
    """Detect if running in Google Colab environment."""
    try:
        import google.colab  # noqa: F401
        return {"is_colab": True, "runtime_type": "colab"}
    except ImportError:
        return {"is_colab": False, "runtime_type": "local"}


def mount_google_drive(drive_path: str = "/content/drive") -> Dict[str, Any]:
    """Mount Google Drive in Colab environment."""
    try:
        from google.colab import drive  # noqa: F401
        drive.mount(drive_path)
        return {"success": True, "path": drive_path}
    except ImportError:
        return {"success": False, "error": "Not running in Google Colab"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ==================== SHARED METHODS REGISTRATION ====================

def register_colab_shared_methods() -> None:
    """Register shared methods for Colab module (Operation Checklist 9.1)."""
    try:
        from smartcash.ui.core.ui_module import SharedMethodRegistry
        
        # Register Colab-specific shared methods
        SharedMethodRegistry.register_method(
            'colab.detect_environment',
            detect_colab_environment,
            description='Detect Colab environment'
        )
        
        SharedMethodRegistry.register_method(
            'colab.mount_drive',
            mount_google_drive,
            description='Mount Google Drive'
        )
        
        SharedMethodRegistry.register_method(
            'colab.get_status',
            lambda: create_colab_uimodule().get_colab_status(),
            description='Get Colab environment status'
        )
        
        logger = get_module_logger("smartcash.ui.setup.colab.shared")
        logger.debug("📋 Registered Colab shared methods")
        
    except Exception as e:
        # Log error but don't raise to avoid breaking module loading
        logger = get_module_logger("smartcash.ui.setup.colab.shared")
        logger.error(f"Failed to register shared methods: {e}")

# Auto-register when module is imported
try:
    register_colab_shared_methods()
except Exception as e:
    # Log but continue - registration is optional
    import logging
    logging.getLogger(__name__).warning(f"Module registration failed: {e}")