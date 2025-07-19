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
        
        # Setup completion tracking
        self._setup_complete = False
        self._setup_flag_file = self._get_setup_flag_file_path()
        
        self.logger.debug("✅ ColabUIModule initialized")
    
    
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
                
                # Button handlers are already set up by the base class
                
                # Flush any buffered logs to operation container
                self._flush_log_buffer()
                
                # Check setup completion status and update UI accordingly
                self._check_setup_completion_status()
                
                # Log environment detection results (Operation Checklist 3.2)
                env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
                self.log(f"🌍 Lingkungan terdeteksi: {env_type}", 'info')
                
                if self._is_colab_environment:
                    self.log("✅ Berjalan di Google Colab - semua fitur tersedia", 'success')
                    self.log("📊 Modul Colab siap digunakan", 'info')
                else:
                    self.log("⚠️ Tidak berjalan di Google Colab - beberapa fitur mungkin terbatas", 'warning')
                    self.log("📊 Modul Colab berjalan dalam mode terbatas", 'info')
            
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
                {'id': 'complete', 'text': '🎉 Selesai', 'description': 'Setup selesai'},
                {'id': 'error', 'text': '❌ Error', 'description': 'Terjadi kesalahan'}
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
        
        # Add colab-specific button handlers
        colab_handlers = {
            'colab_setup': self._handle_setup_button,
            'reset_setup': self.reset_setup_status  # For testing/reset purposes
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

    def _get_setup_flag_file_path(self) -> str:
        """Get the path to the setup completion flag file (local to Colab, not saved to drive)."""
        try:
            # Use /tmp for temporary files that don't persist across sessions
            # This ensures the flag is local to current Colab session only
            return "/tmp/.smartcash_setup_complete"
        except Exception:
            return ".smartcash_setup_complete"  # Fallback to local directory

    def _check_setup_completion_status(self) -> None:
        """Check if setup has been completed and update UI accordingly."""
        try:
            import os
            
            # Check if setup completion flag file exists
            if os.path.exists(self._setup_flag_file):
                self._setup_complete = True
                self.log("✅ Setup sudah selesai sebelumnya", 'success')
                self.log("ℹ️ Button setup telah dinonaktifkan", 'info')
                self._disable_setup_button()
            else:
                self._setup_complete = False
                self.log("🔧 Setup belum selesai - siap untuk memulai", 'info')
                
        except Exception as e:
            self.logger.error(f"Failed to check setup completion status: {e}")
            self._setup_complete = False

    def _disable_setup_button(self) -> None:
        """Disable setup button and update its appearance."""
        try:
            # Get action container to access buttons
            action_container = self.get_component('action_container')
            if action_container and isinstance(action_container, dict):
                buttons = action_container.get('buttons', {})
                setup_button = buttons.get('colab_setup')
                
                if setup_button:
                    setup_button.disabled = True
                    setup_button.description = "✅ Setup Complete"
                    setup_button.tooltip = "Setup telah selesai sebelumnya. File flag ditemukan di /tmp/"
                    setup_button.button_style = 'success'
                    self.logger.debug("✅ Setup button disabled - setup already complete")
                    
        except Exception as e:
            self.logger.error(f"Failed to disable setup button: {e}")

    def _enable_setup_button(self) -> None:
        """Enable setup button and reset its appearance."""
        try:
            # Get action container to access buttons
            action_container = self.get_component('action_container')
            if action_container and isinstance(action_container, dict):
                buttons = action_container.get('buttons', {})
                setup_button = buttons.get('colab_setup')
                
                if setup_button:
                    setup_button.disabled = False
                    setup_button.description = "🚀 Setup Environment"
                    setup_button.tooltip = "Jalankan setup lengkap lingkungan Colab"
                    setup_button.button_style = 'primary'
                    self.logger.debug("✅ Setup button enabled")
                    
        except Exception as e:
            self.logger.error(f"Failed to enable setup button: {e}")

    def _create_setup_completion_flag(self) -> None:
        """Create setup completion flag file."""
        try:
            import os
            from datetime import datetime
            
            # Create flag file with timestamp
            with open(self._setup_flag_file, 'w') as f:
                f.write(f"SmartCash setup completed at: {datetime.now().isoformat()}\n")
                f.write(f"Environment: {'colab' if self._is_colab_environment else 'local'}\n")
                f.write("This file indicates that setup has been completed successfully.\n")
                
            self._setup_complete = True
            self.log("✅ File flag setup completion dibuat", 'success')
            self.log(f"📁 Flag file: {self._setup_flag_file}", 'info')
            
        except Exception as e:
            self.logger.error(f"Failed to create setup completion flag: {e}")

    def _remove_setup_completion_flag(self) -> None:
        """Remove setup completion flag file (for testing/reset purposes)."""
        try:
            import os
            
            if os.path.exists(self._setup_flag_file):
                os.remove(self._setup_flag_file)
                self._setup_complete = False
                self.log("🗑️ File flag setup completion dihapus", 'info')
                self._enable_setup_button()
            else:
                self.log("ℹ️ File flag setup completion tidak ditemukan", 'info')
                
        except Exception as e:
            self.logger.error(f"Failed to remove setup completion flag: {e}")
    
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
            else:
                # Log the full error details including traceback if available
                error_details = []
                if 'error' in result:
                    error_details.append(f"Error: {result['error']}")
                if 'traceback' in result:
                    error_details.append(f"\nTraceback:\n{result['traceback']}")
                if error_details:
                    self.logger.error("\n".join(error_details))
        
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
            # Check if setup is already complete
            if self._setup_complete:
                warning_msg = "⚠️ Setup sudah selesai sebelumnya. Reset jika ingin menjalankan ulang."
                self.log(warning_msg, 'warning')
                return {'success': False, 'message': warning_msg}
            
            # Disable setup button during operation
            self._disable_setup_button_during_operation()
            
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
                    self._set_phase('error')
                    # Re-enable button on failure
                    self._enable_setup_button()
                    return {'success': False, 'message': error_msg}
                
                self.log(f"✅ Fase {phase_name} berhasil", 'success')
            
            # All phases completed successfully
            self._set_phase('complete')
            
            # Create setup completion flag file
            self._create_setup_completion_flag()
            
            # Disable setup button permanently (until reset)
            self._disable_setup_button()
            
            success_msg = "🎉 Setup Colab lengkap berhasil diselesaikan!"
            self.log(success_msg, 'success')
            self.log("🔒 Button setup dinonaktifkan karena setup sudah selesai", 'info')
            
            return {'success': True, 'message': success_msg}
                
        except Exception as e:
            error_msg = f"Error in setup button handler: {e}"
            self.logger.error(error_msg)
            self._set_phase('error')
            # Re-enable button on error
            self._enable_setup_button()
            return {'success': False, 'message': error_msg}

    def _disable_setup_button_during_operation(self) -> None:
        """Disable setup button during operation with processing indicator."""
        try:
            # Get action container to access buttons
            action_container = self.get_component('action_container')
            if action_container and isinstance(action_container, dict):
                buttons = action_container.get('buttons', {})
                setup_button = buttons.get('colab_setup')
                
                if setup_button:
                    setup_button.disabled = True
                    setup_button.description = "⏳ Processing..."
                    setup_button.tooltip = "Setup sedang berjalan, mohon tunggu..."
                    setup_button.button_style = 'warning'
                    self.logger.debug("✅ Setup button disabled during operation")
                    
        except Exception as e:
            self.logger.error(f"Failed to disable setup button during operation: {e}")

    def reset_setup_status(self) -> Dict[str, Any]:
        """Reset setup status by removing flag file and re-enabling button."""
        try:
            self.log("🔄 Mereset status setup...", 'info')
            self._remove_setup_completion_flag()
            success_msg = "✅ Status setup telah direset. Button setup tersedia kembali."
            self.log(success_msg, 'success')
            return {'success': True, 'message': success_msg}
        except Exception as e:
            error_msg = f"Failed to reset setup status: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def _get_current_phase(self) -> str:
        """Get current phase from button state."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                colab_setup_button = self._ui_components.get('colab_setup')
                if colab_setup_button and hasattr(colab_setup_button, 'current_phase'):
                    return colab_setup_button.current_phase
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
        def post_success_callback(result):
            # Use result parameter to avoid unused variable warning
            if result and self._environment_manager:
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
                'initialized': getattr(self, '_is_initialized', False),
                'module_name': self.module_name,
                'environment_detected': self._environment_detected,
                'is_colab': self._is_colab_environment,
                'config_loaded': hasattr(self, '_config_handler') and self._config_handler is not None,
                'ui_created': hasattr(self, '_ui_components') and bool(self._ui_components)
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
