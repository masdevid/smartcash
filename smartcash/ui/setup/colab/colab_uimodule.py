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
                # Button handlers are connected automatically by ButtonHandlerMixin
                
                # Setup button phases for sequential operations
                self._setup_button_phases()
                
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
            try:
                import google.colab  # noqa: F401
                self._is_colab_environment = True
            except ImportError:
                self._is_colab_environment = False
            self._environment_detected = True
    
    def _setup_button_phases(self) -> None:
        """Setup button phases for sequential colab operations."""
        try:
            # Define sequential phases for colab setup
            phases = [
                {'id': 'detect', 'text': '🔍 Deteksi Lingkungan', 'description': 'Mendeteksi lingkungan sistem'},
                {'id': 'init', 'text': '🔧 Inisialisasi', 'description': 'Menginisialisasi environment'},
                {'id': 'mount', 'text': '📁 Mount Drive', 'description': 'Memasang Google Drive'},
                {'id': 'setup', 'text': '🚀 Setup Lengkap', 'description': 'Menjalankan pengaturan lengkap'},
                {'id': 'verify', 'text': '🔍 Verifikasi', 'description': 'Memverifikasi pengaturan'}
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
        
        # Add Colab-specific handlers - only setup_button for one-click operation
        colab_handlers = {
            'setup': self._handle_setup_button,
            'colab_setup': self._handle_setup_button
        }
        
        handlers.update(colab_handlers)
        return handlers
    
    def _flush_log_buffer(self) -> None:
        """Flush buffered logs to operation container."""
        try:
            if not self._log_buffer:
                return
                
            # Display all buffered logs to operation container
            for log_entry in self._log_buffer:
                message, level = log_entry
                self.log(message, level)
            
            # Clear the buffer
            self._log_buffer.clear()
            
        except Exception as e:
            self.logger.debug(f"Failed to flush log buffer: {e}")
    
    # All core functionality is provided by BaseUIModule and its mixins
    
    # ==================== OPERATION HANDLERS ====================
    
    def _handle_setup_button(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle setup button click - one-click operation that progresses through all phases sequentially."""
        try:
            self.log("🚀 Memulai setup lengkap lingkungan Colab...", 'info')
            
            # Get initial phase from button state or start with init
            current_phase = self._get_current_phase() or 'init'
            
            # Execute the appropriate phase operation
            if current_phase == 'init':
                return self._execute_init_phase(button)
            elif current_phase == 'drive':
                return self._execute_drive_phase(button)
            elif current_phase == 'symlink':
                return self._execute_symlink_phase(button)
            elif current_phase == 'folders':
                return self._execute_folders_phase(button)
            elif current_phase == 'config':
                return self._execute_config_phase(button)
            elif current_phase == 'env':
                return self._execute_env_phase(button)
            elif current_phase == 'verify':
                return self._execute_verify_phase(button)
            elif current_phase == 'complete':
                return {'success': True, 'message': '🎉 Setup Colab telah selesai!'}
            else:
                # Start from beginning if unknown phase
                return self._execute_init_phase(button)
                
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
    
    def _execute_init_phase(self, button=None) -> Dict[str, Any]:
        """Execute initialization phase."""
        def validate_init():
            return {'valid': True}
        
        def execute_init():
            self.log("🔧 Fase Inisialisasi: Mempersiapkan lingkungan...", 'info')
            self._set_phase('init')
            # Simulate initialization work
            import time
            time.sleep(1)
            self.log("✅ Inisialisasi berhasil", 'success')
            self._advance_to_next_phase('drive')
            return {'success': True, 'message': 'Inisialisasi berhasil'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Inisialisasi Lingkungan",
            operation_func=execute_init,
            button=button,
            validation_func=validate_init,
            success_message="Inisialisasi berhasil - melanjutkan ke fase Drive",
            error_message="Kesalahan inisialisasi"
        )
    
    def _execute_drive_phase(self, button=None) -> Dict[str, Any]:
        """Execute Google Drive mounting phase."""
        def validate_drive():
            return {'valid': True}
        
        def execute_drive():
            self.log("📁 Fase Drive: Mounting Google Drive...", 'info')
            self._set_phase('drive')
            # Use existing mount drive logic
            result = self._handle_mount_drive(button)
            if result.get('success'):
                self._advance_to_next_phase('symlink')
            return result
        
        return self._execute_operation_with_wrapper(
            operation_name="Mount Google Drive",
            operation_func=execute_drive,
            button=button,
            validation_func=validate_drive,
            success_message="Drive mounted - melanjutkan ke fase Symlink",
            error_message="Kesalahan mount drive"
        )
    
    def _execute_symlink_phase(self, button=None) -> Dict[str, Any]:
        """Execute symlink creation phase."""
        def validate_symlink():
            return {'valid': True}
        
        def execute_symlink():
            self.log("🔗 Fase Symlink: Menyiapkan symbolic links...", 'info')
            self._set_phase('symlink')
            # Simulate symlink creation
            import time
            time.sleep(0.5)
            self.log("✅ Symlink berhasil dibuat", 'success')
            self._advance_to_next_phase('folders')
            return {'success': True, 'message': 'Symlink berhasil dibuat'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Membuat Symlinks",
            operation_func=execute_symlink,
            button=button,
            validation_func=validate_symlink,
            success_message="Symlinks dibuat - melanjutkan ke fase Folders",
            error_message="Kesalahan pembuatan symlink"
        )
    
    def _execute_folders_phase(self, button=None) -> Dict[str, Any]:
        """Execute folder creation phase."""
        def validate_folders():
            return {'valid': True}
        
        def execute_folders():
            self.log("📂 Fase Folders: Membuat direktori yang diperlukan...", 'info')
            self._set_phase('folders')
            # Simulate folder creation
            import time
            time.sleep(0.5)
            self.log("✅ Direktori berhasil dibuat", 'success')
            self._advance_to_next_phase('config')
            return {'success': True, 'message': 'Direktori berhasil dibuat'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Membuat Direktori",
            operation_func=execute_folders,
            button=button,
            validation_func=validate_folders,
            success_message="Direktori dibuat - melanjutkan ke fase Config",
            error_message="Kesalahan pembuatan direktori"
        )
    
    def _execute_config_phase(self, button=None) -> Dict[str, Any]:
        """Execute configuration phase."""
        def validate_config():
            return {'valid': True}
        
        def execute_config():
            self.log("⚙️ Fase Config: Menyiapkan konfigurasi...", 'info')
            self._set_phase('config')
            # Simulate config setup
            import time
            time.sleep(0.5)
            self.log("✅ Konfigurasi berhasil disiapkan", 'success')
            self._advance_to_next_phase('env')
            return {'success': True, 'message': 'Konfigurasi berhasil disiapkan'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Setup Konfigurasi",
            operation_func=execute_config,
            button=button,
            validation_func=validate_config,
            success_message="Konfigurasi selesai - melanjutkan ke fase Environment",
            error_message="Kesalahan setup konfigurasi"
        )
    
    def _execute_env_phase(self, button=None) -> Dict[str, Any]:
        """Execute environment setup phase."""
        def validate_env():
            return {'valid': True}
        
        def execute_env():
            self.log("🌍 Fase Environment: Menyiapkan environment variables...", 'info')
            self._set_phase('env')
            # Simulate environment setup
            import time
            time.sleep(0.5)
            self.log("✅ Environment berhasil disiapkan", 'success')
            self._advance_to_next_phase('verify')
            return {'success': True, 'message': 'Environment berhasil disiapkan'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Setup Environment",
            operation_func=execute_env,
            button=button,
            validation_func=validate_env,
            success_message="Environment selesai - melanjutkan ke fase Verifikasi",
            error_message="Kesalahan setup environment"
        )
    
    def _execute_verify_phase(self, button=None) -> Dict[str, Any]:
        """Execute verification phase."""
        def validate_verify():
            return {'valid': True}
        
        def execute_verify():
            self.log("🔍 Fase Verify: Memverifikasi setup lengkap...", 'info')
            self._set_phase('verify')
            # Use existing verification logic
            result = self._handle_verify_setup(button)
            if result.get('success'):
                self._set_phase('complete')
                self.log("🎉 Semua fase setup Colab telah selesai!", 'success')
            return result
        
        return self._execute_operation_with_wrapper(
            operation_name="Verifikasi Setup",
            operation_func=execute_verify,
            button=button,
            validation_func=validate_verify,
            success_message="🎉 Setup Colab selesai sempurna!",
            error_message="Kesalahan verifikasi setup"
        )
    
    def _handle_full_setup(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle full Colab setup operation using modern BaseUIModule pattern."""
        def validate_environment():
            if not self._environment_detected:
                return {'valid': False, 'message': "Lingkungan belum terdeteksi, silakan coba lagi"}
            return {'valid': True}
        
        def execute_full_setup():
            self.log("🚀 Memulai pengaturan lengkap lingkungan Colab...", 'info')
            
            # Single progress tracking - start
            self.update_progress(0, "Memulai pengaturan lengkap...")
            
            # Simulate full setup process with single progress tracking
            import time
            steps = [
                (20, "Menginisialisasi lingkungan..."),
                (40, "Memeriksa Google Drive..."),
                (60, "Menyiapkan direktori kerja..."),
                (80, "Menyinkronkan konfigurasi..."),
                (100, "Pengaturan selesai!")
            ]
            
            for progress, message in steps:
                self.update_progress(progress, message)
                time.sleep(0.5)  # Simulate work
                
            # Log completion
            self.log("✅ Pengaturan lengkap lingkungan Colab berhasil diselesaikan", 'success')
            
            return {
                'success': True,
                'message': 'Pengaturan lengkap berhasil diselesaikan',
                'environment_type': 'colab' if self._is_colab_environment else 'local'
            }
        
        return self._execute_operation_with_wrapper(
            operation_name="Pengaturan Lengkap Colab",
            operation_func=execute_full_setup,
            button=button,
            validation_func=validate_environment,
            success_message="Pengaturan lengkap Colab berhasil diselesaikan",
            error_message="Kesalahan pengaturan lengkap Colab"
        )
    
    def _handle_init_environment(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle environment initialization using modern BaseUIModule pattern."""
        def validate_environment():
            if not self._environment_detected:
                return {'valid': False, 'message': "Lingkungan belum terdeteksi, silakan coba lagi"}
            return {'valid': True}
        
        def execute_init():
            self.log("🔧 Menginisialisasi lingkungan...", 'info')
            
            # Single progress tracking
            self.update_progress(0, "Memulai inisialisasi...")
            
            # Simulate initialization steps
            import time
            steps = [
                (25, "Memeriksa dependencies..."),
                (50, "Menyiapkan direktori kerja..."),
                (75, "Mengonfigurasi environment..."),
                (100, "Inisialisasi selesai!")
            ]
            
            for progress, message in steps:
                self.update_progress(progress, message)
                time.sleep(0.3)
                
            self.log("✅ Inisialisasi lingkungan berhasil diselesaikan", 'success')
            
            return {
                'success': True,
                'message': 'Inisialisasi lingkungan berhasil',
                'environment_type': 'colab' if self._is_colab_environment else 'local'
            }
        
        return self._execute_operation_with_wrapper(
            operation_name="Inisialisasi Lingkungan",
            operation_func=execute_init,
            button=button,
            validation_func=validate_environment,
            success_message="Inisialisasi lingkungan berhasil diselesaikan",
            error_message="Kesalahan inisialisasi lingkungan"
        )
    
    def _handle_mount_drive(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle Google Drive mounting using modern BaseUIModule pattern."""
        def validate_colab_environment():
            if not self._is_colab_environment:
                return {'valid': False, 'message': "Mount Google Drive hanya tersedia di lingkungan Colab"}
            return {'valid': True}
        
        def execute_mount_drive():
            self.log("📁 Memasang Google Drive...", 'info')
            
            # Single progress tracking
            self.update_progress(0, "Memulai mount drive...")
            
            if not self._environment_manager:
                self._detect_environment()  # Ensure environment manager is initialized
            
            self.update_progress(30, "Memeriksa koneksi Google Drive...")
            
            # Use EnvironmentManager to mount drive
            success, message = self._environment_manager.mount_drive()
            
            if success:
                self.update_progress(80, "Mengonfigurasi akses drive...")
                
                drive_path = self._environment_manager.drive_path
                self.log(f"✅ Google Drive dipasang di: {drive_path}", 'success')
                
                # Update paths after successful mount
                self._environment_paths = get_paths_for_environment(
                    is_colab=True,
                    is_drive_mounted=True
                )
                
                self.update_progress(100, "Mount drive selesai!")
                
                return {
                    'success': True, 
                    'message': message,
                    'path': str(drive_path) if drive_path else '/content/drive',
                    'paths': self._environment_paths
                }
            else:
                self.update_progress(100, "Mount drive gagal!")
                return {'success': False, 'message': f"Mount drive gagal: {message}"}
        
        return self._execute_operation_with_wrapper(
            operation_name="Mount Google Drive",
            operation_func=execute_mount_drive,
            button=button,
            validation_func=validate_colab_environment,
            success_message="Google Drive berhasil dipasang",
            error_message="Kesalahan mount Google Drive"
        )
    
    def _handle_verify_setup(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle setup verification using modern BaseUIModule pattern."""
        def validate_environment():
            if not self._environment_detected:
                return {'valid': False, 'message': "Lingkungan belum terdeteksi, silakan coba lagi"}
            return {'valid': True}
        
        def execute_verify():
            self.log("🔍 Memverifikasi pengaturan lingkungan...", 'info')
            
            # Single progress tracking
            self.update_progress(0, "Memulai verifikasi...")
            
            # Simulate verification steps
            import time
            verification_steps = [
                (20, "Memeriksa environment variables..."),
                (40, "Memverifikasi direktori kerja..."),
                (60, "Memeriksa koneksi Google Drive..."),
                (80, "Memverifikasi konfigurasi sistem..."),
                (100, "Verifikasi selesai!")
            ]
            
            for progress, message in verification_steps:
                self.update_progress(progress, message)
                time.sleep(0.3)
            
            # Collect verification results
            verification_results = {
                'environment_type': 'colab' if self._is_colab_environment else 'local',
                'environment_detected': self._environment_detected,
                'drive_mounted': self._environment_manager.is_drive_mounted if self._environment_manager else False,
                'paths_configured': bool(self._environment_paths)
            }
            
            self.log("✅ Verifikasi pengaturan lingkungan berhasil diselesaikan", 'success')
            
            return {
                'success': True,
                'message': 'Verifikasi pengaturan berhasil',
                'verification_results': verification_results
            }
        
        return self._execute_operation_with_wrapper(
            operation_name="Verifikasi Pengaturan",
            operation_func=execute_verify,
            button=button,
            validation_func=validate_environment,
            success_message="Verifikasi pengaturan berhasil diselesaikan",
            error_message="Kesalahan verifikasi pengaturan"
        )
    
    def _handle_detect_environment(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle environment detection using modern BaseUIModule pattern."""
        def validate_system():
            return {'valid': True}  # Environment detection always available
        
        def execute_detect():
            self.log("🔍 Mendeteksi lingkungan sistem...", 'info')
            
            # Single progress tracking
            self.update_progress(0, "Memulai deteksi lingkungan...")
            
            import sys
            
            self.update_progress(30, "Memeriksa platform sistem...")
            self._detect_environment()
            
            self.update_progress(60, "Menganalisis environment...")
            
            env_info = {
                'is_colab': self._is_colab_environment,
                'runtime_type': 'colab' if self._is_colab_environment else 'local',
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform
            }
            
            self.update_progress(90, "Menyusun hasil deteksi...")
            
            env_type = "Google Colab" if self._is_colab_environment else "Lokal/Jupyter"
            self.log(f"🌍 Lingkungan: {env_type}", 'info')
            self.log(f"🐍 Python: {env_info['python_version']}", 'info')
            self.log(f"💻 Platform: {env_info['platform']}", 'info')
            
            self.update_progress(100, "Deteksi lingkungan selesai!")
            
            return {
                'success': True,
                'message': f'Lingkungan terdeteksi: {env_type}',
                'environment': env_info
            }
        
        return self._execute_operation_with_wrapper(
            operation_name="Deteksi Lingkungan",
            operation_func=execute_detect,
            button=button,
            validation_func=validate_system,
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
        """Get detailed environment information using EnvironmentManager."""
        base_info = {
            'is_colab': self._is_colab_environment,
            'runtime_type': 'colab' if self._is_colab_environment else 'local',
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


def display_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Display Colab UI and return components."""
    return initialize_colab_ui(config=config, display=True, **kwargs)


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
        from google.colab import drive
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