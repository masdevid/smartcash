"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component UI untuk konfigurasi environment - diperbaiki dengan orchestrator dan proper error handling
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.setup.env_config.handlers.environment_orchestrator import EnvironmentOrchestrator
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE

class EnvConfigComponent:
    """
    Component UI untuk konfigurasi environment dengan orchestrator pattern
    """
    
    def __init__(self):
        """Inisialisasi component dengan proper separation of concerns"""
        
        # Create UI components first
        self.ui_components = UIFactory.create_ui_components()
        
        # Setup logging dengan namespace
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        # Setup UI logger
        self.logger = setup_ipython_logging(
            self.ui_components,
            ENV_CONFIG_LOGGER_NAMESPACE,
            redirect_all_logs=False
        )
        
        # Initialize orchestrator
        self.orchestrator = EnvironmentOrchestrator(self.ui_components)
        
        # Connect button click handler
        self.ui_components['setup_button'].on_click(self._handle_setup_click)
        
        # Track if setup was attempted
        self.setup_attempted = False
    
    def _handle_setup_click(self, button):
        """
        Handle setup button click dengan proper error handling dan progress reset
        """
        button.disabled = True
        self.setup_attempted = True
        
        try:
            # Clear previous status dan reset progress
            self._reset_ui_state()
            
            # Perform setup melalui orchestrator
            success = self.orchestrator.perform_setup()
            
            # Update button state berdasarkan hasil
            if success:
                # Keep button disabled jika berhasil
                button.disabled = True
                self._update_status("✅ Setup berhasil - Environment siap digunakan", "success")
            else:
                # Enable button untuk retry jika gagal dan reset progress
                button.disabled = False
                self._update_status("❌ Setup gagal - Silakan coba lagi", "error")
                # Reset progress bar explicitly
                self._reset_progress_bar("Setup gagal - silakan coba lagi")
                
        except Exception as e:
            # Log error, enable button untuk retry, dan reset progress
            self.logger.error(f"Error saat setup: {str(e)}")
            self._update_status(f"❌ Error: {str(e)}", "error")
            self._reset_progress_bar("Error - silakan coba lagi")
            button.disabled = False
    
    def _reset_ui_state(self):
        """Reset UI state sebelum setup"""
        # Reset progress
        self._reset_progress_bar("Memulai setup...")
        
        # Clear log output
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output(wait=True)
    
    def _reset_progress_bar(self, message: str = ""):
        """Reset progress bar dengan pesan"""
        if 'progress_bar' in self.ui_components:
            try:
                from smartcash.ui.setup.env_config.components.progress_tracking import reset_progress
                reset_progress(self.ui_components, message)
            except ImportError:
                # Fallback manual reset
                if 'progress_bar' in self.ui_components:
                    self.ui_components['progress_bar'].value = 0
                    self.ui_components['progress_bar'].description = "0%"
                if 'progress_message' in self.ui_components:
                    self.ui_components['progress_message'].value = message
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status panel"""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    def display(self):
        """
        Display the environment configuration UI dengan auto-check
        """
        # Display the UI components first
        display(self.ui_components['ui_layout'])
        
        # Run auto check untuk determine initial state
        try:
            self.logger.info("🔍 Checking environment status...")
            env_status = self.orchestrator.check_environment()
            
            # Determine button state berdasarkan status
            should_disable_button = self._should_disable_button(env_status)
            self.ui_components['setup_button'].disabled = should_disable_button
            
            # Update status message
            if env_status.get('ready', False):
                self._update_status("✅ Environment sudah terkonfigurasi", "success")
            elif env_status.get('missing_dirs'):
                missing_dirs = ', '.join(env_status['missing_dirs'])
                self._update_status(f"🔧 Perlu setup - Missing: {missing_dirs}", "warning")
            else:
                self._update_status("🔧 Environment perlu dikonfigurasi", "info")
            
            # Initialize config manager jika environment ready
            if env_status.get('ready') and 'config_manager' not in self.ui_components:
                config_manager = self.orchestrator.config_manager.initialize_config_manager()
                if config_manager:
                    self.ui_components['config_manager'] = config_manager
                    
        except Exception as e:
            # Log error tapi jangan crash UI
            self.logger.warning(f"Error saat auto-check: {str(e)}")
            self._update_status("⚠️ Error saat checking - Silakan coba setup", "warning")
            self.ui_components['setup_button'].disabled = False
    
    def _should_disable_button(self, env_status: Dict[str, Any]) -> bool:
        """
        Tentukan apakah button harus disabled berdasarkan status
        
        Args:
            env_status: Status environment dari checker
            
        Returns:
            True jika button harus disabled
        """
        # Jangan disable jika ada missing directories (user perlu bisa setup)
        if env_status.get('missing_dirs'):
            return False
        
        # Jangan disable jika setup pernah gagal (user perlu bisa retry)
        if self.setup_attempted:
            return False
        
        # Disable hanya jika benar-benar ready dan tidak ada issue
        return env_status.get('ready', False)


# Factory function untuk kompatibilitas
def create_env_config_component() -> EnvConfigComponent:
    """Factory function untuk membuat EnvConfigComponent"""
    return EnvConfigComponent()