"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component UI untuk environment config dengan UI yang stabil dan progress yang tersembunyi setelah setup
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.setup.env_config.handlers.environment_config_orchestrator import EnvironmentConfigOrchestrator
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE

class EnvConfigComponent:
    """Component UI untuk environment config dengan orchestrator yang diperbaiki"""
    
    def __init__(self):
        """Inisialisasi component dengan UI yang stabil"""
        self.ui_components = UIFactory.create_ui_components()
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        # Setup logging tanpa redirect berlebihan
        self.logger = setup_ipython_logging(
            self.ui_components,
            ENV_CONFIG_LOGGER_NAMESPACE,
            redirect_all_logs=False
        )
        
        self.orchestrator = EnvironmentConfigOrchestrator(self.ui_components)
        self.ui_components['setup_button'].on_click(self._handle_setup_click)
        self.setup_completed = False
    
    def _handle_setup_click(self, button):
        """Handle setup button dengan UI state management yang proper"""
        button.disabled = True
        
        try:
            self._reset_ui_state()
            success = self.orchestrator.perform_environment_setup()
            
            if success:
                self.setup_completed = True
                self._update_status("✅ Environment siap digunakan", "success")
                # Button tetap disabled, progress tersembunyi
            else:
                button.disabled = False
                self._update_status("❌ Setup gagal - Coba lagi", "error")
                self._show_progress()
                
        except Exception as e:
            self.logger.error(f"Error setup: {str(e)}")
            button.disabled = False
            self._update_status(f"❌ Error: {str(e)}", "error")
            self._show_progress()
    
    def _reset_ui_state(self):
        """Reset UI state sebelum setup"""
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output(wait=True)
        self._show_progress()
    
    def _show_progress(self):
        """Tampilkan progress bar"""
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.visibility = 'visible'
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status panel"""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    def display(self):
        """Display UI dengan auto-check minimal"""
        # Display UI terlebih dahulu untuk menghindari bug tidak muncul
        display(self.ui_components['ui_layout'])
        
        try:
            # Check status tanpa logging berlebihan
            env_status = self.orchestrator.check_environment_status()
            
            if env_status.get('ready', False):
                self.setup_completed = True
                self.ui_components['setup_button'].disabled = True
                self._update_status("✅ Environment sudah terkonfigurasi", "success")
                # Sembunyikan progress jika sudah ready
                if 'progress_container' in self.ui_components:
                    self.ui_components['progress_container'].layout.visibility = 'hidden'
            else:
                # Tampilkan apa yang perlu di-setup
                missing_items = []
                if env_status.get('missing_drive_folders'):
                    missing_items.extend(env_status['missing_drive_folders'])
                if env_status.get('missing_drive_configs'):
                    missing_items.extend(env_status['missing_drive_configs'][:3])  # Max 3
                
                if missing_items:
                    items_str = ', '.join(missing_items[:3])  # Limit display
                    self._update_status(f"🔧 Perlu setup: {items_str}...", "warning")
                else:
                    self._update_status("🔧 Environment perlu dikonfigurasi", "info")
                
                self.ui_components['setup_button'].disabled = False
                self._show_progress()
            
        except Exception as e:
            # Jangan crash UI, tapi beri info
            self._update_status("⚠️ Status check error - Silakan setup", "warning")
            self.ui_components['setup_button'].disabled = False
            self._show_progress()


def create_env_config_component() -> EnvConfigComponent:
    """Factory function untuk membuat component"""
    return EnvConfigComponent()