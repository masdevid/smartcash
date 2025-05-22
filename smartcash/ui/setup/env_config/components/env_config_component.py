"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component UI yang diperbaiki dengan proper UI lifecycle dan progress management
"""

from typing import Dict, Any
from IPython.display import display, clear_output

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.setup.env_config.handlers.environment_orchestrator import EnvironmentOrchestrator
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE

class EnvConfigComponent:
    """Component UI environment config dengan lifecycle management yang lebih baik."""
    
    def __init__(self):
        self._displayed = False
        self._setup_completed = False
        
        # Create UI components dengan error handling
        try:
            self.ui_components = UIFactory.create_ui_components()
        except Exception as e:
            self.ui_components = UIFactory.create_error_ui_components(f"Error creating UI: {str(e)}")
            return
        
        # Setup logging dengan namespace dan level yang tepat
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        # Setup UI logger dengan filtering spam
        self.logger = setup_ipython_logging(
            self.ui_components,
            ENV_CONFIG_LOGGER_NAMESPACE,
            log_level=30,  # WARNING level untuk mengurangi spam
            redirect_all_logs=False
        )
        
        try:
            self.orchestrator = EnvironmentOrchestrator(self.ui_components)
            self.ui_components['setup_button'].on_click(self._handle_setup_click)
        except Exception as e:
            self.logger.error(f"Error setup orchestrator: {str(e)}")
    
    def _handle_setup_click(self, button):
        """Handle setup button dengan proper state management."""
        button.disabled = True
        
        try:
            # Reset UI state
            self._reset_ui_state()
            
            # Perform setup
            success = self.orchestrator.perform_setup()
            
            if success:
                self._setup_completed = True
                button.disabled = True  # Keep disabled after success
                self._hide_progress_after_success()
                self._update_status("✅ Environment siap digunakan", "success")
            else:
                button.disabled = False  # Allow retry
                self._reset_progress_bar("Setup gagal - coba lagi")
                self._update_status("❌ Setup gagal", "error")
                
        except Exception as e:
            self.logger.error(f"Error setup: {str(e)}")
            self._update_status(f"❌ Error: {str(e)}", "error")
            self._reset_progress_bar("Error - coba lagi")
            button.disabled = False
    
    def _reset_ui_state(self):
        """Reset UI state dengan proper cleanup."""
        # Show progress bar
        self._show_progress_bar()
        
        # Reset progress
        self._reset_progress_bar("Memulai setup...")
        
        # Clear log dengan limit
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output(wait=True)
    
    def _show_progress_bar(self):
        """Show progress bar container."""
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.visibility = 'visible'
            self.ui_components['progress_container'].layout.display = 'block'
    
    def _hide_progress_after_success(self):
        """Hide progress bar setelah setup berhasil."""
        if 'progress_container' in self.ui_components and self._setup_completed:
            # Hide dengan delay untuk smooth transition
            import threading
            import time
            
            def hide_with_delay():
                time.sleep(2)  # Wait 2 seconds
                if self._setup_completed:  # Double check
                    try:
                        self.ui_components['progress_container'].layout.visibility = 'hidden'
                        self.ui_components['progress_container'].layout.display = 'none'
                    except:
                        pass  # Ignore errors during cleanup
            
            # Run in background thread
            threading.Thread(target=hide_with_delay, daemon=True).start()
    
    def _reset_progress_bar(self, message: str = ""):
        """Reset progress bar dengan message."""
        try:
            from smartcash.ui.setup.env_config.components.progress_tracking import reset_progress
            reset_progress(self.ui_components, message)
        except ImportError:
            # Fallback manual
            if 'progress_bar' in self.ui_components:
                self.ui_components['progress_bar'].value = 0
                self.ui_components['progress_bar'].description = "0%"
            if 'progress_message' in self.ui_components:
                self.ui_components['progress_message'].value = message
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status panel dengan rate limiting."""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    def display(self):
        """Display UI dengan idempotent behavior."""
        # Prevent multiple displays
        if self._displayed:
            return
        
        try:
            # Clear output sebelum display untuk mencegah UI ganda
            clear_output(wait=True)
            
            # Display UI
            display(self.ui_components['ui_layout'])
            self._displayed = True
            
            # Auto-check environment status
            self._perform_auto_check()
            
        except Exception as e:
            print(f"❌ Error displaying UI: {str(e)}")
    
    def _perform_auto_check(self):
        """Auto-check environment dengan error handling."""
        try:
            env_status = self.orchestrator.check_environment()
            
            # Update button state
            should_disable = self._should_disable_button(env_status)
            self.ui_components['setup_button'].disabled = should_disable
            
            # Update status message
            if env_status.get('ready', False):
                self._setup_completed = True
                self._hide_progress_after_success()
                self._update_status("✅ Environment sudah dikonfigurasi", "success")
            elif env_status.get('missing_dirs'):
                missing_dirs = ', '.join(env_status['missing_dirs'][:3])  # Limit display
                if len(env_status['missing_dirs']) > 3:
                    missing_dirs += "..."
                self._update_status(f"🔧 Perlu setup - Missing: {missing_dirs}", "warning")
            else:
                self._update_status("🔧 Environment perlu dikonfigurasi", "info")
            
            # Initialize config manager jika ready
            if env_status.get('ready') and 'config_manager' not in self.ui_components:
                try:
                    config_manager = self.orchestrator.config_manager.initialize_config_manager()
                    if config_manager:
                        self.ui_components['config_manager'] = config_manager
                except Exception:
                    pass  # Silent fail
                    
        except Exception as e:
            # Silent fail dengan minimal logging
            self._update_status("⚠️ Error checking - Silakan setup", "warning")
            self.ui_components['setup_button'].disabled = False
    
    def _should_disable_button(self, env_status: Dict[str, Any]) -> bool:
        """Tentukan button state berdasarkan status."""
        # Enable jika ada missing directories atau belum ready
        if env_status.get('missing_dirs') or not env_status.get('ready', False):
            return False
        
        # Disable jika sudah complete dan ready
        return env_status.get('ready', False)

def create_env_config_component() -> EnvConfigComponent:
    """Factory function untuk EnvConfigComponent."""
    return EnvConfigComponent()