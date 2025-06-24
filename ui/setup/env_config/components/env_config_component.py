"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component dengan status_message fix dan config auto-cloning integration
"""

import logging
from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.setup.env_config.handlers.environment_config_orchestrator import EnvironmentConfigOrchestrator
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.setup.env_config.utils import (
    show_progress_safe, hide_progress_safe, get_prioritized_missing_items,
    refresh_environment_state_silent
)

# Fixed namespace berdasarkan KNOWN_NAMESPACES
ENV_CONFIG_LOGGER_NAMESPACE = "smartcash.ui.env_config"

class EnvConfigComponent:
    """🎯 Component UI dengan status_message fix dan auto-config integration"""
    
    def __init__(self):
        # Initialize UI components first
        self.ui_components = UIFactory.create_ui_components()
        
        # 🔧 Ensure status_message widget exists
        if 'status_message' not in self.ui_components:
            import ipywidgets as widgets
            self.ui_components['status_message'] = widgets.HTML(
                value="🔄 Initializing environment config...",
                layout=widgets.Layout(width='100%', margin='10px 0px')
            )
        
        # Setup metadata
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        # Ensure log_output registration
        if 'log_output' not in self.ui_components and 'log_accordion' in self.ui_components:
            self.ui_components['log_output'] = self.ui_components['log_accordion'].children[0]
        
        # Initialize logger with UI components
        self.logger = setup_ipython_logging(
            ui_components=self.ui_components,
            module_name=ENV_CONFIG_LOGGER_NAMESPACE,
            log_level=logging.INFO
        )
        
        # Initialize orchestrator with UI components
        self.orchestrator = EnvironmentConfigOrchestrator(self.ui_components)
        
        # Setup button handler
        if 'setup_button' in self.ui_components:
            self.ui_components['setup_button'].on_click(self._handle_setup_click)
        
        self.setup_completed = False
        
        # Initialize environment manager
        self._init_environment_manager_silent()
        
        # Display initial status
        self._update_initial_status()
        
        # Log startup
        if self.logger:
            self.logger.info("🚀 Environment Config Component initialized")
    
    def _init_environment_manager_silent(self):
        """🔧 Initialize environment manager tanpa premature logging"""
        try:
            from smartcash.ui.setup.env_config.helpers.silent_environment_manager import get_silent_environment_manager
            self.env_manager = get_silent_environment_manager()
            self.ui_components['env_manager'] = self.env_manager
        except Exception as e:
            self.ui_components['status_message'].value = f"⚠️ Gagal inisialisasi environment manager: {str(e)}"
    
    def _update_initial_status(self):
        """📊 Update status awal dengan proper error handling"""
        try:
            # Initialize logger if needed
            if not hasattr(self, 'logger') or self.logger is None:
                self.logger = self.orchestrator.init_logger()
            
            # Check environment status
            status = self.orchestrator.check_environment_status()
            
            # Update status message dengan fallback
            if isinstance(status, dict):
                status_message = status.get('status_message', 'Environment status tidak tersedia')
            else:
                status_message = 'Tidak dapat memeriksa status environment'
            
            self.ui_components['status_message'].value = f"ℹ️ {status_message}"
            
            # Log status
            if self.logger:
                self.logger.info(f"📍 Status environment: {status_message}")
            
        except Exception as e:
            error_msg = f"Gagal memuat status awal: {str(e)}"
            self.ui_components['status_message'].value = f"⚠️ {error_msg}"
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"❌ {error_msg}", exc_info=True)
    
    def _handle_setup_click(self, button):
        """🚀 Handle setup click dengan config auto-cloning"""
        try:
            button.disabled = True
            self._reset_ui_state()
            self._update_status("🚀 Memulai setup environment SmartCash...", "info")
            
            # Auto-clone configs sebelum setup
            self._auto_clone_configs()
            
            # Perform setup
            success = self.orchestrator.perform_environment_setup()
            
            if success:
                self.setup_completed = True
                self._update_status("✅ Environment siap digunakan", "success")
                self._display_environment_summary()
            else:
                button.disabled = False
                self._update_status("❌ Setup gagal - Coba lagi", "error")
                show_progress_safe(self.ui_components)
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ Error setup: {str(e)}")
            button.disabled = False
            self._update_status(f"❌ Error: {str(e)}", "error")
            show_progress_safe(self.ui_components)
    
    def _auto_clone_configs(self):
        """📋 Use existing SimpleConfigManager untuk sync configs"""
        try:
            from smartcash.common.config.manager import get_config_manager
            
            if self.logger:
                self.logger.info("📋 Sinkronisasi configs dengan SimpleConfigManager...")
            
            config_manager = get_config_manager()
            
            # Check if config sync needed
            if not config_manager.drive_config_dir.exists():
                config_manager.drive_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Use existing sync functionality
            sync_success = config_manager.sync_configs_to_drive()
            
            if sync_success:
                if self.logger:
                    self.logger.info("✅ Config sync berhasil dengan SimpleConfigManager")
            else:
                if self.logger:
                    self.logger.warning("⚠️ Config sync sebagian berhasil")
                    
        except Exception as e:
            if self.logger:
                self.logger.warning(f"⚠️ Config sync error: {str(e)}")
    
    def _reset_ui_state(self):
        """🔄 Reset UI state dengan utils"""
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output(wait=True)
        show_progress_safe(self.ui_components)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """📊 Update status panel dengan fallback"""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.components import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
        
        # Update status_message sebagai fallback
        emoji_map = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}
        emoji = emoji_map.get(status_type, "ℹ️")
        self.ui_components['status_message'].value = f"{emoji} {message}"
    
    def _display_environment_summary(self):
        """📊 Display environment summary dengan helper integration"""
        try:
            from smartcash.ui.setup.env_config.helpers.system_info_helper import SystemInfoHelper
            
            system_helper = SystemInfoHelper()
            summary = system_helper.get_environment_summary()
            
            if self.logger and summary:
                self.logger.info("📊 Environment Summary:")
                self.logger.info(f"  🔧 Python: {summary.get('python_version', 'Unknown')}")
                self.logger.info(f"  💾 Memory: {summary.get('memory_info', 'Unknown')}")
                self.logger.info(f"  📱 Drive: {summary.get('drive_status', 'Unknown')}")
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"🔍 Summary display error: {str(e)}")
    
    def display(self):
        """🎨 Display UI dengan utils integration"""
        display(self.ui_components['ui_layout'])
        
        # Setup logger after display
        self.logger = setup_ipython_logging(
            self.ui_components, 
            ENV_CONFIG_LOGGER_NAMESPACE, 
            redirect_all_logs=False
        )
        
        # Display environment summary
        self._display_environment_summary()
        
        # Check environment status dengan retry
        try:
            env_status = self._check_environment_status_with_retry()
            
            if env_status.get('ready', False):
                self.setup_completed = True
                if 'setup_button' in self.ui_components:
                    self.ui_components['setup_button'].disabled = True
                self._update_status("✅ Environment sudah terkonfigurasi", "success")
                hide_progress_safe(self.ui_components)
            else:
                self._update_status("🔧 Environment perlu dikonfigurasi", "info")
                
        except Exception as e:
            self._update_status(f"⚠️ Error check status: {str(e)}", "warning")
            if self.logger:
                self.logger.warning(f"⚠️ Status check error: {str(e)}")
    
    def _check_environment_status_with_retry(self, max_retries: int = 3) -> Dict[str, Any]:
        """🔄 Check environment status dengan retry mechanism"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.orchestrator.check_environment_status()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)
        
        # Return fallback status
        return {
            'ready': False,
            'status_message': f'Error check status: {str(last_error)}',
            'error': str(last_error)
        }

# 🎯 Factory function untuk backward compatibility
def create_env_config_component() -> EnvConfigComponent:
    """🏭 Factory function untuk create environment config component"""
    return EnvConfigComponent()