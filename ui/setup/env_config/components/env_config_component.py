"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component dengan flexbox layout dan display method untuk cell script
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
    """🎯 Component UI dengan flexbox layout dan shared components"""
    
    def __init__(self):
        # Initialize UI components dari factory
        self.ui_components = UIFactory.create_ui_components()
        
        # Setup metadata
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        # Initialize logger
        self.logger = setup_ipython_logging(
            ui_components=self.ui_components,
            module_name=ENV_CONFIG_LOGGER_NAMESPACE,
            log_level=logging.INFO
        )
        
        # Initialize orchestrator
        self.orchestrator = EnvironmentConfigOrchestrator(self.ui_components)
        
        # Setup handlers
        self._setup_handlers()
        
        # Component state
        self.setup_completed = False
    
    def display(self):
        """🎨 Display UI dengan shared components"""
        # Display main UI
        display(self.ui_components['ui_layout'])
        
        # Check environment status
        self._check_and_update_status()
    
    def _setup_handlers(self):
        """🔧 Setup event handlers untuk UI components"""
        if 'setup_button' in self.ui_components:
            setup_button = self.ui_components['setup_button']
            setup_button.on_click(self._handle_setup_click)
    
    def _handle_setup_click(self, button):
        """🚀 Handle setup button click dengan prioritized missing items feedback"""
        try:
            # Disable button
            button.disabled = True
            button.description = "🔄 Setting up..."
            
            # Refresh state sebelum setup
            refresh_environment_state_silent()
            
            # Get prioritized items untuk progress feedback
            env_status = self.orchestrator.check_environment_status()
            missing_items = get_prioritized_missing_items(env_status)
            
            if missing_items:
                setup_msg = f"🔄 Setting up {len(missing_items)} items..."
                self.logger.info(f"📋 Setup items: {', '.join(missing_items)}")
            else:
                setup_msg = "🔄 Memulai setup environment..."
            
            # Update status dengan specific feedback
            self._update_status(setup_msg, "info")
            
            # Show progress
            show_progress_safe(self.ui_components)
            
            # Run setup
            result = self.orchestrator.setup_environment()
            
            if result.get('success', False):
                self._update_status("✅ Setup environment berhasil!", "success")
                self.setup_completed = True
                button.description = "✅ Setup Complete"
                
                if self.logger:
                    self.logger.info("✅ Environment setup completed successfully")
            else:
                error_msg = result.get('error', 'Unknown error')
                self._update_status(f"❌ Setup gagal: {error_msg}", "error")
                button.disabled = False
                button.description = "🔄 Retry Setup"
                
                if self.logger:
                    self.logger.error(f"❌ Setup failed: {error_msg}")
                    
        except Exception as e:
            self._update_status(f"🚨 Error setup: {str(e)}", "error")
            button.disabled = False
            button.description = "🔄 Retry Setup"
            
            if self.logger:
                self.logger.error(f"🚨 Setup error: {str(e)}")
        finally:
            # Hide progress setelah selesai
            hide_progress_safe(self.ui_components)
    
    def _check_and_update_status(self):
        """🔍 Check dan update status environment dengan prioritized diagnostics"""
        try:
            # Refresh environment state untuk data terbaru
            refresh_environment_state_silent()
            
            # Check environment status
            env_status = self.orchestrator.check_environment_status()
            
            if env_status.get('ready', False):
                self.setup_completed = True
                if 'setup_button' in self.ui_components:
                    self.ui_components['setup_button'].disabled = True
                    self.ui_components['setup_button'].description = "✅ Already Setup"
                self._update_status("✅ Environment sudah terkonfigurasi", "success")
                hide_progress_safe(self.ui_components)
            else:
                # Get prioritized missing items untuk user-friendly diagnostics
                missing_items = get_prioritized_missing_items(env_status)
                
                if missing_items:
                    items_text = ", ".join(missing_items[:3])  # Show max 3 items
                    more_text = f" (+{len(missing_items)-3} lainnya)" if len(missing_items) > 3 else ""
                    status_msg = f"🔧 Perlu setup: {items_text}{more_text}"
                else:
                    status_msg = "🔧 Environment perlu dikonfigurasi"
                    
                self._update_status(status_msg, "info")
                
                if self.logger and missing_items:
                    self.logger.info(f"📋 Missing items: {', '.join(missing_items)}")
                
        except Exception as e:
            self._update_status(f"⚠️ Error check status: {str(e)}", "warning")
            if self.logger:
                self.logger.warning(f"⚠️ Status check error: {str(e)}")
    
    def _update_status(self, message: str, status_type: str = "info"):
        """📊 Update status message dengan color coding"""
        if 'status_panel' in self.ui_components:
            try:
                # Update status panel jika menggunakan shared components
                status_panel = self.ui_components['status_panel']
                if hasattr(status_panel, 'children') and len(status_panel.children) > 0:
                    status_html = status_panel.children[0]
                    if hasattr(status_html, 'value'):
                        color_map = {
                            'success': '#d4edda',
                            'error': '#f8d7da',
                            'warning': '#fff3cd',
                            'info': '#d1ecf1'
                        }
                        border_map = {
                            'success': '#28a745',
                            'error': '#dc3545',
                            'warning': '#ffc107',
                            'info': '#17a2b8'
                        }
                        
                        status_html.value = f"""
                        <div style="background: {color_map.get(status_type, '#d1ecf1')}; 
                                    padding: 12px; border-radius: 8px; 
                                    border-left: 4px solid {border_map.get(status_type, '#17a2b8')}; 
                                    margin: 10px 0px;">
                            {message}
                        </div>
                        """
            except Exception as e:
                print(f"⚠️ Error updating status: {e}")
    
    def get_ui_components(self) -> Dict[str, Any]:
        """📦 Get UI components untuk external access"""
        return self.ui_components
    
    def is_setup_completed(self) -> bool:
        """✅ Check jika setup sudah completed"""
        return self.setup_completed

# 🎯 Factory function untuk create component
def create_env_config_component() -> EnvConfigComponent:
    """🏭 Factory function untuk create environment config component"""
    return EnvConfigComponent()