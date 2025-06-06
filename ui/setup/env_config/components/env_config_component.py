"""
File: smartcash/ui/setup/env_config/components/env_config_component.py
Deskripsi: Component dengan SmartProgressTracker integration untuk consistent UX
"""

from typing import Dict, Any
from IPython.display import display

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory
from smartcash.ui.setup.env_config.handlers.environment_config_orchestrator import EnvironmentConfigOrchestrator
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE
from smartcash.ui.setup.env_config.utils import (
    show_progress_safe, hide_progress_safe, get_prioritized_missing_items,
    refresh_environment_state_silent, error_progress_safe, reset_progress_safe
)

class EnvConfigComponent:
    """Component UI dengan SmartProgressTracker integration untuk environment setup"""
    
    def __init__(self):
        self.ui_components = UIFactory.create_ui_components()
        self.ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        self.ui_components['env_config_initialized'] = True
        
        self.orchestrator = EnvironmentConfigOrchestrator(self.ui_components)
        self.ui_components['setup_button'].on_click(self._handle_setup_click)
        self.setup_completed = False
        
        self._init_environment_manager_silent()
    
    def _init_environment_manager_silent(self):
        """Initialize environment manager tanpa premature logging"""
        try:
            from smartcash.ui.setup.env_config.helpers.silent_environment_manager import get_silent_environment_manager
            self.env_manager = get_silent_environment_manager()
            self.ui_components['env_manager'] = self.env_manager
        except Exception:
            pass
    
    def _display_environment_summary(self):
        """Display environment summary dengan helper integration"""
        try:
            from smartcash.ui.setup.env_config.helpers.system_info_helper import SystemInfoHelper
            from smartcash.ui.setup.env_config.helpers.silent_environment_manager import get_silent_environment_manager
            
            # Get enhanced info
            silent_env_manager = get_silent_environment_manager()
            enhanced_info = SystemInfoHelper.get_system_info()
            env_system_info = silent_env_manager.get_system_info()
            merged_info = {**enhanced_info, **env_system_info}
            
            # Format summary
            summary_lines = SystemInfoHelper.format_system_summary(merged_info)
            recommendations = SystemInfoHelper.get_system_recommendations(merged_info)
            
            # Build HTML dengan one-liner
            summary_html_parts = ["🌍 <strong>Environment Summary:</strong>"] + [f"   • {line}" for line in summary_lines]
            
            if recommendations:
                summary_html_parts += ["<br><strong>💡 Recommendations:</strong>"] + [f"   • {emoji} {rec}" for emoji, rec in recommendations[:3]]
            
            summary_html = "<br>".join(summary_html_parts)
            
            if 'env_summary_panel' in self.ui_components:
                self.ui_components['env_summary_panel'].value = f"""
                <div style="padding: 12px; background-color: #f8f9fa; color: #333; 
                           border-left: 4px solid #17a2b8; border-radius: 4px; margin: 10px 0;
                           font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.4;">
                    {summary_html}
                </div>
                """
            
            self._log_summary_highlights_minimal(merged_info)
            
        except Exception:
            self._display_basic_environment_summary()
    
    def _display_basic_environment_summary(self):
        """Fallback basic summary"""
        try:
            system_info = self.env_manager.get_system_info() if hasattr(self, 'env_manager') else {}
            
            basic_summary = [
                "🌍 <strong>Environment Summary (Basic):</strong>",
                f"   • 🏠 Platform: {system_info.get('environment', 'Unknown')}",
                f"   • 🐍 Python: {system_info.get('python_version', 'N/A')}",
                f"   • 🎮 GPU: {'✅ Available' if system_info.get('cuda_available') else '❌ CPU Only'}"
            ]
            
            basic_html = "<br>".join(basic_summary)
            
            if 'env_summary_panel' in self.ui_components:
                self.ui_components['env_summary_panel'].value = f"""
                <div style="padding: 12px; background-color: #fff3cd; color: #856404; 
                           border-left: 4px solid #ffc107; border-radius: 4px; margin: 10px 0;">
                    {basic_html}
                    <br><br><em>⚠️ Detailed system info tidak tersedia</em>
                </div>
                """
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"⚠️ Basic summary error: {str(e)}")
    
    def _log_summary_highlights_minimal(self, enhanced_info: Dict[str, Any]):
        """Log minimal highlights dengan utils pattern"""
        try:
            if not hasattr(self, 'logger') or self.logger is None:
                return
            
            env_type = enhanced_info.get('environment', 'Unknown')
            summary_parts = [env_type, "GPU✅" if enhanced_info.get('cuda_available') else "CPU-only"]
            
            if 'available_memory_gb' in enhanced_info:
                memory_gb = enhanced_info['available_memory_gb']
                summary_parts.append(f"{memory_gb:.1f}GB RAM")
            
            self.logger.info(f"📊 {' | '.join(summary_parts)}")
        except Exception:
            pass
    
    def _handle_setup_click(self, button):
        """Handle setup dengan SmartProgressTracker integration"""
        button.disabled = True
        
        try:
            self._reset_ui_state()
            
            if hasattr(self.orchestrator, 'init_logger'):
                self.orchestrator.init_logger()
            
            # Setup dengan SmartProgressTracker akan handle semua progress
            success = self.orchestrator.perform_environment_setup()
            
            if success:
                self.setup_completed = True
                self._update_status("✅ Environment siap digunakan", "success")
                self._display_environment_summary()
            else:
                button.disabled = False
                self._update_status("❌ Setup gagal - Coba lagi", "error")
                error_progress_safe(self.ui_components, "Setup gagal - Silakan coba lagi")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ Error setup: {str(e)}")
            button.disabled = False
            self._update_status(f"❌ Error: {str(e)}", "error")
            error_progress_safe(self.ui_components, f"Error: {str(e)}")
    
    def _reset_ui_state(self):
        """Reset UI state dengan SmartProgressTracker"""
        if 'log_output' in self.ui_components:
            self.ui_components['log_output'].clear_output(wait=True)
        
        # Reset dan show progress dengan SmartProgressTracker
        reset_progress_safe(self.ui_components, "🔄 Preparing setup...")
        show_progress_safe(self.ui_components)
    
    def _update_status(self, message: str, status_type: str = "info"):
        """Update status panel"""
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    def display(self):
        """Display UI dengan SmartProgressTracker integration"""
        display(self.ui_components['ui_layout'])
        
        self.logger = setup_ipython_logging(self.ui_components, ENV_CONFIG_LOGGER_NAMESPACE, redirect_all_logs=False)
        self._display_environment_summary()
        
        try:
            env_status = self._check_environment_status_with_retry()
            
            if env_status.get('ready', False):
                self.setup_completed = True
                self.ui_components['setup_button'].disabled = True
                self._update_status("✅ Environment sudah terkonfigurasi", "success")
                hide_progress_safe(self.ui_components)
            else:
                missing_items = get_prioritized_missing_items(env_status)
                
                if missing_items:
                    items_str = ', '.join(missing_items[:3])
                    self._update_status(f"🔧 Perlu setup: {items_str}...", "warning")
                else:
                    self._update_status("🔧 Environment perlu dikonfigurasi", "info")
                
                self.ui_components['setup_button'].disabled = False
                show_progress_safe(self.ui_components)
            
        except Exception:
            self._update_status("⚠️ Status check error - Silakan setup", "warning")
            self.ui_components['setup_button'].disabled = False
            show_progress_safe(self.ui_components)
    
    def _check_environment_status_with_retry(self, max_retries: int = 3) -> Dict[str, Any]:
        """Check status dengan retry dan SmartProgressTracker integration"""
        import time
        
        for attempt in range(max_retries):
            try:
                if hasattr(self, 'env_manager'):
                    refresh_environment_state_silent(self.env_manager)
                
                if attempt > 0:
                    time.sleep(1)
                
                env_status = self.orchestrator.check_environment_status()
                
                if (hasattr(self, 'env_manager') and self.env_manager.is_drive_mounted and 
                    not env_status.get('drive', {}).get('mounted', False) and attempt < max_retries - 1):
                    continue
                
                return env_status
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {'ready': False, 'error': str(e)}
                continue
        
        return {'ready': False, 'error': 'Status check failed after retries'}


def create_env_config_component() -> EnvConfigComponent:
    """Factory function dengan SmartProgressTracker integration"""
    return EnvConfigComponent()