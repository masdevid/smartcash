"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Fixed initializer dengan proper state management untuk UI berdasarkan environment status
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.ui_components import create_env_config_ui, setup_ui_logger_bridge, update_summary_panels
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.env_config.handlers.system_info_handler import SystemInfoHandler
from smartcash.ui.setup.env_config.handlers.status_handler import StatusHandler

class EnvConfigInitializer:
    """🚀 Fixed orchestrator dengan proper UI state management"""
    
    def __init__(self):
        self.setup_handler = SetupHandler()
        self.system_info_handler = SystemInfoHandler()
        self.status_handler = StatusHandler()
        
    def initialize_env_config_ui(self) -> Dict[str, Any]:
        """Inisialisasi UI dengan proper state management"""
        # Create UI components
        ui_components = create_env_config_ui()
        
        # Setup logger bridge dengan namespace
        logger = setup_ui_logger_bridge(ui_components, "ENV")
        
        # Setup event handlers
        self._setup_event_handlers(ui_components, logger)
        
        # 🔧 FIXED: Comprehensive status check dan UI update
        self._perform_comprehensive_status_check(ui_components, logger)
        
        return ui_components
    
    def _setup_event_handlers(self, ui_components: Dict[str, Any], logger):
        """Setup event handlers untuk UI components"""
        if 'setup_button' in ui_components:
            ui_components['setup_button'].on_click(
                lambda b: self.setup_handler.handle_setup_click(ui_components, logger)
            )
    
    def _perform_comprehensive_status_check(self, ui_components: Dict[str, Any], logger):
        """🔧 FIXED: Comprehensive status check dengan UI state update"""
        logger.info("🔍 Memeriksa status environment...")
        
        try:
            # Get comprehensive status
            env_status = self.status_handler.get_comprehensive_status()
            
            # Update UI berdasarkan status
            self._update_ui_based_on_status(ui_components, env_status, logger)
            
            # Update system info panels
            self._update_system_panels(ui_components, env_status, logger)
            
        except Exception as e:
            logger.error(f"❌ Error checking environment status: {str(e)}")
            self._set_ui_error_state(ui_components, str(e))
    
    def _update_ui_based_on_status(self, ui_components: Dict[str, Any], 
                                  env_status: Dict[str, Any], logger):
        """🔧 FIXED: Update UI state berdasarkan environment status"""
        is_ready = env_status.get('ready', False)
        
        if is_ready:
            # Environment sudah ready - disable button dan update status
            self._set_environment_ready_state(ui_components, logger)
        else:
            # Environment belum ready - enable button dan show missing items
            self._set_environment_setup_needed_state(ui_components, env_status, logger)
    
    def _set_environment_ready_state(self, ui_components: Dict[str, Any], logger):
        """Set UI state untuk environment yang sudah ready"""
        # Disable setup button
        if 'setup_button' in ui_components:
            setup_button = ui_components['setup_button']
            setup_button.disabled = True
            setup_button.description = "✅ Environment Ready"
            setup_button.button_style = 'success'
        
        # Update status panel
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                "<p style='color: #28a745; padding: 10px; margin: 5px 0; "
                "background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px;'>"
                "✅ Environment sudah terkonfigurasi dengan baik dan siap digunakan</p>"
            )
        
        # Update progress to 100%
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
        
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                "<span style='color: #28a745; font-weight: bold;'>✅ Setup selesai - Environment siap digunakan</span>"
            )
        
        logger.success("✅ Environment sudah terkonfigurasi dengan baik")
    
    def _set_environment_setup_needed_state(self, ui_components: Dict[str, Any], 
                                          env_status: Dict[str, Any], logger):
        """Set UI state untuk environment yang perlu setup"""
        # Enable setup button
        if 'setup_button' in ui_components:
            setup_button = ui_components['setup_button']
            setup_button.disabled = False
            setup_button.description = "🚀 Setup Environment"
            setup_button.button_style = 'primary'
        
        # Update status panel dengan missing items
        missing_items = env_status.get('missing_items', [])
        missing_text = f" ({', '.join(missing_items)})" if missing_items else ""
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                f"<p style='color: #856404; padding: 10px; margin: 5px 0; "
                f"background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;'>"
                f"🔧 Environment perlu dikonfigurasi{missing_text}</p>"
            )
        
        # Reset progress
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
        
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                "<span style='color: #856404;'>Siap untuk setup environment</span>"
            )
        
        logger.warning(f"🔧 Environment perlu dikonfigurasi{missing_text}")
    
    def _update_system_panels(self, ui_components: Dict[str, Any], 
                            env_status: Dict[str, Any], logger):
        """Update dual column summary panels"""
        try:
            # Generate environment summary
            env_summary = self._generate_environment_summary(env_status)
            
            # Generate system info (Colab specific)
            colab_info = self.system_info_handler.generate_colab_system_info()
            
            # Update panels
            update_summary_panels(ui_components, env_summary, colab_info)
            
        except Exception as e:
            logger.warning(f"⚠️ Could not update system panels: {str(e)}")
    
    def _generate_environment_summary(self, env_status: Dict[str, Any]) -> str:
        """Generate HTML summary untuk environment status"""
        is_ready = env_status.get('ready', False)
        
        # Basic status items
        status_items = [
            f"<li>Python Environment: <span style='color: #28a745;'>✅ Ready</span></li>",
            f"<li>Google Drive: <span style='color: {'#28a745' if env_status.get('drive_ready') else '#dc3545'};'>{'✅ Connected' if env_status.get('drive_ready') else '❌ Not Connected'}</span></li>",
            f"<li>Configurations: <span style='color: {'#28a745' if env_status.get('configs_complete') else '#dc3545'};'>{'✅ Complete' if env_status.get('configs_complete') else '❌ Incomplete'}</span></li>",
            f"<li>Directory Structure: <span style='color: {'#28a745' if env_status.get('folders_ready') else '#dc3545'};'>{'✅ Ready' if env_status.get('folders_ready') else '❌ Not Ready'}</span></li>"
        ]
        
        return f"""
        <div style="margin-bottom: 10px;">
            <p style="margin: 5px 0; font-weight: bold; color: {'#28a745' if is_ready else '#856404'};">
                Status: {'✅ Ready' if is_ready else '🔧 Setup Needed'}
            </p>
        </div>
        <ul style="margin: 10px 0; padding-left: 20px;">
            {''.join(status_items)}
        </ul>
        """
    
    def _set_ui_error_state(self, ui_components: Dict[str, Any], error_msg: str):
        """Set UI state untuk error condition"""
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                f"<p style='color: #721c24; padding: 10px; margin: 5px 0; "
                f"background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px;'>"
                f"❌ Error checking environment: {error_msg}</p>"
            )
        
        # Keep setup button enabled untuk retry
        if 'setup_button' in ui_components:
            ui_components['setup_button'].disabled = False


def initialize_environment_config_ui() -> Dict[str, Any]:
    """🚀 Entry point dengan proper state management"""
    from IPython.display import display
    
    initializer = EnvConfigInitializer()
    ui_components = initializer.initialize_env_config_ui()
    
    # Display UI
    if 'ui' in ui_components:
        display(ui_components['ui'])
    
    return ui_components