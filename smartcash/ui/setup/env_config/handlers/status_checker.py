"""
File: smartcash/ui/setup/env_config/handlers/status_checker.py
Deskripsi: Fixed status checker dengan koneksi handler yang benar
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.status_handler import StatusHandler
from smartcash.ui.setup.env_config.handlers.system_info_handler import SystemInfoHandler
from smartcash.ui.setup.env_config.utils.ui_updater import update_summary_panels
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

class StatusChecker:
    """🔍 Fixed status checker dengan integrasi handler yang benar"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.status_handler = StatusHandler(logger)
        self.system_info_handler = SystemInfoHandler(logger)
        
    def check_and_update_status(self, ui_components: Dict[str, Any]) -> None:
        """🔍 Check status dan update UI dengan informasi yang benar"""
        try:
            # Setup logger bridge untuk UI
            if not self.logger:
                self.logger = create_ui_logger_bridge(ui_components, 'env_config')
            
            self.logger.info("🔍 Memeriksa status environment...")
            
            # Get comprehensive status
            status_data = self.status_handler.get_comprehensive_status()
            
            # Update status panels dengan posisi yang benar
            self._update_status_panels(ui_components, status_data)
            
            # Update main status display
            self._update_main_status(ui_components, status_data)
            
            # Log summary
            self._log_status_summary(status_data)
            
        except Exception as e:
            self.logger.error(f"❌ Status check failed: {str(e)}")
            self._set_error_status(ui_components, str(e))
    
    def _update_status_panels(self, ui_components: Dict[str, Any], status_data: Dict[str, Any]) -> None:
        """Update dual column status panels dengan posisi yang benar"""
        try:
            # Generate environment summary (kiri) - termasuk Google Drive
            env_summary = self.system_info_handler.generate_environment_summary({
                'configs_complete': status_data.get('summary', {}).get('configs_complete', False),
                'directories_ready': status_data.get('summary', {}).get('drive_ready', False)
            })
            
            # Generate system info (kanan) - specs system saja
            system_info = self.system_info_handler.generate_colab_system_info()
            
            # Update panels menggunakan shared component
            update_summary_panels(ui_components, env_summary, system_info)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not update status panels: {str(e)}")
    
    def _update_main_status(self, ui_components: Dict[str, Any], status_data: Dict[str, Any]) -> None:
        """Update main status panel"""
        try:
            is_ready = status_data.get('ready', False)
            
            if is_ready:
                self._set_ready_status(ui_components)
            else:
                self._set_needs_setup_status(ui_components, status_data)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not update main status: {str(e)}")
    
    def _set_ready_status(self, ui_components: Dict[str, Any]) -> None:
        """Set status sebagai ready"""
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                "<p style='color: #155724; background: #d4edda; padding: 12px; margin: 8px 0; "
                "border: 1px solid #c3e6cb; border-radius: 8px;'>"
                "✅ Environment sudah terkonfigurasi dengan baik</p>"
            )
        
        # Reset progress
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
        
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                "<span style='color: #155724; font-weight: 500;'>✅ Ready - Environment fully configured</span>"
            )
    
    def _set_needs_setup_status(self, ui_components: Dict[str, Any], status_data: Dict[str, Any]) -> None:
        """Set status sebagai needs setup dengan detail"""
        summary = status_data.get('summary', {})
        
        # Identifikasi apa yang belum ready
        missing_items = []
        if not summary.get('drive_ready', False):
            missing_items.append('Google Drive')
        if not summary.get('configs_complete', False):
            missing_items.append('Configurations')
        if not summary.get('symlinks_valid', False):
            missing_items.append('Directory Links')
        
        missing_text = f" ({', '.join(missing_items)})" if missing_items else ""
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                "<p style='color: #856404; background: #fff3cd; padding: 12px; margin: 8px 0; "
                "border: 1px solid #ffeaa7; border-radius: 8px;'>"
                f"🔧 Environment perlu dikonfigurasi{missing_text}</p>"
            )
        
        # Reset progress
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
        
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                "<span style='color: #856404;'>Siap untuk setup environment</span>"
            )
    
    def _set_error_status(self, ui_components: Dict[str, Any], error_msg: str) -> None:
        """Set error status"""
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = (
                "<p style='color: #721c24; background: #f8d7da; padding: 12px; margin: 8px 0; "
                "border: 1px solid #f5c6cb; border-radius: 8px;'>"
                f"❌ Error checking status: {error_msg}</p>"
            )
        
        if 'progress_text' in ui_components:
            ui_components['progress_text'].value = (
                "<span style='color: #721c24;'>❌ Status check failed</span>"
            )
    
    def _log_status_summary(self, status_data: Dict[str, Any]) -> None:
        """Log status summary untuk debugging"""
        if not self.logger:
            return
            
        summary = status_data.get('summary', {})
        ready = status_data.get('ready', False)
        
        self.logger.info(f"📊 Status Summary: {'✅ Ready' if ready else '🔧 Needs Setup'}")
        self.logger.info(f"  🌍 Environment: {'✅' if summary.get('environment_ready') else '❌'}")
        self.logger.info(f"  📱 Drive: {'✅' if summary.get('drive_ready') else '❌'}")
        self.logger.info(f"  📋 Configs: {'✅' if summary.get('configs_complete') else '❌'}")
        self.logger.info(f"  🔗 Symlinks: {'✅' if summary.get('symlinks_valid') else '❌'}")
    
    def quick_check_ready(self) -> bool:
        """🚀 Quick check apakah environment sudah ready"""
        try:
            status_data = self.status_handler.get_comprehensive_status()
            return status_data.get('ready', False)
        except Exception:
            return False