"""
File: smartcash/ui/setup/env_config/handlers/ui_update_handler.py
Deskripsi: Handler untuk update UI dengan data status yang benar dari status_handler
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.status_handler import StatusHandler
from smartcash.ui.setup.env_config.components.ui_components import update_environment_status
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

class UIUpdateHandler:
    """🔄 Handler untuk mengupdate UI dengan status environment yang akurat"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.status_handler = StatusHandler()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger bridge untuk UI"""
        if 'logger' not in self.ui_components:
            logger_bridge = create_ui_logger_bridge(self.ui_components, "smartcash.ui.setup.env_config.ui_update")
            self.ui_components['logger'] = logger_bridge
            return logger_bridge
        return self.ui_components['logger']
    
    def refresh_environment_status(self) -> None:
        """🔍 Refresh dan update status environment di UI"""
        try:
            self.logger.info("🔍 Memperbarui status environment...")
            
            # Get comprehensive status dari status handler
            status_data = self.status_handler.get_comprehensive_status()
            
            # Update environment summary panels
            update_environment_status(self.ui_components, status_data)
            
            # Update status panel dengan overall message
            self._update_status_panel(status_data)
            
            # Log hasil update
            summary = status_data.get('summary', {})
            if summary.get('overall_ready', False):
                self.logger.success("✅ Environment status updated - Ready!")
            else:
                self.logger.warning(f"⚠️ Environment needs setup: {summary.get('setup_message', 'Unknown issues')}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to refresh environment status: {str(e)}")
            self._set_error_status()
    
    def _update_status_panel(self, status_data: Dict[str, Any]) -> None:
        """Update main status panel dengan overall status"""
        if 'status_panel' not in self.ui_components:
            return
        
        try:
            summary = status_data.get('summary', {})
            overall_status = summary.get('overall_status', '❓ Unknown')
            setup_message = summary.get('setup_message', 'Status check dalam progress...')
            
            # Color coding berdasarkan status
            if 'Ready' in overall_status:
                color = '#28a745'
                icon = '✅'
            elif 'Setup Needed' in overall_status:
                color = '#ffc107' 
                icon = '⚙️'
            else:
                color = '#dc3545'
                icon = '❌'
            
            status_html = f"""
            <div style="padding: 15px; border-radius: 8px; border-left: 4px solid {color}; background: #f8f9fa;">
                <div style="font-weight: bold; color: {color}; margin-bottom: 8px;">
                    {icon} {overall_status}
                </div>
                <div style="color: #6c757d; font-size: 14px;">
                    {setup_message}
                </div>
            </div>
            """
            
            self.ui_components['status_panel'].value = status_html
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update status panel: {str(e)}")
    
    def _set_error_status(self) -> None:
        """Set error status jika refresh gagal"""
        if 'status_panel' in self.ui_components:
            error_html = """
            <div style="padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; background: #f8f9fa;">
                <div style="font-weight: bold; color: #dc3545; margin-bottom: 8px;">
                    ❌ Status Check Failed
                </div>
                <div style="color: #6c757d; font-size: 14px;">
                    Tidak dapat memperbarui status environment. Silakan coba refresh manual.
                </div>
            </div>
            """
            self.ui_components['status_panel'].value = error_html
    
    def update_progress(self, value: int, text: str = "") -> None:
        """📊 Update progress bar dan text"""
        try:
            if 'progress_bar' in self.ui_components:
                self.ui_components['progress_bar'].value = min(max(value, 0), 100)
            
            if text and 'progress_text' in self.ui_components:
                self.ui_components['progress_text'].value = f"<span style='color: #6c757d; font-size: 14px;'>{text}</span>"
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update progress: {str(e)}")
    
    def show_setup_needed_message(self) -> None:
        """💬 Show message bahwa setup diperlukan"""
        self.logger.warning("⚠️ Environment belum siap - klik tombol Setup Environment untuk memulai")
        
        # Update status panel untuk menunjukkan setup needed
        if 'status_panel' in self.ui_components:
            setup_html = """
            <div style="padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; background: #f8f9fa;">
                <div style="font-weight: bold; color: #856404; margin-bottom: 8px;">
                    ⚙️ Setup Required
                </div>
                <div style="color: #6c757d; font-size: 14px;">
                    Klik tombol "Setup Environment" di bawah untuk mengkonfigurasi environment SmartCash.
                </div>
            </div>
            """
            self.ui_components['status_panel'].value = setup_html
    
    def on_setup_start(self) -> None:
        """🚀 Handler ketika setup dimulai"""
        self.logger.info("🚀 Memulai setup environment...")
        self.update_progress(0, "Memulai setup environment...")
        
        if 'status_panel' in self.ui_components:
            setup_html = """
            <div style="padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; background: #f8f9fa;">
                <div style="font-weight: bold; color: #007bff; margin-bottom: 8px;">
                    ⚙️ Setup in Progress
                </div>
                <div style="color: #6c757d; font-size: 14px;">
                    Mengkonfigurasi environment SmartCash... Mohon tunggu.
                </div>
            </div>
            """
            self.ui_components['status_panel'].value = setup_html
    
    def on_setup_complete(self, success: bool = True) -> None:
        """✅ Handler ketika setup selesai"""
        if success:
            self.logger.success("🎉 Setup environment berhasil!")
            self.update_progress(100, "Setup environment selesai!")
            
            # Refresh status setelah setup
            self.refresh_environment_status()
        else:
            self.logger.error("❌ Setup environment gagal!")
            self.update_progress(0, "Setup gagal - silakan coba lagi")
            self._set_error_status()

def setup_ui_update_handler(ui_components: Dict[str, Any]) -> UIUpdateHandler:
    """🔧 Factory function untuk membuat UIUpdateHandler"""
    handler = UIUpdateHandler(ui_components)
    
    # Perform initial status refresh
    handler.refresh_environment_status()
    
    return handler