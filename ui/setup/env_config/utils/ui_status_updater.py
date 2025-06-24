"""
File: smartcash/ui/setup/env_config/utils/ui_status_updater.py
Deskripsi: Utility untuk sync UI status dengan handler yang benar
"""

from typing import Dict, Any
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

def update_environment_status_ui(ui_components: Dict[str, Any], status: Dict[str, Any], logger=None):
    """🔄 Update UI environment status dengan sync yang tepat"""
    try:
        # Format status untuk UI
        ui_status = status.get_status_for_ui_update() if hasattr(status, 'get_status_for_ui_update') else status
        
        # Update status panel utama
        _update_main_status_panel(ui_components, ui_status)
        
        # Update environment summary (panel kiri)
        _update_environment_summary_panel(ui_components, ui_status)
        
        # Update system info (panel kanan) - hanya info Drive
        _update_system_info_panel(ui_components, ui_status)
        
        # Update button state
        _update_setup_button_state(ui_components, ui_status)
        
        if logger:
            overall_status = "Ready" if ui_status.get('overall_ready') else "Perlu Setup"
            logger.info(f"🔄 UI status updated - {overall_status}")
            
    except Exception as e:
        if logger:
            logger.error(f"❌ UI status update gagal: {str(e)}")
        _set_error_fallback_ui(ui_components, str(e))

def _update_main_status_panel(ui_components: Dict[str, Any], ui_status: Dict[str, Any]):
    """Update panel status utama"""
    if 'status_panel' not in ui_components:
        return
        
    is_ready = ui_status.get('overall_ready', False)
    
    if is_ready:
        status_html = (
            "<div style='padding: 12px; background: #d4edda; border: 1px solid #c3e6cb; "
            "border-radius: 8px; color: #155724;'>"
            "<h4 style='margin: 0 0 8px 0; color: #155724;'>✅ Environment Ready</h4>"
            "<p style='margin: 0; font-size: 14px;'>Semua komponen environment sudah terkonfigurasi dengan baik</p>"
            "</div>"
        )
    else:
        # Hitung missing components
        components = ui_status.get('components', {})
        missing_items = []
        
        for comp_name, comp_data in components.items():
            if not comp_data.get('ready', False):
                missing_items.append(comp_data.get('label', comp_name))
        
        missing_text = f" ({', '.join(missing_items)})" if missing_items else ""
        
        status_html = (
            "<div style='padding: 12px; background: #fff3cd; border: 1px solid #ffeaa7; "
            "border-radius: 8px; color: #856404;'>"
            "<h4 style='margin: 0 0 8px 0; color: #856404;'>🔧 Setup Required</h4>"
            f"<p style='margin: 0; font-size: 14px;'>Environment perlu dikonfigurasi{missing_text}</p>"
            "</div>"
        )
    
    ui_components['status_panel'].value = status_html

def _update_environment_summary_panel(ui_components: Dict[str, Any], ui_status: Dict[str, Any]):
    """Update environment summary panel (kiri)"""
    if 'summary_panel' not in ui_components:
        return
    
    components = ui_status.get('components', {})
    
    # Environment summary list
    env_items = [
        f"<li>Python Environment: <span style='color: #28a745;'>{components.get('python', {}).get('icon', '❌')} {components.get('python', {}).get('label', 'Unknown')}</span></li>",
        f"<li>Google Drive: <span style='color: {'#28a745' if components.get('drive', {}).get('ready') else '#dc3545'};'>{components.get('drive', {}).get('icon', '❌')} {components.get('drive', {}).get('label', 'Unknown')}</span></li>",
        f"<li>Configurations: <span style='color: {'#28a745' if components.get('configs', {}).get('ready') else '#dc3545'};'>{components.get('configs', {}).get('icon', '❌')} {components.get('configs', {}).get('label', 'Unknown')}</span></li>",
        f"<li>Directory Structure: <span style='color: {'#28a745' if components.get('directories', {}).get('ready') else '#dc3545'};'>{components.get('directories', {}).get('icon', '❌')} {components.get('directories', {}).get('label', 'Unknown')}</span></li>"
    ]
    
    summary_html = (
        "<div style='background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>"
        "<h4 style='margin: 0 0 10px 0; color: #495057; font-size: 16px;'>📊 Environment Summary</h4>"
        "<ul style='margin: 0; padding-left: 20px; font-size: 14px;'>"
        + "".join(env_items) +
        "</ul>"
        "</div>"
    )
    
    ui_components['summary_panel'].value = summary_html

def _update_system_info_panel(ui_components: Dict[str, Any], ui_status: Dict[str, Any]):
    """Update system info panel (kanan) - fokus pada Drive info"""
    if 'system_info_panel' not in ui_components:
        return
    
    components = ui_status.get('components', {})
    drive_info = components.get('drive', {})
    
    # Drive status dengan path info
    drive_status_html = ""
    if drive_info.get('ready', False):
        drive_path = drive_info.get('path', '')
        drive_status_html = (
            f"<div style='background: #d4edda; padding: 8px; border-radius: 4px; margin: 8px 0;'>"
            f"<strong>💾 Google Drive Status:</strong> ✅ Mounted<br>"
            f"<small style='color: #666;'>Path: {drive_path}</small>"
            f"</div>"
        )
    else:
        drive_status_html = (
            f"<div style='background: #f8d7da; padding: 8px; border-radius: 4px; margin: 8px 0;'>"
            f"<strong>💾 Google Drive Status:</strong> ❌ Not Connected"
            f"</div>"
        )
    
    # Config info
    config_info = components.get('configs', {})
    config_count = config_info.get('count', 0)
    config_status_html = (
        f"<div style='background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 8px 0;'>"
        f"<strong>📋 Configurations:</strong> {config_info.get('icon', '❌')} {config_info.get('label', 'Unknown')}<br>"
        f"<small style='color: #666;'>Found: {config_count} config files</small>"
        f"</div>"
    )
    
    system_info_html = (
        "<div style='background: #f8f9fa; padding: 12px; border-radius: 8px;'>"
        "<h4 style='margin: 0 0 10px 0; color: #495057; font-size: 16px;'>🖥️ System Information</h4>"
        + drive_status_html + config_status_html +
        "</div>"
    )
    
    ui_components['system_info_panel'].value = system_info_html

def _update_setup_button_state(ui_components: Dict[str, Any], ui_status: Dict[str, Any]):
    """Update setup button state"""
    if 'setup_button' not in ui_components:
        return
    
    is_ready = ui_status.get('overall_ready', False)
    
    if is_ready:
        ui_components['setup_button'].description = "✅ Environment Ready"
        ui_components['setup_button'].button_style = 'success'
        ui_components['setup_button'].disabled = True
    else:
        ui_components['setup_button'].description = "🔧 Setup Environment"
        ui_components['setup_button'].button_style = 'primary'
        ui_components['setup_button'].disabled = False

def _set_error_fallback_ui(ui_components: Dict[str, Any], error_msg: str):
    """Set error fallback UI state"""
    error_html = (
        "<div style='padding: 12px; background: #f8d7da; border: 1px solid #f5c6cb; "
        "border-radius: 8px; color: #721c24;'>"
        "<h4 style='margin: 0 0 8px 0; color: #721c24;'>❌ Status Check Error</h4>"
        f"<p style='margin: 0; font-size: 14px;'>Error: {error_msg}</p>"
        "</div>"
    )
    
    if 'status_panel' in ui_components:
        ui_components['status_panel'].value = error_html
    
    if 'setup_button' in ui_components:
        ui_components['setup_button'].description = "⚠️ Check Environment"
        ui_components['setup_button'].button_style = 'warning'
        ui_components['setup_button'].disabled = False

def create_status_checker_with_ui_sync(ui_components: Dict[str, Any], logger=None):
    """🔄 Create status checker dengan UI sync otomatis"""
    from smartcash.ui.setup.env_config.handlers.status_handler import StatusHandler
    
    def check_and_sync_status():
        """Internal function untuk check dan sync status"""
        try:
            status_handler = StatusHandler(logger)
            status = status_handler.get_comprehensive_status()
            
            # Update UI dengan status terbaru
            update_environment_status_ui(ui_components, {'overall_ready': status['ready'], 'components': {
                'python': {'ready': status['python_ready'], 'icon': '✅' if status['python_ready'] else '❌', 'label': 'Ready' if status['python_ready'] else 'Not Ready'},
                'drive': {'ready': status['drive_ready'], 'icon': '✅' if status['drive_ready'] else '❌', 'label': 'Connected' if status['drive_ready'] else 'Not Connected', 'path': status.get('drive_path', '')},
                'configs': {'ready': status['configs_ready'], 'icon': '✅' if status['configs_ready'] else '❌', 'label': 'Complete' if status['configs_ready'] else 'Incomplete', 'count': status['config_count']},
                'directories': {'ready': status['directories_ready'], 'icon': '✅' if status['directories_ready'] else '❌', 'label': 'Ready' if status['directories_ready'] else 'Not Ready', 'count': status['directory_count']}
            }}, logger)
            
            return status
            
        except Exception as e:
            if logger:
                logger.error(f"❌ Status check dengan UI sync gagal: {str(e)}")
            _set_error_fallback_ui(ui_components, str(e))
            return None
    
    return check_and_sync_status