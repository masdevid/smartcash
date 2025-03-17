"""
File: smartcash/ui/dataset/download_ui_handler.py
Deskripsi: Handler untuk UI events pada download dataset
"""

from typing import Dict, Any
from IPython.display import display, HTML

def setup_ui_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk events UI.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        from smartcash.ui.utils.constants import COLORS, ICONS, ALERT_STYLES
        from smartcash.ui.components.alerts import create_status_indicator
    except ImportError:
        # Fallback jika constants tidak tersedia
        COLORS = {
            'alert_info_bg': '#d1ecf1',
            'alert_info_text': '#0c5460',
            'alert_warning_bg': '#fff3cd',
            'alert_warning_text': '#856404'
        }
        ICONS = {
            'info': 'ℹ️',
            'warning': '⚠️'
        }
    
    # Helper untuk update progress bar
    def update_progress(progress, total, message=None):
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = progress
            progress_pct = int(progress * 100 / total) if total > 0 else 0
            ui_components['progress_bar'].description = f"{progress_pct}%"
        
        if message and 'status' in ui_components:
            # Import in function to avoid circular import
            try:
                from smartcash.ui.components.alerts import create_status_indicator
                with ui_components['status']:
                    display(create_status_indicator("info", message))
            except ImportError:
                # Fallback
                with ui_components['status']:
                    display(HTML(f"<div style='padding:5px'><span>ℹ️ {message}</span></div>"))
    
    # Store helpers in ui_components for other modules to use
    ui_components['update_progress'] = update_progress
    
    # Handle download option change
    def on_download_option_change(change):
        if change['name'] == 'value' and change['new'] == 'Roboflow (Online)':
            ui_components['download_settings_container'].children = [
                ui_components['roboflow_settings'], 
                widgets.HTML(value=get_api_key_info_html())
            ]
        elif change['name'] == 'value' and change['new'] == 'Local Data (Upload)':
            ui_components['download_settings_container'].children = [ui_components['local_upload']]
    
    def get_api_key_info_html():
        # Try to get API key from Roboflow settings
        api_settings = ui_components['roboflow_settings'].children
        api_key = api_settings[0].value
        
        try:
            # Gunakan konstanta jika tersedia
            from smartcash.ui.utils.constants import COLORS, ICONS
            
            if api_key:
                return f"""<div style="padding: 10px; border-left: 4px solid {COLORS['alert_info_text']}; 
                         color: {COLORS['alert_info_text']}; margin: 5px 0; 
                         border-radius: 4px; background-color: {COLORS['alert_info_bg']}">
                        <p style="margin:5px 0"><i>{ICONS['info']} API Key Roboflow tersedia.</i></p>
                    </div>"""
            else:
                try:
                    # Try to check if available from Secret
                    from google.colab import userdata
                    api_key = userdata.get('ROBOFLOW_API_KEY')
                    if api_key:
                        return f"""<div style="padding: 10px; border-left: 4px solid {COLORS['alert_info_text']}; 
                             color: {COLORS['alert_info_text']}; margin: 5px 0; 
                             border-radius: 4px; background-color: {COLORS['alert_info_bg']}">
                            <p style="margin:5px 0"><i>{ICONS['info']} API Key Roboflow tersedia dari Google Secret.</i></p>
                        </div>"""
                except:
                    pass
                    
                return f"""<div style="padding: 10px; border-left: 4px solid {COLORS['alert_warning_text']}; 
                         color: {COLORS['alert_warning_text']}; margin: 5px 0; 
                         border-radius: 4px; background-color: {COLORS['alert_warning_bg']}">
                        <p style="margin:5px 0"><i>{ICONS['warning']} API Key diperlukan untuk download dari Roboflow</i></p>
                    </div>"""
        except ImportError:
            # Fallback tanpa konstanta
            if api_key:
                return """<div style="padding: 10px; border-left: 4px solid #0c5460; 
                         color: #0c5460; margin: 5px 0; border-radius: 4px; background-color: #d1ecf1">
                        <p style="margin:5px 0"><i>ℹ️ API Key Roboflow tersedia.</i></p>
                    </div>"""
            else:
                try:
                    from google.colab import userdata
                    api_key = userdata.get('ROBOFLOW_API_KEY')
                    if api_key:
                        return """<div style="padding: 10px; border-left: 4px solid #0c5460; 
                             color: #0c5460; margin: 5px 0; border-radius: 4px; background-color: #d1ecf1">
                            <p style="margin:5px 0"><i>ℹ️ API Key Roboflow tersedia dari Google Secret.</i></p>
                        </div>"""
                except:
                    pass
                
                return """<div style="padding: 10px; border-left: 4px solid #856404; 
                         color: #856404; margin: 5px 0; border-radius: 4px; background-color: #fff3cd">
                        <p style="margin:5px 0"><i>⚠️ API Key diperlukan untuk download dari Roboflow</i></p>
                    </div>"""
    
    # Register UI event handler
    try:
        import ipywidgets as widgets
        ui_components['download_options'].observe(on_download_option_change, names='value')
    except ImportError:
        # Fallback if ipywidgets not available
        pass
    
    return ui_components