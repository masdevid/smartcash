"""
File: smartcash/ui/dataset/download_initialization.py
Deskripsi: Inisialisasi komponen untuk download dataset dengan pengecekan dataset yang sudah ada
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML

def setup_initialization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Inisialisasi komponen download dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        from smartcash.ui.utils.constants import COLORS, ICONS
        from smartcash.ui.components.alerts import create_info_alert, create_status_indicator
        
        # Cek API key dari Google Colab Secret
        api_key_info = HTML(
            f"""<div style="padding: 10px; border-left: 4px solid {COLORS['alert_warning_text']}; 
                        color: {COLORS['alert_warning_text']}; margin: 5px 0; 
                        border-radius: 4px; background-color: {COLORS['alert_warning_bg']}">
                    <p style="margin:5px 0"><i>{ICONS['warning']} API Key diperlukan untuk download dari Roboflow</i></p>
                </div>"""
        )
        
        # Inisialisasi data directory
        data_dir = config.get('data', {}).get('dir', 'data')
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
            data_dir = str(env.drive_path / 'data')
            
            # Log penggunaan Google Drive
            if 'logger' in ui_components:
                ui_components['logger'].info(f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset: {data_dir}")
                
            # Update status panel jika tersedia
            if 'status_panel' in ui_components:
                ui_components['status_panel'].value = f"""
                <div style="padding: 10px; background-color: {COLORS['alert_info_bg']}; 
                          color: {COLORS['alert_info_text']}; margin: 10px 0; border-radius: 4px; 
                          border-left: 4px solid {COLORS['alert_info_text']};">
                    <p style="margin:5px 0">{ICONS['info']} Dataset akan disimpan di Google Drive: {data_dir}</p>
                </div>
                """
        
        # Try to get API key from Google Colab Secret
        try:
            from google.colab import userdata
            roboflow_api_key = userdata.get('ROBOFLOW_API_KEY')
            if roboflow_api_key and 'roboflow_settings' in ui_components:
                api_settings = ui_components['roboflow_settings'].children
                api_settings[0].value = roboflow_api_key
                
                api_key_info = HTML(
                    f"""<div style="padding: 10px; border-left: 4px solid {COLORS['alert_info_text']}; 
                         color: {COLORS['alert_info_text']}; margin: 5px 0; 
                         border-radius: 4px; background-color: {COLORS['alert_info_bg']}">
                        <p style="margin:5px 0"><i>{ICONS['info']} API Key Roboflow tersedia dari Google Secret.</i></p>
                    </div>"""
                )
                
                if 'logger' in ui_components:
                    ui_components['logger'].info(f"{ICONS['success']} API Key Roboflow ditemukan dari Google Secret")
        except:
            pass
        
        # Update UI with config if available
        if config and 'data' in config and 'roboflow' in config['data']:
            roboflow_config = config['data']['roboflow']
            if 'roboflow_settings' in ui_components:
                api_settings = ui_components['roboflow_settings'].children
                
                # Jangan override API key jika sudah ada
                api_key = api_settings[0].value
                if not api_key and 'api_key' in roboflow_config:
                    api_settings[0].value = roboflow_config['api_key']
                    
                # Update nilai lainnya
                api_settings[1].value = roboflow_config.get('workspace', 'smartcash-wo2us')
                api_settings[2].value = roboflow_config.get('project', 'rupiah-emisi-2022')
                api_settings[3].value = str(roboflow_config.get('version', '3'))
        
        # Cek dataset yang sudah ada
        try:
            from smartcash.ui.dataset.download_confirmation_handler import check_existing_dataset, get_dataset_stats
            
            if check_existing_dataset(data_dir):
                stats = get_dataset_stats(data_dir)
                
                # Update message status dengan info dataset
                api_key_info = HTML(
                    f"""<div style="padding: 10px; border-left: 4px solid {COLORS['alert_info_text']}; 
                         color: {COLORS['alert_info_text']}; margin: 5px 0; 
                         border-radius: 4px; background-color: {COLORS['alert_info_bg']}">
                        <p style="margin:5px 0"><i>{ICONS['info']} Dataset terdeteksi: {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']})</i></p>
                    </div>"""
                )
                
                # Update status panel
                if 'status_panel' in ui_components:
                    ui_components['status_panel'].value = f"""
                    <div style="padding: 10px; background-color: {COLORS['alert_success_bg']}; 
                              color: {COLORS['alert_success_text']}; margin: 10px 0; border-radius: 4px; 
                              border-left: 4px solid {COLORS['alert_success_text']};">
                        <p style="margin:5px 0">{ICONS['success']} Dataset sudah tersedia dengan {stats['total_images']} gambar</p>
                    </div>
                    """
        except Exception as e:
            # Gagal memeriksa dataset, biarkan saja
            pass
            
        # Initial UI setup
        if 'download_settings_container' in ui_components:
            ui_components['download_settings_container'].children = [
                ui_components['roboflow_settings'], 
                HTML(value=api_key_info.value)
            ]
        
    except ImportError:
        # Fallback jika components tidak tersedia
        pass
        
    return ui_components

def get_api_key_from_secret():
    """Dapatkan API key dari Google Colab Secret"""
    try:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY')
    except:
        return None

def update_status_panel(ui_components, status_type, message):
    """
    Update status panel dengan pesan dan jenis status.
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Jenis status ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    try:
        from smartcash.ui.utils.constants import COLORS, ALERT_STYLES
        
        if 'status_panel' in ui_components:
            # Gunakan ALERT_STYLES jika tersedia
            style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
            bg_color = style['bg_color']
            text_color = style['text_color']
            border_color = style['border_color']
            icon = style['icon']
            
            ui_components['status_panel'].value = f"""
            <div style="padding: 10px; background-color: {bg_color}; 
                        color: {text_color}; margin: 10px 0; border-radius: 4px; 
                        border-left: 4px solid {border_color};">
                <p style="margin:5px 0">{icon} {message}</p>
            </div>
            """
    except ImportError:
        # Fallback jika constants tidak tersedia
        color_mapping = {
            'info': ('#d1ecf1', '#0c5460'),    # Background, Text
            'success': ('#d4edda', '#155724'),
            'warning': ('#fff3cd', '#856404'),
            'error': ('#f8d7da', '#721c24')
        }
        icons = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        colors = color_mapping.get(status_type, color_mapping['info'])
        icon = icons.get(status_type, icons['info'])
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = f"""
            <div style="padding: 10px; background-color: {colors[0]}; 
                        color: {colors[1]}; margin: 10px 0; border-radius: 4px; 
                        border-left: 4px solid {colors[1]};">
                <p style="margin:5px 0">{icon} {message}</p>
            </div>
            """