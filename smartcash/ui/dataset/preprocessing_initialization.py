"""
File: smartcash/ui/dataset/preprocessing_initialization.py
Deskripsi: Inisialisasi komponen untuk preprocessing dataset dengan path handling yang ditingkatkan dan perbaikan warna header
"""

from typing import Dict, Any
from pathlib import Path
import os
from IPython.display import display, HTML
import ipywidgets as widgets

def setup_initialization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Inisialisasi komponen preprocessing dataset dengan utilitas standar."""
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    logger = ui_components.get('logger')
    
    try:
        # Gunakan drive_utils standar untuk deteksi Google Drive
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        
        # Inisialisasi data directory dengan drive_utils standar
        drive_mounted, drive_path = detect_drive_mount()
        
        # Dapatkan paths dari config dengan fallback ke default
        default_data_dir = 'data'
        default_preprocessed_dir = 'data/preprocessed'
        
        # Gunakan Google Drive jika tersedia
        if drive_mounted and drive_path:
            smartcash_dir = f"{drive_path}/SmartCash"
            default_data_dir = f"{smartcash_dir}/data"
            default_preprocessed_dir = f"{smartcash_dir}/data/preprocessed"
        
        # Ekstrak paths dari config jika tersedia
        data_dir = config.get('data', {}).get('dir', default_data_dir)
        preprocessed_dir = config.get('preprocessing', {}).get('output_dir', default_preprocessed_dir)
        
        # Konversi ke path absolut untuk ditampilkan
        abs_data_dir = os.path.abspath(data_dir)
        abs_preprocessed_dir = os.path.abspath(preprocessed_dir)
            
        # Update status panel dengan utils standar
        update_status_panel(
            ui_components, 
            "info", 
            f"Dataset akan dipreprocessing dari sumber: <strong> {abs_data_dir}</strong>"
        )
        
        # Update input fields dengan nilai dari config
        if 'path_input' in ui_components:
            ui_components['path_input'].value = data_dir
        
        if 'preprocessed_input' in ui_components:
            ui_components['preprocessed_input'].value = preprocessed_dir
        
        # Update path info display dengan warna teks yang terlihat
        if 'path_info' in ui_components:
            ui_components['path_info'].value = f"""
            <div style="padding:10px; margin:10px 0; background-color:{COLORS['light']}; 
                    border-radius:5px; border-left:4px solid {COLORS['primary']};">
                <h4 style="color:inherit; margin-top:0;">📂 Lokasi Dataset</h4>
                <p style="color:black;"><strong>Data Source:</strong> <code>{abs_data_dir}</code></p>
                <p style="color:black;"><strong>Preprocessed:</strong> <code>{abs_preprocessed_dir}</code></p>
            </div>
            """
        
        # Handler untuk tombol update path
        def on_update_path(b):
            # Ambil nilai dari input
            local_data_dir = ui_components['path_input'].value
            local_preprocessed_dir = ui_components['preprocessed_input'].value
            
            # Konversi ke path absolut untuk tampilan
            abs_local_data_dir = os.path.abspath(local_data_dir)
            abs_local_preprocessed_dir = os.path.abspath(local_preprocessed_dir)
            
            # Update path info display dengan warna yang terlihat
            ui_components['path_info'].value = f"""
            <div style="padding:10px; margin:10px 0; background-color:{COLORS['light']}; 
                    border-radius:5px; border-left:4px solid {COLORS['primary']};">
                <h4 style="color:{COLORS['dark']}; margin-top:0;">📂 Lokasi Dataset</h4>
                <p style="color:black;"><strong>Data Source:</strong> <code>{abs_local_data_dir}</code></p>
                <p style="color:black;"><strong>Preprocessed:</strong> <code>{abs_local_preprocessed_dir}</code></p>
            </div>
            """
            
            # Update ui_components dengan path baru tapi tetap relative path
            ui_components['data_dir'] = local_data_dir
            ui_components['preprocessed_dir'] = local_preprocessed_dir
            
            # Collapse accordion
            ui_components['path_accordion'].selected_index = None
            
            # Update status panel
            update_status_panel(
                ui_components,
                "success",
                f"{ICONS['success']} Path dataset berhasil diperbarui"
            )
            
            if logger: logger.success(f"{ICONS['success']} Path dataset diperbarui: {abs_local_data_dir}")
        
        # Register event handler
        if 'update_path_button' in ui_components:
            ui_components['update_path_button'].on_click(on_update_path)
        
        # Cek status preprocessed data yang sudah ada
        preprocessed_path = Path(preprocessed_dir)
        is_preprocessed = preprocessed_path.exists() and any(preprocessed_path.glob('**/images/*.jpg'))
        
        if is_preprocessed:
            # Tampilkan informasi dengan utilitas standar
            update_status_panel(
                ui_components,
                "success",
                f"{ICONS['success']} Dataset preprocessed sudah tersedia di: {abs_preprocessed_dir}"
            )
            
            # Tampilkan tombol cleanup dan visualisasi
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'inline-block'
                
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
                
            if 'visualize_button' in ui_components:
                ui_components['visualize_button'].layout.display = 'inline-block'
                
            if 'compare_button' in ui_components:
                ui_components['compare_button'].layout.display = 'inline-block'
                
            if 'summary_button' in ui_components:
                ui_components['summary_button'].layout.display = 'inline-block'
                
            if logger: logger.info(f"{ICONS['folder']} Dataset preprocessed terdeteksi di: {abs_preprocessed_dir}")
        
        # Store data directory di ui_components
        ui_components.update({
            'data_dir': data_dir,
            'preprocessed_dir': preprocessed_dir,
            'on_update_path': on_update_path
        })
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Inisialisasi preprocessing: {str(e)}")
    
    return ui_components

def update_status_panel(ui_components, status_type, message):
    """Update status panel dengan pesan dan jenis status, menggunakan alert_utils standar."""
    try:
        from smartcash.ui.utils.alert_utils import create_info_alert
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
        # Fallback jika alert_utils tidak tersedia
        from smartcash.ui.utils.constants import ALERT_STYLES, ICONS
        
        if 'status_panel' in ui_components:
            style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
            bg_color = style.get('bg_color', '#d1ecf1')
            text_color = style.get('text_color', '#0c5460')
            border_color = style.get('border_color', '#0c5460') 
            icon = style.get('icon', ICONS.get(status_type, 'ℹ️'))
            
            ui_components['status_panel'].value = f"""
            <div style="padding: 10px; background-color: {bg_color}; 
                        color: {text_color}; margin: 10px 0; border-radius: 4px; 
                        border-left: 4px solid {border_color};">
                <p style="margin:5px 0">{icon} {message}</p>
            </div>
            """