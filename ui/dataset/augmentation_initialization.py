"""
File: smartcash/ui/dataset/augmentation_initialization.py
Deskripsi: Inisialisasi komponen untuk augmentasi dataset dengan path handling dan tampilan lokasi dataset
"""

from typing import Dict, Any
from pathlib import Path
import os
from IPython.display import display, HTML
import ipywidgets as widgets

def detect_augmentation_state(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Inisialisasi komponen augmentasi dataset dengan utilitas standar."""
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
        default_augmented_dir = 'data/augmented'
        
        # Gunakan Google Drive jika tersedia
        if drive_mounted and drive_path:
            smartcash_dir = f"{drive_path}/SmartCash"
            default_data_dir = f"{smartcash_dir}/data"
            default_augmented_dir = f"{smartcash_dir}/data/augmented"
        
        # Ekstrak paths dari config jika tersedia
        data_dir = config.get('data', {}).get('dir', default_data_dir)
        augmented_dir = config.get('augmentation', {}).get('output_dir', default_augmented_dir)
        
        # Konversi ke path absolut untuk ditampilkan
        abs_data_dir = os.path.abspath(data_dir)
        abs_augmented_dir = os.path.abspath(augmented_dir)
        
        # Update input values dengan locations aktual
        if 'data_dir_input' in ui_components:
            ui_components['data_dir_input'].value = data_dir
        if 'output_dir_input' in ui_components:
            ui_components['output_dir_input'].value = augmented_dir
            
        # Update status panel dengan utils standar
        update_status_panel(
            ui_components, 
            "info", 
            f"Dataset akan diaugmentasi dari sumber: <strong>{abs_data_dir}</strong> ke <strong>{abs_augmented_dir}</strong>"
        )
        
        # Cek status augmented data yang sudah ada
        augmented_path = Path(augmented_dir)
        is_augmented = augmented_path.exists() and any((augmented_path/'images').glob('*.jpg')) if (augmented_path/'images').exists() else False
        
        if is_augmented:
            # Tampilkan informasi dengan utilitas standar
            update_status_panel(
                ui_components,
                "success",
                f"{ICONS['success']} Data teaugmentasi sudah tersedia di: {abs_augmented_dir}"
            )
            
            # Tampilkan tombol cleanup dan visualisasi
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'inline-block'
                
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
                
            if logger: logger.info(f"{ICONS['folder']} Data augmentasi terdeteksi di: {abs_augmented_dir}")
        
        # Store paths di ui_components
        ui_components.update({
            'data_dir': data_dir,
            'augmented_dir': augmented_dir
        })
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Inisialisasi augmentasi: {str(e)}")
    
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