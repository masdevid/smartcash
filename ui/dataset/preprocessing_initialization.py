"""
File: smartcash/ui/dataset/preprocessing_initialization.py
Deskripsi: Inisialisasi komponen untuk preprocessing dataset dengan utilitas standar
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS, ALERT_STYLES
from IPython.display import display, HTML

def setup_initialization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Inisialisasi komponen preprocessing dataset dengan utilitas standar."""
    
    logger = ui_components.get('logger')
    
    try:
        # Gunakan drive_utils standar untuk deteksi Google Drive
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        
        # Inisialisasi data directory dengan drive_utils standar
        drive_mounted, drive_path = detect_drive_mount()
        
        # Dapatkan paths dari config dengan fallback ke default
        data_dir = config.get('data', {}).get('dir', 'data')
        preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        
        # Gunakan Google Drive jika tersedia
        if drive_mounted and drive_path:
            data_dir = f"{drive_path}/data"
            preprocessed_dir = f"{drive_path}/data/preprocessed"
            
            # Log penggunaan Google Drive jika logger tersedia
            if logger: logger.info(f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset: {data_dir}")
                
            # Update status panel dengan utils standar
            update_status_panel(
                ui_components, 
                "info", 
                f"{ICONS['info']} Dataset akan dipreprocessing dari sumber: {data_dir}"
            )
        
        # Update UI dari config
        if config and 'preprocessing' in config:
            preproc_config = config.get('preprocessing', {})
            preproc_options = ui_components.get('preprocess_options')
            
            if preproc_options and hasattr(preproc_options, 'children'):
                # Update komponen UI dengan nilai dari config menggunakan one-liner
                children = preproc_options.children
                
                # Update image size dengan validasi
                if 'img_size' in preproc_config and len(children) > 0 and hasattr(children[0], 'value'):
                    img_size = preproc_config['img_size']
                    children[0].value = img_size[0] if isinstance(img_size, list) and len(img_size) > 0 else 640
                
                # Update normalization dengan validasi
                if 'normalization' in preproc_config and len(children) > 1 and hasattr(children[1], 'value'):
                    children[1].value = preproc_config['normalization'].get('enabled', True)
                
                # Update preserve aspect ratio dengan validasi
                if 'normalization' in preproc_config and len(children) > 2 and hasattr(children[2], 'value'):
                    children[2].value = preproc_config['normalization'].get('preserve_aspect_ratio', True)
                
                # Update cache dengan validasi
                if 'enabled' in preproc_config and len(children) > 3 and hasattr(children[3], 'value'):
                    children[3].value = preproc_config['enabled']
                
                # Update workers dengan validasi
                if 'num_workers' in preproc_config and len(children) > 4 and hasattr(children[4], 'value'):
                    children[4].value = preproc_config['num_workers']
            
            # Update validation options dengan validasi
            val_options = ui_components.get('validation_options')
            if val_options and hasattr(val_options, 'children') and 'validate' in preproc_config:
                val_config = preproc_config['validate']
                children = val_options.children
                
                # Update validation options dengan validasi menggunakan one-liner
                if len(children) > 0 and hasattr(children[0], 'value'): children[0].value = val_config.get('enabled', True)
                if len(children) > 1 and hasattr(children[1], 'value'): children[1].value = val_config.get('fix_issues', True)
                if len(children) > 2 and hasattr(children[2], 'value'): children[2].value = val_config.get('move_invalid', True)
        
        # Cek status preprocessed data yang sudah ada menggunakan Path standar
        from pathlib import Path
        
        # Cek apakah sudah ada hasil preprocessing
        preprocessed_path = Path(preprocessed_dir)
        
        # Gunakan utilitas standar untuk cek data
        is_preprocessed = preprocessed_path.exists() and any(preprocessed_path.glob('**/images/*.jpg'))
        
        if is_preprocessed:
            # Tampilkan informasi dengan utilitas standar
            update_status_panel(
                ui_components,
                "success",
                f"{ICONS['success']} Dataset preprocessed sudah tersedia di: {preprocessed_dir}"
            )
            
            # Tampilkan tombol cleanup
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'block'
                
            if logger: logger.info(f"{ICONS['folder']} Dataset preprocessed terdeteksi di: {preprocessed_dir}")
        
        # Store data directory di ui_components
        ui_components.update({
            'data_dir': data_dir,
            'preprocessed_dir': preprocessed_dir
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