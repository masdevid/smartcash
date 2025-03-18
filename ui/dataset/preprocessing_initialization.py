"""
File: smartcash/ui/dataset/preprocessing_initialization.py
Deskripsi: Inisialisasi komponen untuk preprocessing dataset dengan pendekatan minimalis
"""

from typing import Dict, Any
from IPython.display import display, HTML
from smartcash.ui.utils.constants import COLORS, ICONS, ALERT_STYLES

def setup_initialization(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Inisialisasi komponen preprocessing dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    try:
        # Inisialisasi data directory
        data_dir = config.get('data', {}).get('dir', 'data')
        preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        
        # Gunakan Google Drive jika tersedia
        if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
            data_dir = str(env.drive_path / 'data')
            preprocessed_dir = str(env.drive_path / 'data/preprocessed')
            
            # Log penggunaan Google Drive
            if logger: logger.info(f"{ICONS['folder']} Menggunakan Google Drive untuk penyimpanan dataset: {data_dir}")
                
            # Update status panel
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
                # Update komponen UI dengan nilai dari config
                children = preproc_options.children
                
                # Update image size
                if 'img_size' in preproc_config and len(children) > 0:
                    img_size = preproc_config['img_size']
                    if isinstance(img_size, list) and len(img_size) > 0:
                        children[0].value = img_size[0]
                
                # Update normalization
                if 'normalization' in preproc_config and len(children) > 1:
                    children[1].value = preproc_config['normalization'].get('enabled', True)
                
                # Update preserve aspect ratio
                if 'normalization' in preproc_config and len(children) > 2:
                    children[2].value = preproc_config['normalization'].get('preserve_aspect_ratio', True)
                
                # Update cache
                if 'enabled' in preproc_config and len(children) > 3:
                    children[3].value = preproc_config['enabled']
                
                # Update workers
                if 'num_workers' in preproc_config and len(children) > 4:
                    children[4].value = preproc_config['num_workers']
            
            # Update validation options
            val_options = ui_components.get('validation_options')
            if val_options and hasattr(val_options, 'children') and 'validate' in preproc_config:
                val_config = preproc_config['validate']
                children = val_options.children
                
                # Update validation options
                if len(children) > 0 and 'enabled' in val_config:
                    children[0].value = val_config['enabled']
                
                if len(children) > 1 and 'fix_issues' in val_config:
                    children[1].value = val_config['fix_issues']
                
                if len(children) > 2 and 'move_invalid' in val_config:
                    children[2].value = val_config['move_invalid']
        
        # Cek status preprocessed data yang sudah ada
        from pathlib import Path
        
        # Cek apakah sudah ada hasil preprocessing
        preprocessed_path = Path(preprocessed_dir)
        if preprocessed_path.exists():
            all_splits_exist = True
            
            for split in ['train', 'valid', 'test']:
                split_path = preprocessed_path / split
                if not split_path.exists() or not any(split_path.iterdir()):
                    all_splits_exist = False
                    break
            
            if all_splits_exist:
                # Tampilkan informasi bahwa data preprocessed sudah ada
                update_status_panel(
                    ui_components,
                    "success",
                    f"{ICONS['success']} Dataset preprocessed sudah tersedia di: {preprocessed_dir}"
                )
                
                # Tampilkan tombol cleanup
                if 'cleanup_button' in ui_components:
                    ui_components['cleanup_button'].layout.display = 'block'
                    
                if logger: logger.info(f"{ICONS['folder']} Dataset preprocessed terdeteksi di: {preprocessed_dir}")
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Inisialisasi preprocessing: {str(e)}")
    
    # Store data directory and preprocessed directory in ui_components
    ui_components['data_dir'] = data_dir
    ui_components['preprocessed_dir'] = preprocessed_dir
    
    return ui_components

def update_status_panel(ui_components, status_type, message):
    """
    Update status panel dengan pesan dan jenis status.
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Jenis status ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    try:
        from smartcash.ui.utils.ui_helpers import create_info_alert
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
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