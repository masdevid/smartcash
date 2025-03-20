"""
File: smartcash/ui/dataset/preprocessing_config_handler.py
Deskripsi: Handler untuk konfigurasi preprocessing dataset dengan integrasi path yang lebih baik
"""

from typing import Dict, Any
import os
import yaml
from pathlib import Path
from IPython.display import display

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Update konfigurasi preprocessing dari nilai UI dengan validasi path.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi existing
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    if config is None:
        config = {}
    
    # Pastikan section preprocessing ada
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    preproc_config = config['preprocessing']
    
    # Dapatkan nilai dari komponen UI dengan pendekatan konsolidasi
    preproc_options = ui_components.get('preprocess_options')
    if preproc_options and hasattr(preproc_options, 'children'):
        # Extract semua nilai opsi dengan validasi menggunakan inline extraction
        img_size = preproc_options.children[0].value if len(preproc_options.children) > 0 and hasattr(preproc_options.children[0], 'value') else 640
        normalize = preproc_options.children[1].value if len(preproc_options.children) > 1 and hasattr(preproc_options.children[1], 'value') else True
        preserve_ratio = preproc_options.children[2].value if len(preproc_options.children) > 2 and hasattr(preproc_options.children[2], 'value') else True
        enable_cache = preproc_options.children[3].value if len(preproc_options.children) > 3 and hasattr(preproc_options.children[3], 'value') else True
        num_workers = preproc_options.children[4].value if len(preproc_options.children) > 4 and hasattr(preproc_options.children[4], 'value') else 4
        
        # Update konfigurasi dengan nilai yang diekstrak
        preproc_config['img_size'] = [img_size, img_size]
        preproc_config['enabled'] = enable_cache
        preproc_config['num_workers'] = num_workers
        
        # Pastikan section normalization ada
        if 'normalization' not in preproc_config:
            preproc_config['normalization'] = {}
            
        preproc_config['normalization']['enabled'] = normalize
        preproc_config['normalization']['preserve_aspect_ratio'] = preserve_ratio
    
    # Dapatkan pengaturan validasi
    validation_options = ui_components.get('validation_options')
    if validation_options and hasattr(validation_options, 'children'):
        # Extract semua nilai validasi dengan one-liner untuk efisiensi
        validate_enabled = validation_options.children[0].value if len(validation_options.children) > 0 and hasattr(validation_options.children[0], 'value') else True
        fix_issues = validation_options.children[1].value if len(validation_options.children) > 1 and hasattr(validation_options.children[1], 'value') else True
        move_invalid = validation_options.children[2].value if len(validation_options.children) > 2 and hasattr(validation_options.children[2], 'value') else True
        invalid_dir = validation_options.children[3].value if len(validation_options.children) > 3 and hasattr(validation_options.children[3], 'value') else 'data/invalid'
        
        # Pastikan section validate ada
        if 'validate' not in preproc_config:
            preproc_config['validate'] = {}
            
        preproc_config['validate']['enabled'] = validate_enabled
        preproc_config['validate']['fix_issues'] = fix_issues
        preproc_config['validate']['move_invalid'] = move_invalid
        preproc_config['validate']['invalid_dir'] = invalid_dir
    
    # PENTING: Update path dari UI components, bukan dari widget input langsung
    # Karena path di UI components sudah dalam format absolute yang benar
    data_dir = ui_components.get('data_dir')
    preprocessed_dir = ui_components.get('preprocessed_dir')
    
    # Update data_dir di config
    if data_dir:
        # Simpan sebagai path relatif jika path absolute diawali dengan Drive path
        drive_mounted = False
        drive_path = None
        
        # Coba dapatkan drive path untuk konversi ke relatif
        try:
            from smartcash.ui.utils.drive_utils import detect_drive_mount
            drive_mounted, drive_path = detect_drive_mount()
        except ImportError:
            pass
            
        # Konversi ke path relatif jika diperlukan
        rel_data_dir = data_dir
        if drive_mounted and drive_path and data_dir.startswith(f"{drive_path}/SmartCash"):
            rel_data_dir = os.path.relpath(data_dir, f"{drive_path}/SmartCash")
        
        # Update data_dir di config (gunakan path relatif)
        if 'data' not in config:
            config['data'] = {}
        config['data']['dir'] = rel_data_dir
    
    # Update preprocessed_dir di config
    if preprocessed_dir:
        # Konversi ke path relatif jika diperlukan
        rel_preprocessed_dir = preprocessed_dir
        if drive_mounted and drive_path and preprocessed_dir.startswith(f"{drive_path}/SmartCash"):
            rel_preprocessed_dir = os.path.relpath(preprocessed_dir, f"{drive_path}/SmartCash")
        
        # Update preprocessed_dir di config (gunakan path relatif)
        preproc_config['output_dir'] = rel_preprocessed_dir
    
    # Ekstrak split selector jika tersedia
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value'):
        split_value = split_selector.value
        # Translate nilai UI ke konfigurasi
        if split_value == 'All Splits':
            preproc_config['splits'] = ['train', 'valid', 'test']
        elif split_value == 'Train Only':
            preproc_config['splits'] = ['train']
        elif split_value == 'Validation Only':
            preproc_config['splits'] = ['valid']
        elif split_value == 'Test Only':
            preproc_config['splits'] = ['test']
    
    return config

def save_preprocessing_config(config: Dict[str, Any], config_path: str = "configs/preprocessing_config.yaml") -> bool:
    """
    Simpan konfigurasi preprocessing ke file dengan integrasi utilitas standar.
    
    Args:
        config: Konfigurasi yang akan disimpan
        config_path: Path file konfigurasi
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    # Coba gunakan utilitas standar config_manager
    try:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        if config_manager:
            # Simpan konfigurasi menggunakan config_manager
            config_manager.config = config  # Pastikan config manager memiliki konfigurasi terbaru
            config_manager.save_config(config_path, create_dirs=True)
            return True
    except ImportError:
        pass
    
    # Fallback: simpan langsung ke file
    try:
        path = Path(config_path)
        
        # Pastikan direktori ada
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception:
        return False

def load_preprocessing_config(config_path: str = "configs/preprocessing_config.yaml") -> Dict[str, Any]:
    """
    Load konfigurasi preprocessing dari file dengan integrasi standar.
    
    Args:
        config_path: Path file konfigurasi
        
    Returns:
        Dictionary konfigurasi
    """
    # Coba gunakan utilitas standar config_manager
    try:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        if config_manager:
            # Periksa apakah config sudah dimuat di config_manager
            if config_manager.config:
                return config_manager.config
            
            # Load config
            return config_manager.load_config(config_path)
    except ImportError:
        pass
    
    # Fallback: load langsung dari file
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
        
    # Default config
    return {
        "preprocessing": {
            "output_dir": "data/preprocessed",
            "img_size": [640, 640],
            "enabled": True,
            "num_workers": 4,
            "normalization": {
                "enabled": True,
                "preserve_aspect_ratio": True
            },
            "validate": {
                "enabled": True,
                "fix_issues": True,
                "move_invalid": True,
                "invalid_dir": "data/invalid"
            }
        }
    }

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update komponen UI dari konfigurasi dengan validasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi
        
    Returns:
        Dictionary UI yang telah diupdate
    """
    # Ekstrak konfigurasi preprocessing
    preproc_config = config.get('preprocessing', {})
    
    # Update opsi preprocessing dengan validasi dan inline extraction
    preproc_options = ui_components.get('preprocess_options')
    if preproc_options and hasattr(preproc_options, 'children'):
        # Update image size dari config
        if 'img_size' in preproc_config and len(preproc_options.children) > 0:
            img_size = preproc_config['img_size']
            if isinstance(img_size, list) and len(img_size) > 0:
                preproc_options.children[0].value = img_size[0]
        
        # Update opsi normalisasi
        if 'normalization' in preproc_config:
            norm_config = preproc_config['normalization']
            # Update normalization enabled
            if 'enabled' in norm_config and len(preproc_options.children) > 1:
                preproc_options.children[1].value = norm_config['enabled']
            # Update preserve aspect ratio
            if 'preserve_aspect_ratio' in norm_config and len(preproc_options.children) > 2:
                preproc_options.children[2].value = norm_config['preserve_aspect_ratio']
        
        # Update cache enabled
        if 'enabled' in preproc_config and len(preproc_options.children) > 3:
            preproc_options.children[3].value = preproc_config['enabled']
        
        # Update workers
        if 'num_workers' in preproc_config and len(preproc_options.children) > 4:
            preproc_options.children[4].value = preproc_config['num_workers']
    
    # Update validation options
    validation_options = ui_components.get('validation_options')
    if validation_options and hasattr(validation_options, 'children') and 'validate' in preproc_config:
        val_config = preproc_config['validate']
        
        # Update validation options dengan validasi
        if 'enabled' in val_config and len(validation_options.children) > 0: validation_options.children[0].value = val_config['enabled']
        if 'fix_issues' in val_config and len(validation_options.children) > 1: validation_options.children[1].value = val_config['fix_issues']
        if 'move_invalid' in val_config and len(validation_options.children) > 2: validation_options.children[2].value = val_config['move_invalid']
        if 'invalid_dir' in val_config and len(validation_options.children) > 3: validation_options.children[3].value = val_config['invalid_dir']
    
    # PENTING: Update komponen path input jika tersedia
    path_input = ui_components.get('path_input')
    preprocessed_input = ui_components.get('preprocessed_input')
    
    # Dapatkan path yang benar (absolute path)
    drive_mounted = False
    drive_path = None
    smartcash_dir = None
    
    # Coba dapatkan informasi drive
    try:
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        drive_mounted, drive_path = detect_drive_mount()
        if drive_mounted and drive_path:
            smartcash_dir = f"{drive_path}/SmartCash"
    except ImportError:
        pass
    
    # Update path input dengan nilai absolute
    if 'data' in config and 'dir' in config['data'] and path_input:
        data_dir = config['data']['dir']
        # Konversi ke absolute path jika relatif dan drive tersedia
        if not os.path.isabs(data_dir) and drive_mounted and smartcash_dir:
            data_dir = os.path.join(smartcash_dir, data_dir)
        path_input.value = data_dir
        ui_components['data_dir'] = data_dir
    
    # Update preprocessed input dengan nilai absolute
    if 'output_dir' in preproc_config and preprocessed_input:
        preprocessed_dir = preproc_config['output_dir']
        # Konversi ke absolute path jika relatif dan drive tersedia
        if not os.path.isabs(preprocessed_dir) and drive_mounted and smartcash_dir:
            preprocessed_dir = os.path.join(smartcash_dir, preprocessed_dir)
        preprocessed_input.value = preprocessed_dir
        ui_components['preprocessed_dir'] = preprocessed_dir
    
    # Update split selector jika ada
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value') and 'splits' in preproc_config:
        splits = preproc_config['splits']
        # Translate konfigurasi ke nilai UI
        if isinstance(splits, list):
            if sorted(splits) == sorted(['train', 'valid', 'test']):
                split_selector.value = 'All Splits'
            elif splits == ['train']:
                split_selector.value = 'Train Only'
            elif splits == ['valid']:
                split_selector.value = 'Validation Only'
            elif splits == ['test']:
                split_selector.value = 'Test Only'
    
    return ui_components

def setup_preprocessing_config_handler(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler konfigurasi preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Load config jika belum tersedia
    if not config:
        config = load_preprocessing_config()
    
    # Update UI dari config
    ui_components = update_ui_from_config(ui_components, config)
    
    # Tambahkan handler untuk tombol save
    if 'save_button' in ui_components:
        def on_save_config(b):
            from smartcash.ui.utils.alert_utils import create_status_indicator
            from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
            
            # Update config dari UI
            updated_config = update_config_from_ui(ui_components, config)
            
            # Simpan config
            success = save_preprocessing_config(updated_config)
            
            # Tampilkan status dengan utility standar
            with ui_components.get('status'):
                display(create_status_indicator(
                    'success' if success else 'error',
                    f"{ICONS['success'] if success else ICONS['error']} Konfigurasi {'berhasil' if success else 'gagal'} disimpan"
                ))
            
            # Update status panel
            update_status_panel(
                ui_components,
                'success' if success else 'error',
                f"{ICONS['success'] if success else ICONS['error']} Konfigurasi {'berhasil' if success else 'gagal'} disimpan"
            )
            
            if logger: logger.info(f"{ICONS['success'] if success else ICONS['error']} Konfigurasi preprocessing {'berhasil' if success else 'gagal'} disimpan")
            
        ui_components['save_button'].on_click(on_save_config)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'update_config_from_ui': update_config_from_ui,
        'save_preprocessing_config': save_preprocessing_config,
        'load_preprocessing_config': load_preprocessing_config,
        'update_ui_from_config': update_ui_from_config,
        'on_save_config': on_save_config
    })
    
    return ui_components