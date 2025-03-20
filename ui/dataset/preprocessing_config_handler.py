"""
File: smartcash/ui/dataset/preprocessing_config_handler.py
Deskripsi: Handler untuk konfigurasi preprocessing dataset dengan penanganan path relatif
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from IPython.display import display

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Update konfigurasi preprocessing dari nilai UI dengan validasi path relatif.
    
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
    
    # Update paths dengan path relatif
    data_dir = ui_components.get('data_dir', 'data')
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    
    # Convert to relative if absolute
    if os.path.isabs(data_dir):
        rel_data_dir = os.path.relpath(data_dir, os.getcwd())
    else:
        rel_data_dir = data_dir
        
    if os.path.isabs(preprocessed_dir):
        rel_preprocessed_dir = os.path.relpath(preprocessed_dir, os.getcwd())
    else:
        rel_preprocessed_dir = preprocessed_dir
    
    # Update paths dalam config
    if 'data' not in config:
        config['data'] = {}
    config['data']['dir'] = rel_data_dir
    config['preprocessing']['output_dir'] = rel_preprocessed_dir
    
    preproc_config = config['preprocessing']
    
    # Dapatkan nilai dari komponen UI dengan pendekatan konsolidasi
    preproc_options = ui_components.get('preprocess_options')
    if preproc_options and hasattr(preproc_options, 'children') and len(preproc_options.children) >= 5:
        # Extract nilai opsi dengan validasi
        img_size = preproc_options.children[0].value
        normalize = preproc_options.children[1].value
        preserve_ratio = preproc_options.children[2].value
        enable_cache = preproc_options.children[3].value
        num_workers = preproc_options.children[4].value
        
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
    if validation_options and hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
        # Extract nilai validasi dengan inline extraction
        validate_enabled = validation_options.children[0].value
        fix_issues = validation_options.children[1].value
        move_invalid = validation_options.children[2].value
        invalid_dir = validation_options.children[3].value
        
        # Convert invalid_dir to relative path
        if os.path.isabs(invalid_dir):
            rel_invalid_dir = os.path.relpath(invalid_dir, os.getcwd())
        else:
            rel_invalid_dir = invalid_dir
        
        # Pastikan section validate ada
        if 'validate' not in preproc_config:
            preproc_config['validate'] = {}
            
        preproc_config['validate']['enabled'] = validate_enabled
        preproc_config['validate']['fix_issues'] = fix_issues
        preproc_config['validate']['move_invalid'] = move_invalid
        preproc_config['validate']['invalid_dir'] = rel_invalid_dir
    
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
            # Simpan config
            config_manager.config.update(config)
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
            try:
                return config_manager.load_config(config_path)
            except FileNotFoundError:
                pass
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
    # Ambil path dan update UI
    data_dir = config.get('data', {}).get('dir', 'data')
    preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
    
    # Update path inputs
    if 'path_input' in ui_components:
        ui_components['path_input'].value = data_dir
    
    if 'preprocessed_input' in ui_components:
        ui_components['preprocessed_input'].value = preprocessed_dir
    
    # Update path info dengan absolute path
    from smartcash.ui.utils.constants import COLORS
    abs_data_dir = os.path.abspath(data_dir)
    abs_preprocessed_dir = os.path.abspath(preprocessed_dir)
    
    if 'path_info' in ui_components:
        ui_components['path_info'].value = f"""
        <div style="padding:10px; margin:10px 0; background-color:{COLORS['light']}; 
                border-radius:5px; border-left:4px solid {COLORS['primary']}; color: black;">
            <h4 style="color:inherit; margin-top:0;">📂 Lokasi Dataset</h4>
            <p><strong>Data Source:</strong> <code>{abs_data_dir}</code></p>
            <p><strong>Preprocessed:</strong> <code>{abs_preprocessed_dir}</code></p>
        </div>
        """
    
    # Simpan path di ui_components
    ui_components['data_dir'] = data_dir
    ui_components['preprocessed_dir'] = preprocessed_dir
    
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
        if 'normalization' in preproc_config and len(preproc_options.children) > 1:
            norm_config = preproc_config['normalization']
            # Update normalization enabled dan preserve aspect ratio dengan one-liner
            if len(preproc_options.children) > 1: preproc_options.children[1].value = norm_config.get('enabled', True)
            if len(preproc_options.children) > 2: preproc_options.children[2].value = norm_config.get('preserve_aspect_ratio', True)
        
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
        if len(validation_options.children) > 0: validation_options.children[0].value = val_config.get('enabled', True)
        if len(validation_options.children) > 1: validation_options.children[1].value = val_config.get('fix_issues', True)
        if len(validation_options.children) > 2: validation_options.children[2].value = val_config.get('move_invalid', True)
        if len(validation_options.children) > 3: validation_options.children[3].value = val_config.get('invalid_dir', 'data/invalid')
    
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
    
    # Load config jika belum tersedia
    if not config:
        config = load_preprocessing_config()
    
    # Update UI dari config
    ui_components = update_ui_from_config(ui_components, config)
    
    # Tambahkan handler untuk tombol save
    if 'save_button' in ui_components:
        def on_save_config(b):
            from smartcash.ui.utils.alert_utils import create_status_indicator
            from smartcash.ui.utils.constants import ICONS
            
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
            from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
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
        'update_ui_from_config': update_ui_from_config
    })
    
    return ui_components