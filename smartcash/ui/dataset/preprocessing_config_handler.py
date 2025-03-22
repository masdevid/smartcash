"""
File: smartcash/ui/dataset/preprocessing_config_handler.py
Deskripsi: Handler untuk konfigurasi preprocessing dataset dengan integrasi config_manager dan config_sync
"""

from typing import Dict, Any, Optional
import os
import yaml
from pathlib import Path
from IPython.display import display

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Update konfigurasi preprocessing dari nilai UI dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi existing
        
    Returns:
        Dictionary konfigurasi yang telah diupdate
    """
    # Inisialisasi config jika None
    config = config or {}
    
    # Pastikan section preprocessing ada
    if 'preprocessing' not in config: config['preprocessing'] = {}
    
    # Konsolidasi path handling dengan satu fungsi untuk konversi path
    def get_relative_path(path: str) -> str:
        """Convert path ke relative jika absolut"""
        return os.path.relpath(path, os.getcwd()) if os.path.isabs(path) else path
    
    # Ekstrak dan update paths dengan single-line assignment
    data_dir = get_relative_path(ui_components.get('data_dir', 'data'))
    preprocessed_dir = get_relative_path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
    
    # Update paths dalam config dengan destructuring
    if 'data' not in config: config['data'] = {}
    config['data']['dir'] = data_dir
    config['preprocessing']['output_dir'] = preprocessed_dir
    
    # Konsolidasi ekstraksi nilai komponen UI dengan error checking
    preproc_options = ui_components.get('preprocess_options', {})
    if not hasattr(preproc_options, 'children') or len(preproc_options.children) < 5: return config
    
    # Extract semua nilai dengan one-liner per value
    img_size, normalize, preserve_ratio, enable_cache, num_workers = [
        preproc_options.children[i].value for i in range(5)
    ]
    
    # Update konfigurasi dengan nilai tersebut
    config['preprocessing'].update({
        'img_size': [img_size, img_size],
        'enabled': enable_cache,
        'num_workers': num_workers,
        'normalization': {
            'enabled': normalize,
            'preserve_aspect_ratio': preserve_ratio,
            'target_size': [img_size, img_size]
        }
    })
    
    # Ekstrak dan set validasi dengan destructuring
    validation_options = ui_components.get('validation_options')
    if hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
        validate_enabled, fix_issues, move_invalid, invalid_dir = [
            validation_options.children[i].value for i in range(4)
        ]
        
        # Update validate config dengan inline dictionary
        config['preprocessing']['validate'] = {
            'enabled': validate_enabled,
            'fix_issues': fix_issues,
            'move_invalid': move_invalid,
            'invalid_dir': get_relative_path(invalid_dir)
        }
    
    # Ekstrak split selector dengan map nilai
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value'):
        # Map nilai UI ke konfigurasi dengan dictionary
        split_map = {
            'All Splits': ['train', 'valid', 'test'],
            'Train Only': ['train'],
            'Validation Only': ['valid'],
            'Test Only': ['test']
        }
        config['preprocessing']['splits'] = split_map.get(split_selector.value, ['train', 'valid', 'test'])
    
    return config

def save_preprocessing_config(config: Dict[str, Any], config_path: str = "configs/preprocessing_config.yaml") -> bool:
    """
    Simpan konfigurasi preprocessing ke file dengan integrasi config_manager dan config_sync.
    
    Args:
        config: Konfigurasi yang akan disimpan
        config_path: Path file konfigurasi
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    # Gunakan config_manager dan config_sync jika tersedia
    try:
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        
        # Simpan config ke config_manager dan ke file
        config_manager.config.update(config)
        config_manager.save_config(config_path, create_dirs=True)
        
        # Sinkronisasi ke drive jika di Colab
        try:
            from smartcash.common.config_sync import sync_config_with_drive
            from smartcash.common.environment import get_environment_manager
            
            env_manager = get_environment_manager()
            if env_manager.is_drive_mounted:
                # Sinkronisasi konfigurasi ke drive dengan lebih sedikit parameter
                sync_config_with_drive(config_path, sync_strategy='local_priority')
        except (ImportError, AttributeError):
            pass
            
        return True
    except ImportError:
        # Fallback: simpan langsung ke file
        try:
            Path(config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            return True
        except Exception:
            return False

def load_preprocessing_config(config_path: str = "configs/preprocessing_config.yaml") -> Dict[str, Any]:
    """
    Load konfigurasi preprocessing dari file dengan integrasi config_manager dan config_sync.
    
    Args:
        config_path: Path file konfigurasi
        
    Returns:
        Dictionary konfigurasi
    """
    # Coba gunakan config_manager dengan integasi config_sync
    try:
        from smartcash.common.config import get_config_manager
        from smartcash.common.environment import get_environment_manager
        
        # Dapatkan config_manager dan environment manager
        config_manager = get_config_manager()
        env_manager = get_environment_manager()
        
        # Coba sinkronisasi dari drive terlebih dahulu jika di Colab
        try:
            if env_manager.is_drive_mounted:
                from smartcash.common.config_sync import sync_config_with_drive
                # Sinkronisasi config dari drive dengan prioritas drive
                sync_config_with_drive(config_path, sync_strategy='drive_priority')
        except (ImportError, AttributeError):
            pass
            
        # Periksa apakah config sudah dimuat di config_manager
        if config_manager.config: return config_manager.config
        
        # Load config dari file
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
        
    # Default config jika semua cara gagal
    return {
        "preprocessing": {
            "output_dir": "data/preprocessed",
            "img_size": [640, 640],
            "enabled": True,
            "num_workers": 4,
            "normalization": {
                "enabled": True,
                "preserve_aspect_ratio": True,
                "target_size": [640, 640]
            },
            "validate": {
                "enabled": True,
                "fix_issues": True,
                "move_invalid": True,
                "invalid_dir": "data/invalid"
            },
            "splits": ["train", "valid", "test"]
        }
    }

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update komponen UI dari konfigurasi dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi
        
    Returns:
        Dictionary UI yang telah diupdate
    """
    # Ekstrak path dengan default
    data_dir = config.get('data', {}).get('dir', 'data')
    preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
    
    # Update path input dengan single-line assignments
    if 'path_input' in ui_components: ui_components['path_input'].value = data_dir
    if 'preprocessed_input' in ui_components: ui_components['preprocessed_input'].value = preprocessed_dir
    
    # Update path info dengan path absolute menggunakan f-string DRY
    from smartcash.ui.utils.constants import COLORS
    abs_data_dir, abs_preprocessed_dir = map(os.path.abspath, [data_dir, preprocessed_dir])
    
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
    ui_components.update({'data_dir': data_dir, 'preprocessed_dir': preprocessed_dir})
    
    # Konsolidasi update UI options dengan satu loop
    preproc_config = config.get('preprocessing', {})
    preproc_options = ui_components.get('preprocess_options')
    
    if preproc_options and hasattr(preproc_options, 'children'):
        # Daftar pasangan (indeks, nilai, fungsi_konversi)
        option_updates = [
            (0, preproc_config.get('img_size', [640, 640])[0], None),  # Image size
            (1, preproc_config.get('normalization', {}).get('enabled', True), None),  # Normalize
            (2, preproc_config.get('normalization', {}).get('preserve_aspect_ratio', True), None),  # Aspect ratio
            (3, preproc_config.get('enabled', True), None),  # Cache enabled
            (4, preproc_config.get('num_workers', 4), None)  # Workers
        ]
        
        # Update setiap opsi dengan batasan ukuran children
        for i, value, converter in option_updates:
            if i < len(preproc_options.children):
                preproc_options.children[i].value = converter(value) if converter else value
    
    # Update validation options dengan konsolidasi
    validation_options = ui_components.get('validation_options')
    if validation_options and hasattr(validation_options, 'children') and 'validate' in preproc_config:
        val_config = preproc_config['validate']
        # List update untuk validasi dengan index < len() check dalam single line
        [(validation_options.children[i].value if i < len(validation_options.children) else None) for i, v in 
         enumerate([val_config.get('enabled', True), val_config.get('fix_issues', True), 
                   val_config.get('move_invalid', True), val_config.get('invalid_dir', 'data/invalid')])]
    
    # Update split selector dengan pendekatan map/filter/reduce
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value') and 'splits' in preproc_config:
        # Konversi list split ke nilai UI dengan dictionary
        split_value_map = {
            str(sorted(['train', 'valid', 'test'])): 'All Splits',
            str(['train']): 'Train Only',
            str(['valid']): 'Validation Only',
            str(['test']): 'Test Only'
        }
        # Dapatkan nilai UI berdasarkan split atau default ke 'All Splits'
        split_selector.value = split_value_map.get(str(sorted(preproc_config['splits'])), 'All Splits')
    
    return ui_components

def setup_preprocessing_config_handler(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler konfigurasi preprocessing dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Load config jika belum tersedia
    config = config or load_preprocessing_config()
    
    # Update UI dari config
    ui_components = update_ui_from_config(ui_components, config)
    
    # Handler untuk tombol save dengan closure
    def on_save_config(b):
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
        
        # Update config dari UI dan simpan dalam satu operasi
        updated_config, success = update_config_from_ui(ui_components, config), False
        try:
            success = save_preprocessing_config(updated_config)
            status, icon = ('success', ICONS['success']) if success else ('error', ICONS['error'])
            message = f"{icon} Konfigurasi {'berhasil' if success else 'gagal'} disimpan"
            
            # Update status panel dan log dalam satu template message
            with ui_components.get('status'): display(create_status_indicator(status, message))
            update_status_panel(ui_components, status, message)
            if logger: logger.info(message)
        except Exception as e:
            error_msg = f"{ICONS['error']} Error menyimpan konfigurasi: {str(e)}"
            with ui_components.get('status'): display(create_status_indicator('error', error_msg))
            if logger: logger.error(error_msg)
    
    # Tambahkan handler ke tombol jika tersedia
    if 'save_button' in ui_components: ui_components['save_button'].on_click(on_save_config)
    
    # Tambahkan semua fungsi ke ui_components dalam satu assignment
    ui_components.update({
        'update_config_from_ui': update_config_from_ui,
        'save_preprocessing_config': save_preprocessing_config,
        'load_preprocessing_config': load_preprocessing_config,
        'update_ui_from_config': update_ui_from_config,
        'on_save_config': on_save_config
    })
    
    return ui_components