"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
import yaml
import copy
from pathlib import Path
from IPython.display import display
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_PREPROCESSED_DIR, DEFAULT_INVALID_DIR, DEFAULT_IMG_SIZE

def setup_preprocessing_config_handler(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi preprocessing dengan persistensi yang ditingkatkan.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Load konfigurasi jika belum tersedia
    if config is None:
        config = load_preprocessing_config(ui_components=ui_components)
    
    # Update UI dari konfigurasi
    ui_components = update_ui_from_config(ui_components, config)
    
    # Handler untuk tombol save config
    def on_save_config(b):
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        # Update config dari UI dan simpan
        updated_config = update_config_from_ui(ui_components, ui_components.get('config', config))
        success = save_preprocessing_config(updated_config)
        
        # Simpan kembali config yang diupdate ke ui_components
        ui_components['config'] = updated_config
        
        # Tampilkan status
        status_type = 'success' if success else 'error'
        message = f"{ICONS['success' if success else 'error']} Konfigurasi {'berhasil' if success else 'gagal'} disimpan"
        
        # Update status
        with ui_components['status']: 
            display(create_status_indicator(status_type, message))
            
        # Update status panel
        from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel
        update_status_panel(ui_components, status_type, message)
        
        # Log
        if logger: 
            log_method = logger.success if success else logger.error
            log_method(message)
    
    # Register handler untuk tombol save
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(on_save_config)
    
    # Tambahkan referensi fungsi ke UI components
    ui_components.update({
        'update_config_from_ui': update_config_from_ui,
        'save_preprocessing_config': save_preprocessing_config,
        'load_preprocessing_config': load_preprocessing_config,
        'update_ui_from_config': update_ui_from_config,
        'on_save_config': on_save_config,
        'config': config  # Simpan referensi config di ui_components
    })
    
    return ui_components

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Ekstrak dan update konfigurasi dari UI dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary konfigurasi yang diupdate
    """
    # Inisialisasi config dengan deep copy untuk mencegah modifikasi tidak sengaja
    config = copy.deepcopy(config or {})
    logger = ui_components.get('logger')
    
    # Pastikan section preprocessing dan data ada
    if 'preprocessing' not in config: config['preprocessing'] = {}
    if 'data' not in config: config['data'] = {}
    
    # Ekstrak paths dari ui_components
    data_dir = ui_components.get('data_dir', 'data')
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    
    # Update paths dalam config
    config['data']['dir'] = data_dir
    config['preprocessing']['output_dir'] = preprocessed_dir
    
    # Ekstrak nilai dari preprocess_options
    options = ui_components.get('preprocess_options', {})
    if hasattr(options, 'children') and len(options.children) >= 5:
        # Ekstrak semua nilai dengan list comprehension
        values = [child.value for child in options.children[:5]] 
        img_size, normalize, preserve_ratio, enable_cache, num_workers = values
        
        # Update konfigurasi preprocessing
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
    
    # Ekstrak nilai validation
    validation = ui_components.get('validation_options')
    if hasattr(validation, 'children') and len(validation.children) >= 4:
        # Ekstrak nilai dengan list comprehension
        values = [child.value for child in validation.children[:4]]
        validate_enabled, fix_issues, move_invalid, invalid_dir = values
        
        # Update konfigurasi validation
        config['preprocessing']['validate'] = {
            'enabled': validate_enabled,
            'fix_issues': fix_issues,
            'move_invalid': move_invalid,
            'invalid_dir': invalid_dir
        }
    
    # Ekstrak split selector
    split_selector = ui_components.get('split_selector')
    if hasattr(split_selector, 'value'):
        # Map nilai UI ke konfigurasi dengan dictionary
        split_map = {
            'All Splits': DEFAULT_SPLITS,
            'Train Only': ['train'],
            'Validation Only': ['valid'],
            'Test Only': ['test']
        }
        config['preprocessing']['splits'] = split_map.get(split_selector.value, DEFAULT_SPLITS)
    
    # Simpan referensi config di ui_components untuk memastikan persistensi
    ui_components['config'] = config
    
    if logger: logger.debug(f"🔄 Konfigurasi preprocessing berhasil diupdate dari UI")
    
    return config

def save_preprocessing_config(config: Dict[str, Any], config_path: str = "configs/preprocessing_config.yaml") -> bool:
    """
    Simpan konfigurasi preprocessing dengan penanganan persistensi yang lebih baik.
    
    Args:
        config: Konfigurasi aplikasi
        config_path: Path file konfigurasi
        
    Returns:
        Boolean status keberhasilan
    """
    logger = None
    try:
        # Ambil logger dari lingkungan jika tersedia
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("preprocessing_config")
        except ImportError:
            pass
        
        # Buat deep copy untuk mencegah modifikasi tidak sengaja
        save_config = copy.deepcopy(config)
        
        # Pastikan direktori config ada
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Cek jika file sudah ada, baca dulu untuk mempertahankan konfigurasi lain
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f) or {}
                
            # Merge existing config dengan config baru, prioritaskan config baru
            merged_config = copy.deepcopy(existing_config)
            
            # Update preprocessing section dengan deep merge
            if 'preprocessing' in save_config:
                if 'preprocessing' not in merged_config:
                    merged_config['preprocessing'] = {}
                merged_config['preprocessing'].update(save_config['preprocessing'])
            
            # Update data section jika ada
            if 'data' in save_config:
                if 'data' not in merged_config:
                    merged_config['data'] = {}
                merged_config['data'].update(save_config['data'])
                    
            # Gunakan config yang sudah di-merge
            save_config = merged_config
        
        # Simpan ke file dengan YAML
        with open(config_path, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False)
        
        # Coba sync dengan drive jika tersedia
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Cek apakah path sama dengan realpath untuk mencegah error pada symlink
                if os.path.realpath(config_path) == os.path.realpath(drive_config_path):
                    if logger: logger.info(f"🔄 File lokal dan drive identik: {config_path}, melewati salinan")
                else:
                    # Buat direktori jika belum ada
                    os.makedirs(Path(drive_config_path).parent, exist_ok=True)
                    
                    # Salin file ke Google Drive
                    with open(drive_config_path, 'w') as f:
                        yaml.dump(save_config, f, default_flow_style=False)
                    if logger: logger.info(f"📤 Konfigurasi disimpan ke drive: {drive_config_path}")
        except (ImportError, AttributeError) as e:
            if logger: logger.debug(f"ℹ️ Tidak dapat menyalin ke drive: {str(e)}")
            
        return True
    except Exception as e:
        error_msg = f"Error menyimpan konfigurasi: {str(e)}"
        if logger: 
            logger.error(f"❌ {error_msg}")
        else:
            print(f"❌ {error_msg}")
        return False

def load_preprocessing_config(config_path: str = "configs/preprocessing_config.yaml", ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load konfigurasi preprocessing dengan persistensi yang disempurnakan.
    
    Args:
        config_path: Path file konfigurasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi
    """
    logger = None
    try:
        # Ambil logger dari lingkungan atau ui_components
        if ui_components and 'logger' in ui_components:
            logger = ui_components['logger']
        else:
            try:
                from smartcash.common.logger import get_logger
                logger = get_logger("preprocessing_config")
            except ImportError:
                pass
            
        # Cek apakah ada config tersimpan di ui_components
        if ui_components and 'config' in ui_components and ui_components['config']:
            if logger: logger.info("ℹ️ Menggunakan konfigurasi dari UI components")
            return ui_components['config']
            
        # Coba load dari Google Drive terlebih dahulu
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Cek apakah path sama dengan realpath untuk mencegah error symlink
                if os.path.realpath(config_path) == os.path.realpath(drive_config_path):
                    if logger: logger.info(f"🔄 File lokal dan drive identik: {config_path}, menggunakan lokal")
                elif os.path.exists(drive_config_path):
                    # Baca langsung dari file drive untuk mendapatkan versi terbaru
                    with open(drive_config_path, 'r') as f:
                        drive_config = yaml.safe_load(f)
                        
                    if drive_config:
                        # Salin juga ke lokal untuk digunakan sebagai cache
                        os.makedirs(Path(config_path).parent, exist_ok=True)
                        with open(config_path, 'w') as f:
                            yaml.dump(drive_config, f, default_flow_style=False)
                            
                        if logger: logger.info(f"📥 Konfigurasi dimuat dari drive: {drive_config_path}")
                        
                        # Simpan ke ui_components jika tersedia
                        if ui_components: ui_components['config'] = drive_config
                        return drive_config
        except (ImportError, AttributeError) as e:
            if logger: logger.debug(f"ℹ️ Tidak dapat memuat dari drive: {str(e)}")
        
        # Load dari ConfigManager jika tersedia untuk konsistensi
        try:
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            # Paksa reload untuk mendapatkan data terbaru
            full_config = config_manager.load_config(config_path)
            
            if full_config and ('preprocessing' in full_config or 'data' in full_config):
                if logger: logger.info(f"✅ Konfigurasi dimuat dari {config_path} via ConfigManager")
                
                # Simpan ke ui_components jika tersedia
                if ui_components: ui_components['config'] = full_config
                return full_config
        except (ImportError, AttributeError):
            pass
        
        # Fallback: Load dari local file
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                if logger: logger.info(f"✅ Konfigurasi dimuat dari {config_path}")
                
                # Verifikasi struktur dasar config ada
                if 'preprocessing' not in config: config['preprocessing'] = {}
                if 'data' not in config: config['data'] = {}
                
                # Simpan ke ui_components jika tersedia
                if ui_components: ui_components['config'] = config
                return config
    except Exception as e:
        if logger: logger.warning(f"⚠️ Error saat memuat konfigurasi: {str(e)}")
    
    # Default config jika tidak ada file
    default_config = {
        "preprocessing": {
            "output_dir": DEFAULT_PREPROCESSED_DIR,
            "img_size": DEFAULT_IMG_SIZE,
            "enabled": True,
            "num_workers": 4,
            "normalization": {
                "enabled": True,
                "preserve_aspect_ratio": True,
                "target_size": DEFAULT_IMG_SIZE
            },
            "validate": {
                "enabled": True,
                "fix_issues": True,
                "move_invalid": True,
                "invalid_dir": DEFAULT_INVALID_DIR
            },
            "splits": DEFAULT_SPLITS
        },
        "data": {
            "dir": "data"
        }
    }
    
    if logger: logger.info("ℹ️ Menggunakan konfigurasi default")
    
    # Simpan default config ke ui_components jika tersedia
    if ui_components: ui_components['config'] = default_config
    
    return default_config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update komponen UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Update data paths dan simpan di UI components
    data_dir = config.get('data', {}).get('dir', 'data')
    preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
    ui_components.update({'data_dir': data_dir, 'preprocessed_dir': preprocessed_dir})
    
    # Update preprocess_options dengan nilai dari config
    preproc_config = config.get('preprocessing', {})
    preproc_options = ui_components.get('preprocess_options')
    
    if preproc_options and hasattr(preproc_options, 'children'):
        # Update opsi dengan list comprehension
        if len(preproc_options.children) >= 5:
            # Ambil komponen pertama dari array img_size jika tuple/list
            img_size = preproc_config.get('img_size', DEFAULT_IMG_SIZE)
            img_size_value = img_size[0] if isinstance(img_size, (list, tuple)) else img_size
            
            # Definisi pasangan field dan nilai dari config
            option_values = [
                img_size_value,  # Image size (integer)
                preproc_config.get('normalization', {}).get('enabled', True),  # Normalize
                preproc_config.get('normalization', {}).get('preserve_aspect_ratio', True),  # Aspect ratio
                preproc_config.get('enabled', True),  # Cache enabled
                preproc_config.get('num_workers', 4)  # Workers
            ]
            
            # Update nilai options satu per satu untuk menghindari error
            for i, value in enumerate(option_values):
                if i < len(preproc_options.children):
                    preproc_options.children[i].value = value
    
    # Update validation options dengan data dari config
    validation_options = ui_components.get('validation_options')
    validation_config = preproc_config.get('validate', {})
    
    if validation_options and hasattr(validation_options, 'children'):
        if len(validation_options.children) >= 4:
            # Definisi pasangan field dan nilai dari config
            validation_values = [
                validation_config.get('enabled', True),  # Enable validation
                validation_config.get('fix_issues', True),  # Fix issues
                validation_config.get('move_invalid', True),  # Move invalid
                validation_config.get('invalid_dir', 'data/invalid')  # Invalid dir
            ]
            
            # Update nilai options satu per satu
            for i, value in enumerate(validation_values):
                if i < len(validation_options.children):
                    validation_options.children[i].value = value
    
    # Update split selector
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value'):
        # Map dari config value ke UI value
        split_list = preproc_config.get('splits', DEFAULT_SPLITS)
        split_str = str(sorted(split_list))
        
        split_map = {
            str(sorted(DEFAULT_SPLITS)): 'All Splits',
            str([DEFAULT_SPLITS[0]]): 'Train Only',
            str(['valid']): 'Validation Only',
            str(['test']): 'Test Only'
        }
        
        split_selector.value = split_map.get(split_str, 'All Splits')
    
    # Simpan referensi config di ui_components untuk persistensi
    ui_components['config'] = config
    
    if logger: logger.debug(f"🔄 UI berhasil diupdate dari konfigurasi")
    
    return ui_components