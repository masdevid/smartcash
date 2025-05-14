"""
File: smartcash/ui/dataset/augmentation/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk augmentasi dataset
"""

from typing import Dict, Any, Optional, List
import os
import yaml
import copy
from pathlib import Path
from IPython.display import display
import ipywidgets as widgets

def setup_augmentation_config_handler(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi augmentasi dengan persistensi yang ditingkatkan.
    
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
        config = load_augmentation_config(ui_components=ui_components)
    
    # Update UI dari konfigurasi
    ui_components = update_ui_from_config(ui_components, config)
    
    # Handler untuk tombol save config
    def on_save_config(b):
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        # Update config dari UI dan simpan
        updated_config = update_config_from_ui(ui_components, ui_components.get('config', config))
        success = save_augmentation_config(updated_config)
        
        # Simpan kembali config yang diupdate ke ui_components
        ui_components['config'] = updated_config
        
        # Tampilkan status
        status_type = 'success' if success else 'error'
        message = f"{ICONS['success' if success else 'error']} Konfigurasi {'berhasil' if success else 'gagal'} disimpan"
        
        # Update status
        with ui_components['status']: 
            display(create_status_indicator(status_type, message))
            
        # Update status panel
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel
        update_status_panel(ui_components, status_type, message)
        
        # Log
        if logger: 
            log_method = logger.success if success else logger.error
            log_method(message)
    
    # Register handler untuk tombol save
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(on_save_config)
        
    # Register handler untuk tombol augment (jika ada)
    if 'augment_button' in ui_components:
        ui_components['augment_button'].on_click(lambda b: None)
    
    # Tambahkan referensi fungsi ke UI components
    ui_components.update({
        'update_config_from_ui': update_config_from_ui,
        'save_augmentation_config': save_augmentation_config,
        'load_augmentation_config': load_augmentation_config,
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
    
    # Pastikan section augmentation dan data ada
    if 'augmentation' not in config: config['augmentation'] = {}
    if 'data' not in config: config['data'] = {}
    
    # Ekstrak paths dari ui_components
    data_dir = ui_components.get('data_dir', 'data')
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
    
    # Update paths dalam config
    config['data']['dir'] = data_dir
    config['augmentation']['input_dir'] = preprocessed_dir
    config['augmentation']['output_dir'] = augmented_dir
    
    # Ekstrak nilai dari aug_options
    options = ui_components.get('aug_options', {})
    if hasattr(options, 'children') and len(options.children) >= 7:
        # Ekstrak nilai dengan list comprehension
        aug_types = options.children[0].value  # SelectMultiple
        aug_prefix = options.children[2].value  # Text
        aug_factor = options.children[3].value  # IntSlider
        target_split = options.children[4].value  # Dropdown
        balance_classes = options.children[5].value  # Checkbox
        num_workers = options.children[6].value  # IntSlider
        
        # Update konfigurasi augmentation
        config['augmentation'].update({
            'types': list(aug_types) if aug_types else ['Combined (Recommended)'],
            'prefix': aug_prefix,
            'factor': aug_factor,
            'split': target_split,  # Ubah 'target_split' menjadi 'split' agar konsisten dengan kunci yang digunakan di update_ui_from_config
            'balance_classes': balance_classes,
            'num_workers': num_workers,
            'enabled': True
        })
    
    # Simpan referensi config di ui_components untuk memastikan persistensi
    ui_components['config'] = config
    
    if logger: logger.debug(f"🔄 Konfigurasi augmentasi berhasil diupdate dari UI")
    
    return config

def save_augmentation_config(config: Dict[str, Any], config_path: str = "configs/augmentation_config.yaml") -> bool:
    """
    Simpan konfigurasi augmentasi dengan penanganan persistensi yang lebih baik.
    
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
            logger = get_logger("augmentation_config")
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
            
            # Update augmentation section dengan deep merge
            if 'augmentation' in save_config:
                if 'augmentation' not in merged_config:
                    merged_config['augmentation'] = {}
                merged_config['augmentation'].update(save_config['augmentation'])
            
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

def load_augmentation_config(config_path: str = "configs/augmentation_config.yaml", ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load konfigurasi augmentasi dengan persistensi yang disempurnakan.
    
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
                logger = get_logger("augmentation_config")
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
            
            if full_config and ('augmentation' in full_config or 'data' in full_config):
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
                if 'augmentation' not in config: config['augmentation'] = {}
                if 'data' not in config: config['data'] = {}
                
                # Simpan ke ui_components jika tersedia
                if ui_components: ui_components['config'] = config
                return config
    except Exception as e:
        if logger: logger.warning(f"⚠️ Error saat memuat konfigurasi: {str(e)}")
    
    # Default config jika tidak ada file
    default_config = {
        "augmentation": {
            "input_dir": "data/preprocessed",
            "output_dir": "data/augmented",
            "types": ["Combined (Recommended)"],
            "prefix": "aug",
            "factor": 2,
            "target_split": "train",
            "balance_classes": True,
            "num_workers": 4,
            "enabled": True
        },
        "data": {
            "dir": "data"
        }
    }
    
    if logger: logger.info("ℹ️ Menggunakan konfigurasi default")
    
    # Simpan default config ke ui_components jika tersedia
    if ui_components: ui_components['config'] = default_config
    
    return default_config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update komponen UI dari konfigurasi dengan pendekatan yang lebih robust.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
    """
    logger = ui_components.get('logger')
    if logger: logger.debug(f"{ICONS['info']} Updating UI from config...")
    
    # Pastikan ada komponen aug_options
    if 'aug_options' not in ui_components or not ui_components['aug_options']:
        if logger: logger.warning(f"{ICONS['warning']} Tidak dapat menemukan komponen aug_options")
        return
    
    try:
        # Ambil bagian augmentation dari config
        aug_config = config.get('augmentation', {})
        
        # Log konfigurasi yang ditemukan
        if logger: logger.debug(f"{ICONS['info']} Konfigurasi augmentasi ditemukan: {aug_config}")
        
        # Dapatkan komponen UI
        aug_options = ui_components['aug_options'].children
        
        # Update UI components sesuai config
        if len(aug_options) >= 7:  # Pastikan jumlah komponen sesuai
            # Update aug types
            if 'types' in aug_config and hasattr(aug_options[0], 'value'):
                aug_options[0].value = aug_config['types']
                if logger: logger.debug(f"{ICONS['success']} Berhasil set aug_types: {aug_config['types']}")
            
            # Update prefix
            if 'prefix' in aug_config and hasattr(aug_options[2], 'value'):
                aug_options[2].value = aug_config['prefix']
                if logger: logger.debug(f"{ICONS['success']} Berhasil set prefix: {aug_config['prefix']}")
            
            # Update factor
            if 'factor' in aug_config and hasattr(aug_options[3], 'value'):
                aug_options[3].value = aug_config['factor']
                if logger: logger.debug(f"{ICONS['success']} Berhasil set factor: {aug_config['factor']}")
            
            # Update split
            if 'split' in aug_config and hasattr(aug_options[4], 'value'):
                aug_options[4].value = aug_config['split']
                if logger: logger.debug(f"{ICONS['success']} Berhasil set split: {aug_config['split']}")
            
            # Update balance_classes
            if 'balance_classes' in aug_config and hasattr(aug_options[5], 'value'):
                aug_options[5].value = aug_config['balance_classes']
                if logger: logger.debug(f"{ICONS['success']} Berhasil set balance_classes: {aug_config['balance_classes']}")
            
            # Update num_workers
            if 'num_workers' in aug_config and hasattr(aug_options[6], 'value'):
                aug_options[6].value = aug_config['num_workers']
                if logger: logger.debug(f"{ICONS['success']} Berhasil set num_workers: {aug_config['num_workers']}")
            
        # Simpan referensi config ke ui_components untuk memastikan persistensi
        ui_components['config'] = config
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error saat update UI dari config: {str(e)}")
    
    if logger: logger.debug(f"{ICONS['success']} UI berhasil diupdate dari konfigurasi")