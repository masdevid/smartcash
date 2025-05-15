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
from smartcash.ui.utils.constants import ICONS, COLORS

# Import persistensi di level modul untuk memastikan tersedia di semua fungsi
from smartcash.ui.dataset.augmentation.handlers.config_persistence import (
    ensure_ui_persistence, get_augmentation_config, save_augmentation_config
)

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
    # ConfigManager untuk persistensi sudah diimport di level modul
    # Pastikan ui_components tidak None
    if ui_components is None:
        ui_components = {}
        
    # Dapatkan logger jika tersedia
    logger = ui_components.get('logger')
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("augmentation_config")
            ui_components['logger'] = logger
        except ImportError:
            pass
    
    # Load konfigurasi jika belum ada
    if config is None:
        config = load_augmentation_config(ui_components=ui_components)
    
    # Pastikan config tidak None
    if config is None:
        config = {}
        if logger: logger.warning("⚠️ config adalah None, membuat dictionary kosong")
    
    # Update UI dari konfigurasi jika ada komponen UI
    if ui_components:
        ui_components = update_ui_from_config(ui_components, config)
    
    # Handler untuk tombol save config
    def on_save_config(b):
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
        try:
            # Pastikan ui_components tidak None dan memiliki config
            if 'config' not in ui_components or ui_components['config'] is None:
                ui_components['config'] = {}
                if 'augmentation' not in ui_components['config']:
                    ui_components['config']['augmentation'] = {}
                if 'data' not in ui_components['config']:
                    ui_components['config']['data'] = {}
            
            # Update config dari UI dan simpan
            updated_config = update_config_from_ui(ui_components, ui_components.get('config', config))
            
            # Pastikan updated_config memiliki struktur yang benar
            if 'augmentation' not in updated_config:
                updated_config['augmentation'] = {}
            if 'data' not in updated_config:
                updated_config['data'] = {}
                
            # Simpan konfigurasi
            success = save_augmentation_config(updated_config)
            
            # Simpan kembali config yang diupdate ke ui_components
            ui_components['config'] = updated_config
            
            # Tampilkan status
            status_type = 'success' if success else 'error'
            message = "✅ Konfigurasi augmentasi berhasil disimpan" if success else "❌ Gagal menyimpan konfigurasi augmentasi"
        except Exception as e:
            status_type = 'error'
            message = f"❌ Error saat menyimpan konfigurasi: {str(e)}"
            if logger: logger.error(message)
        
        # Update status jika ada
        if 'status' in ui_components:
            with ui_components['status']: 
                display(create_status_indicator(status_type, message))
                
            # Update status panel
            try:
                from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel
                update_status_panel(ui_components, status_type, message)
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error saat update status panel: {str(e)}")
        
        # Log
        if logger: 
            log_method = logger.success if success else logger.error
            log_method(message)
    
    # Register handler untuk tombol save jika ada
    if ui_components and 'save_button' in ui_components:
        ui_components['save_button'].on_click(on_save_config)
        
    # Register handler untuk tombol augment (jika ada)
    if ui_components and 'augment_button' in ui_components:
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
    # Validasi parameter untuk mencegah error NoneType is not iterable
    if ui_components is None:
        import logging
        logger = logging.getLogger('augmentation')
        logger.warning("⚠️ ui_components adalah None saat update_config_from_ui")
        return {}
    
    # Inisialisasi config dengan deep copy untuk mencegah modifikasi tidak sengaja
    config = copy.deepcopy(config or {})
    logger = ui_components.get('logger')
    
    # Pastikan section augmentation dan data ada
    if 'augmentation' not in config: config['augmentation'] = {}
    if 'data' not in config: config['data'] = {}
    
    # Validasi config manager jika tersedia
    try:
        from smartcash.common.config.manager import ConfigManager
        config_manager = ConfigManager.get_instance()
        if config_manager and hasattr(config_manager, 'validate_param'):
            # Pastikan section augmentation ada dan valid
            config['augmentation'] = config_manager.validate_param(
                config.get('augmentation', {}), 
                default={}, 
                param_name='augmentation'
            )
    except Exception as e:
        if logger: logger.debug(f"🔶 Tidak dapat menggunakan ConfigManager: {str(e)}")
    
    # Ekstrak paths dari ui_components
    data_dir = ui_components.get('data_dir', 'data')
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
    
    # Update paths dalam config
    config['data']['dir'] = data_dir
    config['augmentation']['input_dir'] = preprocessed_dir
    config['augmentation']['output_dir'] = augmented_dir
    
    # Ekstrak nilai dari aug_options dengan validasi yang lebih kuat
    options = ui_components.get('aug_options', {})
    if hasattr(options, 'children') and len(options.children) >= 6:  # Sekarang hanya 6 children karena menghilangkan 2 komponen dan menambahkan 1 komponen info
        # Inisialisasi nilai default yang aman
        aug_types = ['Combined (Recommended)']  # Nilai tetap
        aug_prefix = 'aug'
        aug_factor = 2
        target_split = 'train'  # Nilai tetap
        balance_classes = True
        num_workers = 4
        
        # Ekstrak nilai dengan validasi untuk mencegah None
        try:
            # Ekstrak prefix dengan validasi (sekarang di posisi 1)
            extracted_prefix = options.children[1].value  # Text
            if extracted_prefix is not None and isinstance(extracted_prefix, str) and extracted_prefix.strip():
                aug_prefix = extracted_prefix
            if logger: logger.debug(f"🔍 Ekstrak prefix: {aug_prefix}")
            
            # Ekstrak factor dengan validasi (sekarang di posisi 2)
            extracted_factor = options.children[2].value  # IntSlider
            if extracted_factor is not None and isinstance(extracted_factor, (int, float)) and extracted_factor > 0:
                aug_factor = extracted_factor
            if logger: logger.debug(f"🔍 Ekstrak factor: {aug_factor}")
            
            # Ekstrak balance_classes dengan validasi (sekarang di posisi 4)
            extracted_balance = options.children[4].value  # Checkbox
            if extracted_balance is not None:
                balance_classes = bool(extracted_balance)
            if logger: logger.debug(f"🔍 Ekstrak balance_classes: {balance_classes}")
            
            # Ekstrak num_workers dengan validasi (sekarang di posisi 5)
            extracted_workers = options.children[5].value  # IntSlider
            if extracted_workers is not None and isinstance(extracted_workers, int) and extracted_workers > 0:
                num_workers = extracted_workers
            if logger: logger.debug(f"🔍 Ekstrak num_workers: {num_workers}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat ekstrak nilai dari UI: {str(e)}")
        
        # Update konfigurasi augmentation dengan nilai yang sudah divalidasi
        config['augmentation'].update({
            'types': aug_types,  # Nilai tetap
            'prefix': aug_prefix,
            'factor': aug_factor,
            'split': target_split,  # Nilai tetap
            'balance_classes': balance_classes,
            'num_workers': num_workers,
            'enabled': True
        })
    
    # Simpan referensi config di ui_components untuk memastikan persistensi
    ui_components['config'] = config
    
    # Pastikan persistensi UI components dengan ConfigManager
    ensure_ui_persistence(ui_components)
    
    # Validasi final untuk memastikan aug_types tidak None
    if config['augmentation'].get('types') is None:
        config['augmentation']['types'] = ['Combined (Recommended)']
        if logger: logger.warning(f"{ICONS['warning']} Memperbaiki aug_types yang None dengan nilai default")
    
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
    # Import ConfigManager untuk persistensi
    from smartcash.common.config.manager import get_config_manager
    logger = None
    try:
        # Ambil logger dari lingkungan jika tersedia
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("augmentation_config")
        except ImportError:
            logger = None
            
        # Validasi config
        if config is None:
            if logger: logger.warning("⚠️ config adalah None, membuat dictionary kosong")
            config = {}
            
        # Pastikan struktur config benar
        if 'augmentation' not in config:
            config['augmentation'] = {}
    
        # Simpan ke ConfigManager (metode utama)
        config_manager = get_config_manager()
        success = config_manager.save_module_config('augmentation', config)
        if success:
            if logger: logger.info("✅ Konfigurasi augmentasi berhasil disimpan ke ConfigManager")
            return True
            
        # Fallback ke metode lama jika ConfigManager gagal
        try:
            # Pastikan path ada
            config_dir = os.path.dirname(config_path)
            os.makedirs(config_dir, exist_ok=True)
            
            # Simpan konfigurasi
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            if logger: logger.info(f"✅ Konfigurasi augmentasi berhasil disimpan ke {config_path}")
            return True
        except Exception as e:
            if logger: logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
            return False
        
        # Coba sync dengan drive jika tersedia
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Cek apakah path sama dengan realpath untuk mencegah error pada symlink
                if os.path.realpath(config_path) == os.path.realpath(drive_config_path):
                    # Meskipun path identik, kita tetap perlu memastikan konten diperbarui
                    # karena mungkin ada perubahan yang belum tersimpan
                    with open(drive_config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    if logger: logger.info(f"🔄 File lokal dan drive identik, memperbarui konten: {config_path}")
                else:
                    # Buat direktori jika belum ada
                    os.makedirs(Path(drive_config_path).parent, exist_ok=True)
                    
                    # Salin file ke drive
                    shutil.copy2(config_path, drive_config_path)
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
    # ConfigManager untuk persistensi sudah diimport di level modul
    logger = None
    
    # Validasi ui_components untuk mencegah error NoneType is not iterable
    if ui_components is None:
        ui_components = {}
    
    # Ambil logger dari ui_components jika tersedia
    if 'logger' in ui_components:
        logger = ui_components['logger']
    
    # Default config
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
    
    # Flag untuk status loading
    loaded = False
    config = {}
    
    try:
        # Coba load dari ConfigManager (metode utama)
        try:
            from smartcash.common.logger import get_logger
            if logger is None:
                logger = get_logger("augmentation_config")
        except ImportError:
            pass
                
        # Load dari ConfigManager
        config = get_augmentation_config(default_config)
        
        if config and 'augmentation' in config:
            if logger: logger.info("✅ Konfigurasi augmentasi dimuat dari ConfigManager")
            loaded = True
        else:
            # Fallback ke metode lama jika ConfigManager gagal
            try:
                # Coba load dari path yang diberikan
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                    if logger: logger.info(f"✅ Konfigurasi augmentasi dimuat dari {config_path}")
                    loaded = True
                else:
                    # Coba load dari path alternatif
                    alt_config_path = os.path.join('smartcash', 'configs', os.path.basename(config_path))
                    if os.path.exists(alt_config_path):
                        with open(alt_config_path, 'r') as f:
                            config = yaml.safe_load(f) or {}
                        if logger: logger.info(f"✅ Konfigurasi augmentasi dimuat dari {alt_config_path}")
                        loaded = True
            except Exception as e:
                if logger: logger.warning(f"⚠️ Error saat memuat konfigurasi dari file: {str(e)}")
                loaded = False
    except Exception as e:
        if logger: logger.error(f"❌ Error saat memuat konfigurasi: {str(e)}")
        loaded = False
    
    # Jika tidak berhasil dimuat, gunakan default
    if not loaded or not config:
        if logger: logger.info("ℹ️ Menggunakan konfigurasi default")
        config = default_config
    
    # Pastikan struktur config lengkap
    if 'augmentation' not in config:
        config['augmentation'] = default_config['augmentation']
    if 'data' not in config:
        config['data'] = default_config['data']
    
    return default_config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update komponen UI dari konfigurasi dengan pendekatan yang lebih robust.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # ConfigManager untuk persistensi sudah diimport di level modul
    # Validasi parameter untuk mencegah error NoneType is not iterable
    if ui_components is None:
        import logging
        logger = logging.getLogger('augmentation')
        logger.warning("⚠️ ui_components adalah None saat update_ui_from_config")
        return {}
    
    if config is None:
        config = {}
        if 'logger' in ui_components:
            ui_components['logger'].warning("⚠️ config adalah None saat update_ui_from_config")
    
    logger = ui_components.get('logger')
    if logger: logger.debug(f"{ICONS['info']} Updating UI from config...")
    
    # Pastikan ada komponen aug_options
    if 'aug_options' not in ui_components or not ui_components['aug_options']:
        if logger: logger.warning(f"{ICONS['warning']} Tidak dapat menemukan komponen aug_options")
        return ui_components
    
    try:
        # Ambil bagian augmentation dari config
        aug_config = config.get('augmentation', {})
        
        # Log konfigurasi yang ditemukan
        if logger: logger.debug(f"{ICONS['info']} Konfigurasi augmentasi ditemukan: {aug_config}")
        
        # Dapatkan komponen UI dengan validasi
        try:
            aug_options = ui_components['aug_options'].children
            
            # Update UI components sesuai config
            if len(aug_options) >= 6:  # Pastikan jumlah komponen sesuai (sekarang 6 children)
                # Jenis augmentasi dan target split sekarang tetap, tidak perlu diupdate dari config
                
                # Update prefix (sekarang di posisi 1)
                if 'prefix' in aug_config and hasattr(aug_options[1], 'value'):
                    aug_options[1].value = aug_config['prefix']
                    if logger: logger.debug(f"{ICONS['success']} Berhasil set prefix: {aug_config['prefix']}")
                
                # Update factor (sekarang di posisi 2)
                if 'factor' in aug_config and hasattr(aug_options[2], 'value'):
                    aug_options[2].value = aug_config['factor']
                    if logger: logger.debug(f"{ICONS['success']} Berhasil set factor: {aug_config['factor']}")
                
                # Update balance_classes (sekarang di posisi 4)
                if 'balance_classes' in aug_config and hasattr(aug_options[4], 'value'):
                    aug_options[4].value = aug_config['balance_classes']
                    if logger: logger.debug(f"{ICONS['success']} Berhasil set balance_classes: {aug_config['balance_classes']}")
                
                # Update num_workers (sekarang di posisi 5)
                if 'num_workers' in aug_config and hasattr(aug_options[5], 'value'):
                    aug_options[5].value = aug_config['num_workers']
                    if logger: logger.debug(f"{ICONS['success']} Berhasil set num_workers: {aug_config['num_workers']}")
                    
                # Pastikan nilai tetap selalu tersimpan di config
                aug_config['types'] = ['Combined (Recommended)']
                aug_config['split'] = 'train'
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error saat akses aug_options: {str(e)}")
        
        # Simpan referensi config ke ui_components untuk memastikan persistensi
        ui_components['config'] = config
        
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Error saat update UI dari config: {str(e)}")
    
    if logger: logger.debug(f"{ICONS['success']} UI berhasil diupdate dari konfigurasi")
    
    return ui_components