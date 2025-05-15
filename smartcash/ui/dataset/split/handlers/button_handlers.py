"""
File: smartcash/ui/dataset/split/handlers/button_handlers.py
Deskripsi: Handler untuk tombol UI konfigurasi split dataset
"""

from typing import Dict, Any
from IPython.display import display, clear_output
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.ui_logger import create_direct_ui_logger
from smartcash.common.logger import get_logger
import yaml
from pathlib import Path

# Import fungsi yang selalu dibutuhkan
from smartcash.ui.dataset.split.handlers.config_handlers import (
    update_config_from_ui, save_config_with_manager, 
    load_default_config, get_config_manager_instance
)
from smartcash.ui.dataset.split.handlers.ui_initializer import (
    initialize_ui_from_config, ensure_ui_persistence, update_ui_from_config
)

# Definisi fungsi load_split_config_config langsung di file ini untuk menghindari masalah import
def load_split_config_config() -> Dict[str, Any]:
    """
    Load konfigurasi split dataset dari file YAML.
    
    Returns:
        Dictionary berisi konfigurasi split dataset
    """
    try:
        # Path ke file konfigurasi dataset
        config_path = Path("config/dataset_config.yaml")
        if not config_path.exists():
            # Coba cari di lokasi alternatif
            alt_path = Path("../config/dataset_config.yaml")
            if alt_path.exists():
                config_path = alt_path
            else:
                return load_default_config()
        
        # Load konfigurasi dari file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Pastikan struktur konfigurasi benar
        if not config:
            config = {}
        if 'data' not in config:
            config['data'] = {}
        if 'split' not in config['data']:
            config['data']['split'] = {'train': 0.7, 'val': 0.15, 'test': 0.15, 'stratified': True}
        
        return config
    except Exception as e:
        print(f"{ICONS['error']} Error saat memuat konfigurasi split dataset: {str(e)}")
        # Kembalikan konfigurasi default
        return load_default_config()

def setup_button_handlers(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol-tombol pada UI konfigurasi split dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Inisialisasi ui_components jika None
    if ui_components is None:
        ui_components = {}
    
    # Dapatkan ui_logger untuk logging yang lebih baik di UI
    logger = ui_components.get('logger', create_direct_ui_logger(ui_components, 'split_config'))
    
    # Pastikan konfigurasi data ada
    if not config:
        # Coba dapatkan konfigurasi dari ConfigManager
        config_manager = get_config_manager_instance()
        if config_manager:
            try:
                config = config_manager.get_module_config('dataset')
                if logger: logger.debug(f"{ICONS['info']} Konfigurasi berhasil dimuat dari ConfigManager")
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi dari ConfigManager: {str(e)}")
                # Fallback ke load dari file
                config = load_split_config_config()
        else:
            # Fallback ke load dari file
            config = load_split_config_config()
    
    # Pastikan struktur konfigurasi benar
    if not isinstance(config, dict):
        config = {}
    if 'data' not in config:
        config['data'] = {}
    
    # Pastikan UI components terdaftar untuk persistensi
    ensure_ui_persistence(ui_components, config, logger)
    
    # Inisialisasi UI dari konfigurasi
    initialize_ui_from_config(ui_components, config)
    
    # Register handler untuk save button
    if 'save_button' in ui_components and ui_components['save_button']:
        ui_components['save_button'].on_click(
            lambda b: handle_save_button(b, ui_components, config, env, logger)
        )
        if logger: logger.debug(f"{ICONS['link']} Handler untuk save button terdaftar")
    
    # Register handler untuk reset button
    if 'reset_button' in ui_components and ui_components['reset_button']:
        ui_components['reset_button'].on_click(
            lambda b: handle_reset_button(b, ui_components, config, env, logger)
        )
        if logger: logger.debug(f"{ICONS['link']} Handler untuk reset button terdaftar")
    
    return ui_components

def handle_save_button(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Handler untuk tombol save konfigurasi.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    # Import fungsi yang diperlukan untuk memastikan tersedia dalam scope
    from smartcash.ui.dataset.split.handlers.config_handlers import (
        update_config_from_ui, save_config_with_manager
    )
    
    # Pastikan output_box tersedia
    output_box = ui_components.get('output_box')
    if not output_box:
        print(f"{ICONS['warning']} Output box tidak tersedia")
        return
    
    # Gunakan status_panel jika tersedia
    status_panel = ui_components.get('status_panel')
    
    # Clear output dan tampilkan status
    with output_box:
        clear_output()
        display(create_status_indicator("Menyimpan konfigurasi...", "processing"))
    
    try:
        # Update konfigurasi dari UI
        config = update_config_from_ui(config, ui_components)
        
        # Simpan konfigurasi dengan ConfigManager atau fallback
        success = save_config_with_manager(config, ui_components, logger)
        
        # Tampilkan hasil
        with output_box:
            clear_output()
            if success:
                display(create_status_indicator("Konfigurasi berhasil disimpan", "success"))
                if logger: logger.debug(f"{ICONS['success']} Konfigurasi split dataset berhasil disimpan")
            else:
                display(create_status_indicator("Gagal menyimpan konfigurasi", "error"))
                if logger: logger.error(f"{ICONS['error']} Gagal menyimpan konfigurasi split dataset")
    except Exception as e:
        # Tampilkan error
        with output_box:
            clear_output()
            display(create_status_indicator(f"Error: {str(e)}", "error"))
        if logger: logger.error(f"{ICONS['error']} Error saat menyimpan konfigurasi split dataset: {str(e)}")

def handle_reset_button(b, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Handler untuk tombol reset konfigurasi.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    # Import fungsi yang diperlukan untuk memastikan tersedia dalam scope
    from smartcash.ui.dataset.split.handlers.config_handlers import (
        save_config_with_manager
    )
    from smartcash.ui.dataset.split.handlers.ui_initializer import (
        update_ui_from_config
    )
    
    # Pastikan output_box tersedia
    output_box = ui_components.get('output_box')
    if not output_box:
        print(f"{ICONS['warning']} Output box tidak tersedia")
        return
    
    # Clear output dan tampilkan status
    with output_box:
        clear_output()
        display(create_status_indicator("Mereset konfigurasi...", "processing"))
    
    try:
        # Load konfigurasi default
        default_config = load_default_config()
        
        # Update UI dari konfigurasi default - menggunakan update_ui_from_config alih-alih initialize_ui_from_config
        # untuk memastikan nilai slider diperbarui dengan benar
        update_ui_from_config(ui_components, default_config)
        
        # Simpan konfigurasi default dengan ConfigManager atau fallback
        success = save_config_with_manager(default_config, ui_components, logger)
        
        # Tampilkan hasil
        with output_box:
            clear_output()
            if success:
                display(create_status_indicator("Konfigurasi berhasil direset ke default", "success"))
                if logger: logger.debug(f"{ICONS['success']} Konfigurasi split dataset berhasil direset ke default")
            else:
                display(create_status_indicator("Gagal mereset konfigurasi", "error"))
                if logger: logger.error(f"{ICONS['error']} Gagal mereset konfigurasi split dataset")
    except Exception as e:
        # Tampilkan error
        with output_box:
            clear_output()
            display(create_status_indicator(f"Error: {str(e)}", "error"))
        if logger: logger.error(f"{ICONS['error']} Error saat mereset konfigurasi split dataset: {str(e)}")
