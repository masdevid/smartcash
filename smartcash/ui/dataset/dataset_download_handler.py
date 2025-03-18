"""
File: smartcash/ui/dataset/dataset_download_handler.py
Deskripsi: Handler utama untuk download dataset SmartCash dengan validasi defensif
"""

from typing import Dict, Any
import logging
from IPython.display import display
from smartcash.ui.utils.constants import ICONS

def setup_dataset_download_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk download dataset dengan validasi defensif.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logging terintegrasi UI
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "dataset_download", log_level=logging.INFO)
        if logger: 
            ui_components['logger'] = logger
            logger.info(f"{ICONS['info']} Download dataset handler diinisialisasi")
    except ImportError:
        pass
    
    # Initialization handler
    try:
        # Setup komponen dasar
        from smartcash.ui.dataset.download_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Tambahkan dataset manager jika tidak ada
        if 'dataset_manager' not in ui_components:
            try:
                from smartcash.dataset.manager import DatasetManager
                dataset_manager = DatasetManager(config=config, logger=logger)
                ui_components['dataset_manager'] = dataset_manager
                if logger: logger.info(f"{ICONS['success']} Dataset Manager berhasil diinisialisasi")
            except ImportError as e:
                if logger: logger.warning(f"{ICONS['warning']} Tidak dapat membuat DatasetManager: {str(e)}")
        
        # Setup handlers UI dan download
        from smartcash.ui.dataset.download_ui_handler import setup_ui_handlers
        from smartcash.ui.dataset.download_click_handler import setup_click_handlers
        
        ui_components = setup_ui_handlers(ui_components, env, config)  
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Setup konfirmasi download
        try:
            from smartcash.ui.dataset.download_confirmation_handler import setup_confirmation_handlers
            ui_components = setup_confirmation_handlers(ui_components, env, config)
        except ImportError:
            pass
        
        # Cek dataset yang sudah ada
        data_dir = get_data_directory(ui_components, env, config)
        
        if data_dir:
            try:
                from smartcash.ui.dataset.download_confirmation_handler import check_existing_dataset, get_dataset_stats
                
                if check_existing_dataset(data_dir):
                    stats = get_dataset_stats(data_dir)
                    if logger: logger.info(f"{ICONS['folder']} Dataset terdeteksi: {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']})")
                    
                    # Validasi struktur jika fungsi tersedia
                    if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
                        ui_components['validate_dataset_structure'](data_dir)
            except Exception as e:
                if logger: logger.warning(f"{ICONS['warning']} Gagal memeriksa dataset: {str(e)}")
                
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Error saat setup handlers: {str(e)}")
    
    return ui_components

def get_data_directory(ui_components: Dict[str, Any], env=None, config=None) -> str:
    """
    Dapatkan direktori data dengan validasi defensif.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Path direktori data
    """
    # Default data directory
    data_dir = "data" 
    
    # Coba ambil dari config jika tersedia
    if config and isinstance(config, dict) and 'data' in config:
        data_dir = config.get('data', {}).get('dir', 'data')
    
    # Gunakan Google Drive jika tersedia
    if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
        data_dir = str(env.drive_path / 'data')
    
    # Simpan ke ui_components untuk penggunaan lain
    ui_components['data_dir'] = data_dir
    
    return data_dir