"""
File: smartcash/ui/dataset/dataset_download_handler.py
Deskripsi: Handler utama untuk download dataset SmartCash yang disederhanakan
"""

from typing import Dict, Any
import logging
from smartcash.ui.utils.constants import ICONS

def setup_dataset_download_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk download dataset dengan integrasi logging.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logging terintegrasi UI
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "dataset_download", log_level=logging.INFO)
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Download dataset handler diinisialisasi")
    except ImportError:
        logger = None
    
    # Import dan setup handlers dengan error handling sederhana
    try:
        # Initialization handler
        from smartcash.ui.dataset.download_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Tambahkan dataset manager jika tidak ada
        if 'dataset_manager' not in ui_components:
            try:
                from smartcash.dataset.manager import DatasetManager
                dataset_manager = DatasetManager(config=config, logger=logger)
                ui_components['dataset_manager'] = dataset_manager
                if logger:
                    logger.info(f"{ICONS['success']} Dataset Manager berhasil diinisialisasi")
            except ImportError as e:
                if logger:
                    logger.warning(f"{ICONS['warning']} DatasetManager tidak tersedia: {str(e)}")
        
        # Setup handlers untuk UI dan download
        from smartcash.ui.dataset.download_ui_handler import setup_ui_handlers
        ui_components = setup_ui_handlers(ui_components, env, config)
        
        from smartcash.ui.dataset.download_click_handler import setup_click_handlers  
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Cek dataset yang sudah ada
        try:
            from smartcash.ui.dataset.download_confirmation_handler import check_existing_dataset, get_dataset_stats
            data_dir = config.get('data', {}).get('dir', 'data')
            if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted:
                data_dir = str(env.drive_path / 'data')
                
            if check_existing_dataset(data_dir):
                stats = get_dataset_stats(data_dir)
                if logger:
                    logger.info(f"{ICONS['folder']} Dataset terdeteksi: {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']})")
                
                if 'validate_dataset_structure' in ui_components:
                    ui_components['validate_dataset_structure'](data_dir)
        except Exception:
            # Abaikan jika pengecekan dataset gagal
            pass
            
    except Exception as e:
        if logger:
            logger.error(f"{ICONS['error']} Error saat setup handlers: {str(e)}")
    
    return ui_components