"""
File: smartcash/ui/dataset/dataset_download_handler.py
Deskripsi: Handler utama untuk download dataset SmartCash yang disederhanakan
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS

def setup_dataset_download_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler untuk download dataset dengan integrasi logging."""
    # Setup logging terintegrasi UI
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "dataset_download", redirect_root=True)
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
                if logger: logger.warning(f"{ICONS['warning']} DatasetManager tidak tersedia: {str(e)}")
        
        # Setup handlers UI dan download dengan penanganan error yang lebih baik
        try:
            from smartcash.ui.dataset.download_ui_handler import setup_ui_handlers
            ui_components = setup_ui_handlers(ui_components, env, config)
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error setup UI handlers: {str(e)}")
            
        try:
            from smartcash.ui.dataset.download_click_handler import setup_click_handlers
            ui_components = setup_click_handlers(ui_components, env, config)
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error setup click handlers: {str(e)}")
        
        # Cek dataset yang sudah ada
        try:
            from smartcash.ui.dataset.download_confirmation_handler import check_existing_dataset, get_dataset_stats
            
            # Setup direktori data
            data_dir = config.get('data', {}).get('dir', 'data') if config else 'data'
            if env and hasattr(env, 'is_drive_mounted') and env.is_drive_mounted and hasattr(env, 'drive_path'):
                data_dir = str(env.drive_path / 'data')
                
            # Periksa dataset yang sudah ada    
            if check_existing_dataset(data_dir):
                stats = get_dataset_stats(data_dir)
                if logger: logger.info(f"{ICONS['folder']} Dataset terdeteksi: {stats['total_images']} gambar (Train: {stats['train']}, Valid: {stats['valid']}, Test: {stats['test']})")
                
                # Validasi struktur jika fungsi tersedia
                if 'validate_dataset_structure' in ui_components and callable(ui_components['validate_dataset_structure']):
                    ui_components['validate_dataset_structure'](data_dir)
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Error cek dataset: {str(e)}")
                
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Error saat setup handlers: {str(e)}")
    
    return ui_components