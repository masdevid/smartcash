"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Koordinator utama untuk handler preprocessing dataset SmartCash dengan ThreadPool
"""

from typing import Dict, Any
import logging
import atexit
from smartcash.ui.utils.constants import ICONS

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk preprocessing dataset dengan ThreadPool dan cleanup resources.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup logger terintegrasi UI
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "preprocessing", log_level=logging.INFO)
        if logger: 
            ui_components['logger'] = logger
            logger.info(f"{ICONS['info']} Komponen preprocessing dataset siap digunakan")
    except ImportError:
        pass
    
    # Setup handlers komponen
    try:
        # Inisialisasi
        from smartcash.ui.dataset.preprocessing_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Setup handler untuk tombol preprocessing dengan ThreadPool
        from smartcash.ui.dataset.preprocessing_click_handler import setup_click_handlers
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Setup handler untuk cleanup
        from smartcash.ui.dataset.preprocessing_cleanup_handler import setup_cleanup_handler
        ui_components = setup_cleanup_handler(ui_components, env, config)
        
        # Setup handler untuk progress tracking
        from smartcash.ui.dataset.preprocessing_progress_handler import setup_progress_handler
        ui_components = setup_progress_handler(ui_components, env, config)
        
        # Tambahkan dataset manager
        if 'dataset_manager' not in ui_components and config:
            from smartcash.dataset.manager import DatasetManager
            dataset_manager = DatasetManager(config=config, logger=logger)
            ui_components['dataset_manager'] = dataset_manager
            if logger: logger.info(f"{ICONS['success']} Dataset Manager berhasil diinisialisasi")
        
        # Register cleanup function untuk atexit
        def cleanup_resources():
            # Shutdown ThreadPoolExecutor jika ada
            executor = ui_components.get('thread_executor')
            if executor:
                try:
                    executor.shutdown(wait=False)
                    if logger: logger.info(f"{ICONS['cleanup']} ThreadPool dibersihkan")
                except Exception as e:
                    if logger: logger.warning(f"{ICONS['warning']} Error saat membersihkan ThreadPool: {str(e)}")
            
            # Reset flag preprocessing
            ui_components['preprocessing_running'] = False
        
        # Register cleanup function
        atexit.register(cleanup_resources)
        ui_components['cleanup_resources'] = cleanup_resources
    
    except ImportError as e:
        if logger: logger.warning(f"{ICONS['warning']} Modul tidak tersedia: {str(e)}")
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Error saat setup handlers: {str(e)}")
    
    return ui_components