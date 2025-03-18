"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Koordinator utama untuk handler preprocessing dataset SmartCash
"""

from typing import Dict, Any
import logging

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk preprocessing dataset dengan integrasi logging.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Setup logger yang terintegrasi dengan UI
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        
        # Buat atau dapatkan logger
        logger = setup_ipython_logging(
            ui_components, 
            logger_name="preprocessing", 
            log_level=logging.INFO
        )
        
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Komponen preprocessing dataset siap digunakan")
    except ImportError as e:
        print(f"⚠️ Tidak dapat mengintegrasikan logger: {str(e)}")
    
    try:
        # Setup handlers untuk komponen-komponen UI
        from smartcash.ui.dataset.preprocessing_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Setup handler untuk tombol preprocessing
        from smartcash.ui.dataset.preprocessing_click_handler import setup_click_handlers
        ui_components = setup_click_handlers(ui_components, env, config)
        
        # Setup handler untuk cleanup
        from smartcash.ui.dataset.preprocessing_cleanup_handler import setup_cleanup_handler
        ui_components = setup_cleanup_handler(ui_components, env, config)
        
        # Setup handler untuk progress tracking
        from smartcash.ui.dataset.preprocessing_progress_handler import setup_progress_handler
        ui_components = setup_progress_handler(ui_components, env, config)
        
        # Tambahkan dataset manager jika tersedia
        try:
            from smartcash.dataset.manager import DatasetManager
            
            if 'dataset_manager' not in ui_components and config:
                dataset_manager = DatasetManager(config=config, logger=ui_components.get('logger'))
                ui_components['dataset_manager'] = dataset_manager
                
                if 'logger' in ui_components:
                    ui_components['logger'].info("✅ Dataset Manager berhasil diinisialisasi")
        except ImportError as e:
            if 'logger' in ui_components:
                ui_components['logger'].warning(f"⚠️ Tidak dapat menggunakan DatasetManager: {str(e)}")
                ui_components['logger'].info("ℹ️ Beberapa fitur mungkin tidak tersedia")
    
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Error saat setup handlers: {str(e)}")
        else:
            print(f"❌ Error saat setup handlers: {str(e)}")
    
    return ui_components