"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Koordinator utama untuk handler preprocessing dataset dengan antarmuka yang ditingkatkan
"""

from typing import Dict, Any
import logging
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler preprocessing dataset dengan antarmuka yang ditingkatkan."""
    # Setup logger terintegrasi UI dengan utilitas standar
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, "preprocessing", log_level=logging.INFO)
        if logger: 
            ui_components['logger'] = logger
            logger.info(f"{ICONS['info']} Komponen preprocessing dataset siap digunakan")
    except ImportError:
        pass
    
    # Setup observer handlers untuk menangani event notification
    try:
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        ui_components = setup_observer_handlers(ui_components, "preprocessing_observers")
        if logger: logger.info(f"{ICONS['success']} Observer handlers berhasil diinisialisasi")
    except ImportError as e:
        if logger: logger.warning(f"{ICONS['warning']} Observer handlers tidak tersedia: {str(e)}")
    
    # Setup handlers komponen secara berurutan dengan pendekatan terpusat
    try:
        # Inisialisasi dengan utilitas standar
        from smartcash.ui.dataset.preprocessing_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Tambahkan dataset manager jika belum ada dengan validasi
        if 'dataset_manager' not in ui_components and config:
            from smartcash.dataset.manager import DatasetManager
            ui_components['dataset_manager'] = DatasetManager(config=config, logger=logger)
            if logger: logger.info(f"{ICONS['success']} Dataset Manager berhasil diinisialisasi")
        
        # Setup dengan standar error handling
        from smartcash.ui.handlers.error_handler import try_except_decorator
        
        # Setup handler dengan decorator error handling
        @try_except_decorator(ui_components.get('status'))
        def setup_handlers_safely():
            # Setup semua handler yang diperlukan dengan pendekatan terpusat
            from smartcash.ui.dataset.preprocessing_progress_handler import setup_progress_handler
            from smartcash.ui.dataset.preprocessing_click_handler import setup_click_handlers
            from smartcash.ui.dataset.preprocessing_cleanup_handler import setup_cleanup_handler
            from smartcash.ui.dataset.preprocessing_config_handler import setup_preprocessing_config_handler
            from smartcash.ui.dataset.preprocessing_visualization_handler import setup_visualization_handler
            from smartcash.ui.dataset.preprocessing_summary_handler import setup_summary_handler
            
            # Setup secara berurutan
            ui_components.update(setup_progress_handler(ui_components, env, config))
            ui_components.update(setup_preprocessing_config_handler(ui_components, config, env))
            ui_components.update(setup_click_handlers(ui_components, env, config))
            ui_components.update(setup_cleanup_handler(ui_components, env, config))
            ui_components.update(setup_visualization_handler(ui_components, env, config))
            ui_components.update(setup_summary_handler(ui_components, env, config))
            
            return ui_components
        
        # Jalankan setup dengan penanganan error
        setup_handlers_safely()
        
        # Reorganisasi layout tombol berdasarkan status yang sudah disimpan
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        import os
        is_preprocessed = False
        
        try:
            from pathlib import Path
            preprocessed_path = Path(preprocessed_dir)
            is_preprocessed = preprocessed_path.exists() and any(preprocessed_path.glob('**/images/*.jpg'))
        except Exception:
            pass
        
        # Update tampilan tombol berdasarkan status preprocessed
        if is_preprocessed:
            # Tampilkan tombol visualisasi
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'flex'
                
            # Tampilkan tombol cleanup
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'inline-block'
                
            # Tampilkan semua tombol visualisasi
            for btn_name in ['visualize_button', 'compare_button', 'summary_button']:
                if btn_name in ui_components:
                    ui_components[btn_name].layout.display = 'inline-block'
    
    except Exception as e:
        # Gunakan handler error standar
        from smartcash.ui.handlers.error_handler import handle_ui_error
        handle_ui_error(e, ui_components.get('status'), True, f"{ICONS['error']} Error saat setup handlers preprocessing")
    
    # Register cleanup handler
    def cleanup_resources():
        """Cleanup resources yang digunakan oleh handler."""
        # Cleanup observers jika ada
        if 'observer_group' in ui_components:
            try:
                from smartcash.components.observer.manager_observer import ObserverManager
                observer_manager = ObserverManager()
                observer_manager.unregister_group(ui_components['observer_group'])
                if logger: logger.info(f"{ICONS['cleanup']} Observer handlers dibersihkan")
            except ImportError:
                pass
    
    ui_components['cleanup'] = cleanup_resources
    
    return ui_components