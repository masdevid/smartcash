"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Koordinator handler preprocessing dengan integrasi inisialisasi path dan pengaturan UI yang ditingkatkan
"""

from typing import Dict, Any
import logging
from IPython.display import display, clear_output

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler preprocessing dataset dengan integrasi komponen path dan pengaturan UI yang ditingkatkan."""
    
    # Setup logger terintegrasi UI
    logger = None
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        from smartcash.ui.utils.constants import ICONS
        
        logger = setup_ipython_logging(ui_components, "preprocessing", log_level=logging.INFO)
        if logger: 
            ui_components['logger'] = logger
            logger.info(f"{ICONS['info']} Komponen preprocessing dataset siap digunakan")
    except ImportError:
        pass
    
    # Inisialisasi dengan urutan yang benar: path setup dulu, baru handler lainnya
    try:
        # Step 1: Inisialisasi path dan environment - PENTING: ini dilakukan pertama
        from smartcash.ui.dataset.preprocessing_initialization import setup_initialization
        ui_components = setup_initialization(ui_components, env, config)
        
        # Step 2: Setup observer handlers untuk notifikasi kemajuan
        try:
            from smartcash.ui.handlers.observer_handler import setup_observer_handlers
            ui_components = setup_observer_handlers(ui_components, "preprocessing_observers")
            if logger: logger.info(f"{ICONS['success']} Observer handlers berhasil diinisialisasi")
        except ImportError as e:
            if logger: logger.debug(f"{ICONS['warning']} Observer handlers tidak tersedia: {str(e)}")
        
        # Step 3: Tambahkan dataset manager jika belum ada
        if 'dataset_manager' not in ui_components and config:
            try:
                from smartcash.dataset.manager import DatasetManager
                ui_components['dataset_manager'] = DatasetManager(config=config, logger=logger)
                if logger: logger.info(f"{ICONS['success']} Dataset Manager berhasil diinisialisasi")
            except ImportError as e:
                if logger: logger.warning(f"{ICONS['warning']} Dataset Manager tidak tersedia: {str(e)}")
        
        # Step 4: Setup semua handler yang diperlukan dengan pendekatan terkonsolidasi
        from smartcash.ui.handlers.error_handler import try_except_decorator
        
        @try_except_decorator(ui_components.get('status'))
        def setup_all_handlers():
            # Import semua handler yang diperlukan
            handlers_to_setup = [
                "preprocessing_progress_handler.setup_progress_handler",
                "preprocessing_config_handler.setup_preprocessing_config_handler",
                "preprocessing_click_handler.setup_click_handlers",
                "preprocessing_cleanup_handler.setup_cleanup_handler",
                "preprocessing_visualization_handler.setup_visualization_handler",
                "preprocessing_summary_handler.setup_summary_handler"
            ]
            
            # Setup semua handler secara berurutan
            for handler_path in handlers_to_setup:
                try:
                    module_path, func_name = handler_path.rsplit('.', 1)
                    module = __import__(f"smartcash.ui.dataset.{module_path}", fromlist=[func_name])
                    setup_func = getattr(module, func_name)
                    ui_components.update(setup_func(ui_components, env, config))
                    if logger: logger.debug(f"✅ Handler {func_name} berhasil dimuat")
                except (ImportError, AttributeError) as e:
                    if logger: logger.debug(f"⚠️ Handler {handler_path} tidak tersedia: {str(e)}")
            
            return ui_components
        
        # Jalankan setup semua handler
        setup_all_handlers()
        
        # Step 5: Aktifkan tampilan tombol visualisasi jika data preprocessed sudah ada
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        if os.path.exists(preprocessed_dir):
            # Cek apakah ada file preprocessed
            try:
                if any(Path(preprocessed_dir).glob('**/images/*.jpg')):
                    # Tampilkan tombol visualisasi dan cleanup
                    if 'visualization_buttons' in ui_components:
                        ui_components['visualization_buttons'].layout.display = 'flex'
                    if 'cleanup_button' in ui_components:
                        ui_components['cleanup_button'].layout.display = 'block'
                    if logger: logger.info(f"📊 Dataset preprocessed terdeteksi, tombol visualisasi diaktifkan")
            except Exception:
                pass
    
    except Exception as e:
        # Gunakan handler error standar
        from smartcash.ui.handlers.error_handler import handle_ui_error
        from smartcash.ui.utils.constants import ICONS
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

# Diperlukan import ini karena digunakan dalam fungsi
import os
from pathlib import Path