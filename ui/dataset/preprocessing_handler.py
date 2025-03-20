"""
File: smartcash/ui/dataset/preprocessing_handler.py
Deskripsi: Koordinator utama untuk handler preprocessing dataset dengan integrasi observer dan error handler
"""

from typing import Dict, Any
import logging
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS

def setup_preprocessing_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handler preprocessing dataset dengan integrasi observer dari handler standar."""
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
        
        # Tambahkan tombol visualisasi dan komparasi jika belum ada
        if 'visualize_button' not in ui_components and 'summary_container' in ui_components:
            visualize_button = widgets.Button(
                description='Visualisasi Sampel',
                button_style='info',
                icon='image',
                layout=widgets.Layout(width='auto', margin='5px', display='none')
            )
            ui_components['visualize_button'] = visualize_button
            
            compare_button = widgets.Button(
                description='Bandingkan Dataset',
                button_style='info',
                icon='columns',
                layout=widgets.Layout(width='auto', margin='5px', display='none')
            )
            ui_components['compare_button'] = compare_button
            
            summary_button = widgets.Button(
                description='Tampilkan Ringkasan',
                button_style='info',
                icon='list-alt',
                layout=widgets.Layout(width='auto', margin='5px', display='none')
            )
            ui_components['summary_button'] = summary_button
            
            # Tambahkan ke container
            buttons_container = widgets.HBox([
                visualize_button, compare_button, summary_button
            ], layout=widgets.Layout(margin='10px 0'))
            
            # Tambahkan ke summary container
            with ui_components['summary_container']:
                display(buttons_container)
            
            # Daftarkan handler untuk tombol-tombol baru
            if 'on_visualize_click' in ui_components and callable(ui_components['on_visualize_click']):
                visualize_button.on_click(ui_components['on_visualize_click'])
            
            if 'on_compare_click' in ui_components and callable(ui_components['on_compare_click']):
                compare_button.on_click(ui_components['on_compare_click'])
            
            if 'on_summary_click' in ui_components and callable(ui_components['on_summary_click']):
                summary_button.on_click(ui_components['on_summary_click'])
        
        # Cek direktori preprocessed, tampilkan tombol visualisasi jika tersedia
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        import os
        if os.path.exists(preprocessed_dir):
            # Tampilkan tombol visualisasi dan komparasi
            if 'visualize_button' in ui_components:
                ui_components['visualize_button'].layout.display = 'inline-flex'
            if 'compare_button' in ui_components:
                ui_components['compare_button'].layout.display = 'inline-flex'
            if 'summary_button' in ui_components:
                ui_components['summary_button'].layout.display = 'inline-flex'
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'block'
    
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