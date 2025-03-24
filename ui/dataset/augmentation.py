"""
File: smartcash/ui/dataset/augmentation.py
Deskripsi: Koordinator utama untuk instalasi dependency SmartCash
"""

def setup_augmentation():
    """Koordinator utama setup dan konfigurasi dataset augmentation dengan integrasi utilities"""
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.dataset.augmentation_component import create_augmentation_ui
        from smartcash.ui.dataset.augmentation_handler import setup_augmentation_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging

        cell_name = 'augmentation'
        # Setup environment dan load config
        env, config = setup_notebook_environment(cell_name)

        # Setup komponen UI dengan utils terstandarisasi
        ui_components = create_augmentation_ui(env, config)

        # Setup dataset handler dengan utils terstandarisasi
        ui_components = setup_augmentation_handlers(ui_components, env, config)

        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, cell_name)
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Cell augmentation diinisialisasi")
            
        from IPython import get_ipython
        if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            cleanup = ui_components['cleanup']
            get_ipython().events.register('pre_run_cell', cleanup)

    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    
    return ui_components