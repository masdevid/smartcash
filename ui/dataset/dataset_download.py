"""
File: smartcash/ui/dataset/dataset_download.py
Deskripsi: Koordinator utama untuk instalasi dependency SmartCash
"""

def setup_dataset_download():
    """Koordinator utama setup dan konfigurasi dataset download dengan integrasi utilities"""
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.dataset.dataset_download_component import create_dataset_download_ui
        from smartcash.ui.dataset.dataset_download_handler import setup_dataset_download_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging

        cell_name = 'dataset_download'
        # Setup environment dan load config
        env, config = setup_notebook_environment(cell_name)

        # Setup komponen UI dengan utils terstandarisasi
        ui_components = create_dataset_download_ui(env, config)

        # Setup dataset handler dengan utils terstandarisasi
        ui_components = setup_dataset_download_handlers(ui_components, env, config)

        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, cell_name)
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Cell dataset_download diinisialisasi")
            
        from IPython import get_ipython
        if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            cleanup = ui_components['cleanup']
            get_ipython().events.register('pre_run_cell', cleanup)

    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    
    return ui_components
