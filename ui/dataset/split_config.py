"""
File: smartcash/ui/dataset/split_config.py
Deskripsi: Koordinator utama untuk konfigurasi split dataset
"""
def setup_split_config():
    """Koordinator utama split config dengan integrasi utilities"""
    ui_components = {}
    
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.dataset.split_config_component import create_split_config_ui
        from smartcash.ui.dataset.split_config_handler import setup_split_config_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging

        cell_name = 'split_config'
        # Setup environment dan load config
        env, config = setup_notebook_environment(cell_name)

        # Setup komponen UI dengan utils terstandarisasi
        ui_components = create_split_config_ui(env, config)

        # Setup dataset handler dengan utils terstandarisasi
        ui_components = setup_split_config_handlers(ui_components, env, config)

        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, cell_name)
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Cell split_config diinisialisasi")
            
        # Register cleanup handler untuk saat cell dijalankan
        from IPython import get_ipython
        if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            cleanup = ui_components['cleanup']
            get_ipython().events.register('pre_run_cell', cleanup)

    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", 'warning', ui_components)
    
    return ui_components