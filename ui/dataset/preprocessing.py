"""
File: smartcash/ui/dataset/preprocessing.py
Deskripsi: Koordinator utama preprocessing dataset dengan antarmuka visual yang ditingkatkan
"""

def setup_preprocessing():
    """Koordinator utama data preprocessing dengan integrasi utilities yang disederhanakan."""
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.dataset.preprocessing_component import create_preprocessing_ui
        from smartcash.ui.dataset.preprocessing_handler import setup_preprocessing_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging

        # Setup environment dan load config
        env, config = setup_notebook_environment('preprocessing')

        # Setup komponen UI dengan utils terstandarisasi
        ui_components = create_preprocessing_ui(env, config)

        # Setup dataset handler
        ui_components = setup_preprocessing_handlers(ui_components, env, config)

        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, 'preprocessing')
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Cell preprocessing diinisialisasi")
            
        # Register cleanup handler untuk event pre_run_cell
        from IPython import get_ipython
        if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            cleanup = ui_components['cleanup']
            get_ipython().events.register('pre_run_cell', cleanup)
            
        # Display UI
        from smartcash.ui.utils.cell_utils import display_ui
        display_ui(ui_components)
        
        return ui_components

    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import show_status
        ui_components = {'module_name': 'preprocessing'}
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
        return ui_components