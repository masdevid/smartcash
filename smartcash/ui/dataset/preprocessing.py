"""
File: smartcash/ui/dataset/preprocessing.py
Deskripsi: Koordinator utama preprocessing dataset dengan antarmuka visual yang ditingkatkan
"""

def setup_preprocessing():
    """Koordinator utama data preprocessing dengan integrasi utilities yang disederhanakan."""
    # Cek jika instance sudah dibuat sebelumnya (mencegah duplikasi)
    # import builtins
    # if hasattr(builtins, '_preprocessing_ui_instance'):
    #     # Hanya kembalikan instance yang telah ada
    #     return builtins._preprocessing_ui_instance
        
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.dataset.preprocessing_component import create_preprocessing_ui
        from smartcash.ui.dataset.preprocessing_handler import setup_preprocessing_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        import ipywidgets as widgets

        # Setup environment dan load config
        env, config = setup_notebook_environment('preprocessing')

        # Setup komponen UI dengan utils terstandarisasi
        ui_components = create_preprocessing_ui(env, config)

        # Setup logging untuk UI sebelum setup handler
        logger = setup_ipython_logging(ui_components, 'preprocessing')
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Cell preprocessing diinisialisasi")

        # Setup dataset handler (setelah logger diinisialisasi)
        ui_components = setup_preprocessing_handlers(ui_components, env, config)
            
        # Register cleanup handler untuk event pre_run_cell
        from IPython import get_ipython
        if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            cleanup = ui_components['cleanup']
            # Cek apakah event handler sudah terdaftar untuk mencegah duplikasi
            try:
                if not hasattr(get_ipython().events, '_preprocessing_cleanup_registered'):
                    get_ipython().events.register('pre_run_cell', cleanup)
                    setattr(get_ipython().events, '_preprocessing_cleanup_registered', True)
            except:
                pass
            
        # Display UI - hanya tampilkan 1 kali
        # from smartcash.ui.utils.cell_utils import display_ui
        # display_ui(ui_components)

        # Simpan instance sebagai global untuk mencegah duplikasi
        # builtins._preprocessing_ui_instance = ui_components
        
        # Kembalikan ui_components untuk digunakan sebagai state global
        return ui_components

    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        error_msg = f"⚠️ Beberapa komponen tidak tersedia: {str(e)}"
        ui_components = create_fallback_ui({'module_name': 'preprocessing'}, error_msg, "warning")
        return ui_components