"""
File: smartcash/ui/setup/cell_1_3_dependency_installer.py
Deskripsi: Cell instalasi dependencies untuk SmartCash dengan pendekatan modular
"""

from IPython.display import display

def run_cell():
    """Runner utama untuk cell instalasi dependencies"""
    try:
        from smartcash.ui.setup.dependency_installer_component import create_dependency_installer_ui
        from smartcash.ui.setup.dependency_installer_handler import setup_dependency_installer_handlers
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
    except ImportError as e:
        from IPython.display import HTML
        display(HTML(f"<div style='color:red'>❌ Error: {str(e)}</div>"))
        display(HTML("<div style='color:orange'>⚠️ Pastikan repository SmartCash sudah di-clone dengan benar</div>"))
        return

    try:
        # Coba dapatkan environment manager dan config
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        env = get_environment_manager()
        config_manager = get_config_manager()
        config = config_manager.config
    except ImportError:
        env, config = None, {}

    # Buat komponen UI
    ui_components = create_dependency_installer_ui(env, config)
    
    # Setup handlers
    ui_components = setup_dependency_installer_handlers(ui_components, config)
    
    # Setup ipython logging menggunakan utilitas yang sudah ada
    logger = setup_ipython_logging(ui_components, "dependency_installer")
    if logger:
        logger.info("🚀 Cell dependency_installer diinisialisasi")
    
    # Tampilkan UI - Pastikan hanya menampilkan widget UI, bukan dictionary UI components
    display(ui_components['ui'])
    
    return ui_components

# Jalankan cell dan simpan hasil tanpa menampilkannya
ui_components = run_cell()