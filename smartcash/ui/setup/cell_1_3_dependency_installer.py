"""
File: smartcash/ui/setup/cell_1_3_dependency_installer.py
Deskripsi: Cell instalasi dependencies untuk SmartCash dengan pendekatan modular
"""

import ipywidgets as widgets
from IPython.display import display, HTML

def run_cell():
    """Runner utama untuk cell instalasi dependencies"""
    try:
        from smartcash.ui.setup.dependency_installer_component import create_dependency_installer_ui
        from smartcash.ui.setup.dependency_installer_handler import setup_dependency_installer_handlers
    except ImportError as e:
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
    
    # Tambahkan logger jika tersedia
    try:
        from smartcash.common.logger import get_logger
        logger = get_logger("dependency_installer")
        ui_components['logger'] = logger
        logger.info("🚀 Cell dependency_installer diinisialisasi")
    except ImportError:
        pass
    
    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components

# Jalankan cell
run_cell()