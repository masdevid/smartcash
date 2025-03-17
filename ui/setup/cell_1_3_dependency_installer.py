"""
File: smartcash/ui/setup/cell_1_3_dependency_installer.py
Deskripsi: Cell instalasi dependencies untuk SmartCash dengan pendekatan modular
"""

def run_cell(config_path="configs/colab_config.yaml"):
    """
    Runner utama untuk cell instalasi dependencies
    
    Args:
        config_path: Path ke file konfigurasi
    """
    from smartcash.ui.setup.dependency_installer_component import create_dependency_installer_ui
    from smartcash.ui.setup.dependency_installer_handler import setup_dependency_installer_handlers
    from smartcash.ui.utils.ui_helpers import inject_css_styles
    from smartcash.ui.handlers.error_handler import setup_error_handlers
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    from IPython.display import display

    try:
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        env = get_environment_manager()
        config_manager = get_config_manager()
        config = config_manager.config
    except ImportError:
        env, config = None, {}

    # Inject konsisten styling
    inject_css_styles()

    # Buat komponen UI
    ui_components = create_dependency_installer_ui(env, config)
    
    # Setup handlers
    ui_components = setup_error_handlers(ui_components)
    ui_components = setup_observer_handlers(ui_components, "dependency_installer_observers")
    ui_components = setup_dependency_installer_handlers(ui_components, config)

    # Tampilkan UI
    display(ui_components['ui'])
    
    return ui_components

# Entry point
if __name__ == "__main__":
    run_cell()