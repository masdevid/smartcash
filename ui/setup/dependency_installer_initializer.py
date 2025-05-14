"""
File: smartcash/ui/setup/dependency_installer_initializer.py
Deskripsi: Initializer untuk instalasi dependencies dengan alur otomatis yang lebih robust
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.setup.dependency_installer_component import create_dependency_installer_ui
from smartcash.ui.setup.dependency_installer_handler import setup_dependency_installer_handlers

def setup_dependency_installer_specific(ui_components: Dict[str, Any], env: Any, config: Any) -> Dict[str, Any]:
    """Setup handler spesifik untuk dependency installer"""
    # Deteksi packages yang sudah terinstall dan siapkan instalasi
    try:
        analyze_func = ui_components.get('analyze_installed_packages')
        if analyze_func and callable(analyze_func):
            analyze_func(ui_components)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].warning(f"⚠️ Gagal mendeteksi packages otomatis: {str(e)}")
    
    # Setup handler spesifik
    return setup_dependency_installer_handlers(ui_components, env, config)

def initialize_dependency_installer() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk instalasi dependencies.
    
    Returns:
        Dictionary UI components
    """
    # Tombol yang perlu diattach dengan ui_components
    button_keys = ['install_button', 'check_button', 'reset_button']
    
    # Gunakan base initializer
    return initialize_module_ui(
        module_name='dependency_installer',
        create_ui_func=create_dependency_installer_ui,
        setup_specific_handlers_func=setup_dependency_installer_specific,
        button_keys=button_keys
    )