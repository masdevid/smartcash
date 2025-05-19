"""
File: smartcash/ui/setup/dependency_installer/handlers/analyzer_handler.py
Deskripsi: Handler untuk analisis package yang terinstall
"""

from typing import Dict, Any

def setup_analyzer_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk analisis package
    
    Args:
        ui_components: Dictionary UI components
    """
    # Expose function untuk analisis otomatis
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    ui_components['analyze_installed_packages'] = lambda: analyze_installed_packages(ui_components)
