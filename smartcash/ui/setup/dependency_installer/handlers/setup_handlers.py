"""
File: smartcash/ui/setup/dependency_installer/handlers/setup_handlers.py
Deskripsi: Fixed setup handlers menggunakan existing progress tracking
"""

from typing import Dict, Any

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """Setup handlers untuk dependency installer"""
    # Import existing handlers
    from smartcash.ui.setup.dependency_installer.handlers.install_handler import setup_install_handler
    from smartcash.ui.setup.dependency_installer.handlers.analyzer_handler import setup_analyzer_handler
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
    
    # Setup progress tracking menggunakan existing component
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    
    # Create progress tracker
    progress_container = create_progress_tracking_container()
    ui_components.update({
        'progress_tracker': progress_container['tracker'],
        'progress_container': progress_container['container']
    })
    
    # Setup handlers
    setup_install_handler(ui_components)
    setup_analyzer_handler(ui_components)
    
    # Deteksi packages yang sudah terinstall
    try:
        analyze_installed_packages(ui_components)
    except Exception as e:
        log_message(ui_components, f"⚠️ Gagal mendeteksi packages: {str(e)}", "warning")
    
    return ui_components