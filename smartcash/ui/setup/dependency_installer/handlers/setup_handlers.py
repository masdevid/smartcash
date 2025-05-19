"""
File: smartcash/ui/setup/dependency_installer/handlers/setup_handlers.py
Deskripsi: Setup handlers untuk UI dependency installer
"""

from typing import Dict, Any

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handlers untuk UI dependency installer
    
    Args:
        ui_components: Dictionary UI components
        env: Environment manager (opsional)
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Dictionary UI components yang diupdate
    """
    # Setup handlers
    from smartcash.ui.setup.dependency_installer.handlers.install_handler import setup_install_handler
    from smartcash.ui.setup.dependency_installer.handlers.analyzer_handler import setup_analyzer_handler
    from smartcash.ui.setup.dependency_installer.utils.package_utils import analyze_installed_packages
    
    # Setup progress tracking
    from smartcash.ui.handlers.single_progress import setup_progress_tracking
    
    # Setup progress tracking
    tracker = setup_progress_tracking(
        ui_components, 
        tracker_name="dependency_installer",
        progress_widget_key="install_progress",
        progress_label_key="progress_label",
        total=100,
        description="Instalasi dependencies"
    )
    
    # Pastikan tracker tersedia di ui_components dengan kunci yang benar
    if tracker and 'dependency_installer_tracker' not in ui_components:
        ui_components['dependency_installer_tracker'] = tracker
    
    # Setup handlers
    setup_install_handler(ui_components)
    setup_analyzer_handler(ui_components)
    
    # Deteksi packages yang sudah terinstall dan siapkan instalasi
    try:
        analyze_installed_packages(ui_components)
    except Exception as e:
        if 'logger' in ui_components:
            ui_components['logger'].warning(f"⚠️ Gagal mendeteksi packages otomatis: {str(e)}")
    
    return ui_components
