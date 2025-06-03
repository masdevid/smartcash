"""
File: smartcash/ui/setup/dependency_installer/handlers/install_handler.py
Deskripsi: Wrapper untuk kompatibilitas dengan implementasi lama, menggunakan package_installer.py
"""

from typing import Dict, Any, List
import time
from IPython.display import display, clear_output
from smartcash.ui.setup.dependency_installer.utils.package_installer import install_required_packages
from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import analyze_installed_packages
from smartcash.ui.setup.dependency_installer.handlers.package_handler import run_batch_installation, get_all_missing_packages

def setup_install_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol install (wrapper untuk kompatibilitas)"""
    ui_components['install_button'].on_click(lambda b: on_install_click(b, ui_components))

def on_install_click(b, ui_components: Dict[str, Any]) -> None:
    """Handler untuk tombol install (wrapper untuk kompatibilitas)"""
    # Import dari existing implementations
    from smartcash.ui.utils.alert_utils import create_info_alert
    
    # Reset log output
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output()
    
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.visibility = 'visible'
    
    # Log mulai instalasi
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message']("🚀 Memulai proses instalasi dependency", "info")
    
    # Jalankan analisis jika belum ada hasil analisis
    if 'analysis_result' not in ui_components:
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message']("🔍 Menjalankan analisis package terlebih dahulu...", "info")
        analyze_installed_packages(ui_components)
    
    # Dapatkan package yang perlu diinstall
    packages_to_install = get_all_missing_packages(ui_components)
    
    # Install package yang dibutuhkan menggunakan run_batch_installation untuk kompatibilitas dengan test
    run_batch_installation(packages_to_install, ui_components)
    
    # Juga panggil implementasi baru untuk memastikan fungsionalitas lengkap
    install_required_packages(ui_components)
