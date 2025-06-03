"""
File: smartcash/ui/setup/dependency_installer/utils/package_manager.py
Deskripsi: Wrapper untuk kompatibilitas dengan implementasi lama, menggunakan analyzer_utils.py dan package_installer.py
"""

from typing import Dict, Any, List, Tuple
from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import (
    get_installed_packages, parse_requirement, check_version_compatibility as check_package_version,
    analyze_requirements
)
from smartcash.ui.setup.dependency_installer.utils.package_installer import (
    install_package, install_packages_parallel
)

# Fungsi-fungsi berikut adalah wrapper untuk kompatibilitas dengan implementasi lama
# Implementasi sebenarnya ada di analyzer_utils.py dan package_installer.py

def install_packages(packages: List[str], ui_components: Dict[str, Any] = None, parallel: bool = False) -> Dict[str, bool]:
    """Install multiple packages (wrapper untuk kompatibilitas)
    
    Args:
        packages: List package yang akan diinstall
        ui_components: Dictionary komponen UI untuk logging (opsional)
        parallel: True untuk instalasi paralel, False untuk sekuensial
        
    Returns:
        Dictionary dengan nama package sebagai key dan status instalasi sebagai value
    """
    if not packages:
        return {}
    
    # Update progress
    if ui_components and 'show_for_operation' in ui_components:
        ui_components['show_for_operation']('install')
    
    # Gunakan implementasi baru dari package_installer.py
    if parallel:
        return install_packages_parallel(packages, ui_components)
    
    # Instalasi sekuensial
    results = {}
    total = len(packages)
    
    for i, package in enumerate(packages):
        # Update progress
        if ui_components and 'update_progress' in ui_components:
            progress = int((i / total) * 100)
            ui_components['update_progress']('overall', progress, f"Menginstall {package}...")
        
        # Install package menggunakan implementasi baru
        results[package] = install_package(package, ui_components)
    
    # Update progress selesai
    if ui_components and 'update_progress' in ui_components:
        ui_components['update_progress']('overall', 100, "Instalasi selesai")
    
    return results
