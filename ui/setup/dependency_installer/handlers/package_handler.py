"""
File: smartcash/ui/setup/dependency_installer/handlers/package_handler.py
Deskripsi: Wrapper untuk kompatibilitas dengan implementasi lama, menggunakan package_installer.py
"""

import time
import subprocess
import sys
from typing import Dict, Any, List, Tuple
from smartcash.ui.setup.dependency_installer.utils.package_installer import install_package, install_packages_parallel
from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import analyze_installed_packages

def get_all_missing_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Dapatkan semua package yang perlu diinstall (wrapper untuk kompatibilitas)"""
    # Jalankan analisis jika belum ada hasil analisis
    if 'analysis_result' not in ui_components:
        analyze_installed_packages(ui_components)
    
    # Dapatkan package yang perlu diinstall dari hasil analisis
    missing_packages = ui_components.get('analysis_categories', {}).get('missing', [])
    upgrade_packages = ui_components.get('analysis_categories', {}).get('upgrade', [])
    
    # Gabungkan package yang perlu diinstall
    packages_to_install = missing_packages + upgrade_packages
    
    # Log hasil analisis
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"✅ Analisis selesai: {len(packages_to_install)} package perlu diinstall", "success")
    
    return packages_to_install

def run_batch_installation(packages: List[str], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Jalankan instalasi batch untuk package (wrapper untuk kompatibilitas)"""
    if not packages:
        return {'total': 0, 'success': 0, 'failed': 0, 'duration': 0, 'errors': []}
    
    # Initialize stats
    start_time = time.time()
    
    # Install package secara parallel menggunakan fungsi dari package_installer.py
    results = install_packages_parallel(packages, ui_components)
    
    # Hitung statistik
    stats = {
        'total': len(packages),
        'success': sum(1 for success in results.values() if success),
        'failed': sum(1 for success in results.values() if not success),
        'duration': time.time() - start_time,
        'errors': [(pkg, "Instalasi gagal") for pkg, success in results.items() if not success]
    }
    
    return stats
