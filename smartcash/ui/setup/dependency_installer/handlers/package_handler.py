"""
File: smartcash/ui/setup/dependency_installer/handlers/package_handler.py
Deskripsi: Enhanced package handler dengan progress tracking yang lebih informatif dan pendekatan DRY
"""

import time
import subprocess
import sys
from typing import Dict, Any, List, Tuple
from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message
from smartcash.ui.setup.dependency_installer.utils.observer_helper import (
    notify_install_start, notify_install_progress, notify_install_error, notify_install_complete
)
from smartcash.ui.setup.dependency_installer.utils.progress_helper import (
    update_progress_step, update_overall_progress, update_current_progress,
    calculate_batch_progress, start_operation, complete_operation,
    handle_item_error, handle_item_success
)

def get_all_missing_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Dapatkan semua package yang perlu diinstall berdasarkan UI state dengan pendekatan DRY"""
    from smartcash.ui.setup.dependency_installer.utils.package_utils import (
        get_package_groups, parse_custom_packages, get_installed_packages, check_missing_packages
    )
    
    # Update progress step dengan pendekatan DRY
    update_progress_step(ui_components, 25, "Menganalisis packages...", "#007bff")
    log_message(ui_components, "🔍 Menganalisis packages yang diperlukan...", "info")
    
    # Dapatkan data packages dengan pendekatan one-liner
    installed_packages = get_installed_packages()
    package_groups = get_package_groups()
    missing_packages = []
    
    # Add selected packages dari groups dengan pendekatan one-liner
    for pkg_key, pkg_list in package_groups.items():
        checkbox = ui_components.get(pkg_key)
        if checkbox and checkbox.value:
            package_list = pkg_list() if callable(pkg_list) else pkg_list
            missing_packages.extend(check_missing_packages(package_list, installed_packages))
    
    # Add custom packages dengan pendekatan one-liner
    custom_text = ui_components.get('custom_packages', type('', (), {'value': ''})).value.strip()
    missing_packages.extend(check_missing_packages(parse_custom_packages(custom_text), installed_packages)) if custom_text else None
    
    # Update progress step - Analisis selesai dengan pendekatan DRY
    update_progress_step(ui_components, 50, "Analisis selesai", "#28a745")
    log_message(ui_components, f"✅ Analisis selesai: {len(missing_packages)} package perlu diinstall", "success")
    
    return list(dict.fromkeys(missing_packages))  # Remove duplicates

def run_batch_installation(packages: List[str], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Jalankan instalasi batch untuk package dengan progress tracking yang lebih informatif dan observer pattern menggunakan pendekatan DRY"""
    if not packages:
        return {'total': 0, 'success': 0, 'failed': 0, 'duration': 0, 'errors': []}
    
    # Initialize stats dengan pendekatan one-liner
    stats = {'total': len(packages), 'success': 0, 'failed': 0, 'duration': 0, 'errors': []}
    start_time = time.time()
    
    # Mulai operasi dengan progress tracking dan logging menggunakan pendekatan DRY
    start_operation(ui_components, "instalasi package", len(packages))
    
    # Notify start via observer dengan pendekatan DRY
    notify_install_start(ui_components, len(packages))
    
    # Install each package dengan pendekatan DRY
    for i, package in enumerate(packages):
        # Hitung progress dan update dengan pendekatan DRY
        progress_pct = calculate_batch_progress(i, len(packages))
        progress_message = f"Menginstall package {i+1}/{len(packages)}"
        
        # Update progress tracker dengan pendekatan DRY
        update_overall_progress(ui_components, progress_pct, progress_message)
        update_current_progress(ui_components, 0, f"Menginstall {package}...")
        
        # Notify progress via observer dengan pendekatan DRY
        notify_install_progress(ui_components, package, i, len(packages))
        
        try:
            # Install package dengan subprocess
            cmd = [sys.executable, "-m", "pip", "install", package]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check result dengan pendekatan DRY
            if result.returncode == 0:
                # Handle success dengan pendekatan DRY
                stats['success'] += 1
                handle_item_success(ui_components, package)
            else:
                # Handle error dengan pendekatan DRY
                stats['failed'] += 1
                error_msg = result.stderr.strip() or "Unknown error"
                stats['errors'].append((package, error_msg))
                handle_item_error(ui_components, package, error_msg)
                
                # Notify error via observer dengan pendekatan DRY
                notify_install_error(ui_components, package, error_msg)
        
        except Exception as e:
            # Handle exception dengan pendekatan DRY
            stats['failed'] += 1
            error_msg = str(e)
            stats['errors'].append((package, error_msg))
            handle_item_error(ui_components, package, error_msg)
            
            # Notify error via observer dengan pendekatan DRY
            notify_install_error(ui_components, package, error_msg)
    
    # Update stats dengan pendekatan one-liner
    stats['duration'] = time.time() - start_time
    
    # Selesaikan operasi dengan progress tracking dan logging menggunakan pendekatan DRY
    complete_operation(ui_components, "instalasi package", stats)
    
    # Notify complete via observer dengan pendekatan DRY
    notify_install_complete(ui_components, stats)
    
    return stats