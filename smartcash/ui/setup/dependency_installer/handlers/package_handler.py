"""
File: smartcash/ui/setup/dependency_installer/handlers/package_handler.py
Deskripsi: Fixed package handler menggunakan existing implementations
"""

import time
import subprocess
import sys
from typing import Dict, Any, List, Tuple
from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message

def get_all_missing_packages(ui_components: Dict[str, Any]) -> List[str]:
    """Dapatkan semua package yang perlu diinstall berdasarkan UI state"""
    from smartcash.ui.setup.dependency_installer.utils.package_utils import (
        get_package_groups, parse_custom_packages, get_installed_packages, check_missing_packages
    )
    
    installed_packages = get_installed_packages()
    package_groups = get_package_groups()
    missing_packages = []
    
    # Add selected packages dari groups
    for pkg_key, pkg_list in package_groups.items():
        checkbox = ui_components.get(pkg_key)
        if checkbox and checkbox.value:
            package_list = pkg_list() if callable(pkg_list) else pkg_list
            missing = check_missing_packages(package_list, installed_packages)
            missing_packages.extend(missing)
    
    # Add custom packages
    custom_text = ui_components.get('custom_packages', type('', (), {'value': ''})).value.strip()
    if custom_text:
        custom_packages = parse_custom_packages(custom_text)
        missing = check_missing_packages(custom_packages, installed_packages)
        missing_packages.extend(missing)
    
    return list(dict.fromkeys(missing_packages))  # Remove duplicates

def run_batch_installation(packages: List[str], ui_components: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Jalankan instalasi batch untuk package"""
    if not packages:
        return True, {'total': 0, 'success': 0, 'failed': 0, 'duration': 0, 'errors': []}
    
    # Get progress tracker
    progress_tracker = ui_components.get('progress_tracker')
    
    # Initialize stats
    stats = {'total': len(packages), 'success': 0, 'failed': 0, 'duration': 0, 'errors': []}
    start_time = time.time()
    
    log_message(ui_components, f"🚀 Memulai instalasi {len(packages)} package...", "info")
    
    # Install each package
    for i, package in enumerate(packages):
        progress_pct = int((i / len(packages)) * 100)
        
        # Update progress tracker
        if progress_tracker:
            progress_tracker.update('overall', progress_pct, f"Installing {package}...")
            progress_tracker.update('current', 0, f"Package {i+1}/{len(packages)}")
        
        log_message(ui_components, f"📦 Installing {package}...", "info")
        
        try:
            # Run pip install silently
            cmd = [sys.executable, "-m", "pip", "install", package, "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                stats['success'] += 1
                log_message(ui_components, f"✅ Berhasil install {package}", "success")
                if progress_tracker:
                    progress_tracker.update('current', 100, f"✅ {package}")
            else:
                stats['failed'] += 1
                error_msg = result.stderr.strip() or "Unknown error"
                stats['errors'].append((package, error_msg))
                log_message(ui_components, f"❌ Gagal install {package}: {error_msg}", "error")
                if progress_tracker:
                    progress_tracker.update('current', 100, f"❌ {package}")
        
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append((package, str(e)))
            log_message(ui_components, f"💥 Error install {package}: {str(e)}", "error")
            if progress_tracker:
                progress_tracker.update('current', 100, f"💥 {package}")
    
    # Final progress update
    if progress_tracker:
        progress_tracker.update('overall', 100, "Installation complete")
    
    stats['duration'] = time.time() - start_time
    
    log_message(ui_components, f"📊 Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal ({stats['duration']:.1f}s)", "success")
    
    return stats['failed'] == 0, stats