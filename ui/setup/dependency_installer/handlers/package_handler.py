"""
File: smartcash/ui/setup/dependency_installer/handlers/package_handler.py
Deskripsi: Handler untuk manajemen package
"""

import time
import subprocess
import sys
import contextlib
from typing import Dict, Any, List, Tuple
from tqdm.notebook import tqdm
from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message

@contextlib.contextmanager
def disable_tqdm():
    """Context manager to temporarily disable tqdm output"""
    old_disable = getattr(tqdm, 'disable', False)
    tqdm.disable = True
    try:
        yield
    finally:
        tqdm.disable = old_disable

class SilentTqdm(tqdm):
    """Custom tqdm class that doesn't display to console"""
    def __init__(self, *args, **kwargs):
        kwargs['display'] = False  # Disable display
        kwargs['disable'] = True   # Completely disable tqdm
        super().__init__(*args, **kwargs)
    
    def display(self, *args, **kwargs):
        """Override display to do nothing"""
        pass

def get_all_missing_packages(ui_components: Dict[str, Any]) -> List[str]:
    """
    Dapatkan semua package yang perlu diinstall berdasarkan UI state
    
    Args:
        ui_components: Dictionary UI components
        
    Returns:
        List package yang perlu diinstall
    """
    from smartcash.ui.setup.dependency_installer.utils.package_utils import (
        get_package_groups, parse_custom_packages, get_installed_packages, check_missing_packages
    )
    
    # Get installed packages
    installed_packages = get_installed_packages()
    
    # Dapatkan package groups
    PACKAGE_GROUPS = get_package_groups()
    
    # Collect packages to install
    missing_packages = []
    
    # Add selected packages from groups
    for pkg_key, pkg_list in PACKAGE_GROUPS.items():
        checkbox = ui_components.get(pkg_key)
        if checkbox and checkbox.value:
            package_list = pkg_list() if callable(pkg_list) else pkg_list
            missing = check_missing_packages(package_list, installed_packages)
            missing_packages.extend(missing)
    
    # Add custom packages
    custom_text = ui_components.get('custom_packages').value.strip()
    if custom_text:
        custom_packages = parse_custom_packages(custom_text)
        missing = check_missing_packages(custom_packages, installed_packages)
        missing_packages.extend(missing)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(missing_packages))

def run_batch_installation(packages: List[str], ui_components: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Jalankan instalasi batch untuk package
    
    Args:
        packages: List package yang akan diinstall
        ui_components: Dictionary UI components
        
    Returns:
        Tuple (success, stats)
    """
    # Jika tidak ada package yang perlu diinstall
    if not packages:
        return True, {
            'total': 0,
            'success': 0,
            'failed': 0,
            'duration': 0,
            'errors': []
        }
    
    # Dapatkan komponen UI
    progress_bar = ui_components.get('install_progress')
    progress_label = ui_components.get('progress_label')
    tracker = ui_components.get('dependency_installer_tracker')
    
    # Statistik instalasi
    stats = {
        'total': len(packages),
        'success': 0,
        'failed': 0,
        'duration': 0,
        'errors': []
    }
    
    # Waktu mulai
    start_time = time.time()
    
    # Log info ke UI dengan namespace khusus
    log_message(ui_components, f"Memulai instalasi {len(packages)} package...", "info")
    
    # Gunakan context manager untuk menonaktifkan tqdm output
    with disable_tqdm():
        # Gunakan SilentTqdm untuk progress bar yang tidak menampilkan ke console
        for i, package in enumerate(SilentTqdm(packages, desc="Instalasi Package")):
            # Update progress
            progress_pct = int((i / len(packages)) * 100)
            if progress_bar: progress_bar.value = progress_pct
            if progress_label: progress_label.value = f"Menginstall {package}... ({i+1}/{len(packages)})"
            if tracker: tracker.update(progress_pct)
            
            # Log info ke UI dengan namespace khusus
            log_message(ui_components, f"Menginstall {package}...", "info")
            
            try:
                # Jalankan pip install dengan --quiet untuk mengurangi output
                cmd = [sys.executable, "-m", "pip", "install", package, "--quiet"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Sukses
                    stats['success'] += 1
                    log_message(ui_components, f"Berhasil menginstall {package}", "success")
                else:
                    # Gagal
                    stats['failed'] += 1
                    error_msg = result.stderr.strip() or "Unknown error"
                    stats['errors'].append((package, error_msg))
                    log_message(ui_components, f"Gagal menginstall {package}: {error_msg}", "error")
            except Exception as e:
                # Error
                stats['failed'] += 1
                stats['errors'].append((package, str(e)))
                log_message(ui_components, f"Error saat menginstall {package}: {str(e)}", "error")
    
    # Update progress ke 100%
    if progress_bar: progress_bar.value = 100
    if progress_label: progress_label.value = f"Selesai menginstall {len(packages)} package"
    if tracker: tracker.update(100)
    
    # Hitung durasi
    stats['duration'] = time.time() - start_time
    
    # Log ringkasan ke UI dengan namespace khusus
    log_message(ui_components, f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal", "success")
    log_message(ui_components, f"Waktu: {stats['duration']:.1f} detik", "info")
    
    # Return success jika semua berhasil
    return stats['failed'] == 0, stats
