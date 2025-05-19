"""
File: smartcash/ui/setup/utils/package_installer.py
Deskripsi: Utilitas untuk instalasi package dengan progress tracking
"""

import sys
import time
import subprocess
from typing import Dict, List, Tuple, Any
from IPython.display import display

def install_single_package(package: str) -> Tuple[bool, str]:
    """
    Install package tunggal dengan pip.
    
    Args:
        package: Nama package dengan atau tanpa versi
        
    Returns:
        Tuple (success, error_message)
    """
    # Skip instalasi tqdm
    if package.lower().startswith('tqdm'):
        return True, "Instalasi tqdm dilewati sesuai konfigurasi"
        
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            capture_output=True, text=True, check=False
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)

def run_batch_installation(
    packages: List[str], 
    ui_components: Dict[str, Any], 
    update_status: callable = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Jalankan instalasi batch packages dengan progress tracking.
    
    Args:
        packages: List package yang akan diinstall
        ui_components: Dictionary UI components
        update_status: Fungsi untuk update status (opsional)
        
    Returns:
        Tuple (success, stats)
    """
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    # Status output dan progress widgets
    status_output = ui_components.get('status')
    progress_bar = ui_components.get('install_progress')
    progress_label = ui_components.get('progress_label')
    logger = ui_components.get('logger')
    
    # Filter out tqdm dari packages yang akan diinstal
    filtered_packages = [pkg for pkg in packages if not pkg.lower().startswith('tqdm')]
    
    # Hitung berapa package yang dilewati
    skipped_count = len(packages) - len(filtered_packages)
    
    # Stats untuk instalasi
    stats = {
        'total': len(packages),
        'success': 0,
        'failed': 0,
        'skipped': skipped_count,  # Inisialisasi dengan jumlah paket yang dilewati
        'duration': 0,
        'errors': []
    }
    
    # Log packages yang dilewati
    if skipped_count > 0 and logger:
        logger.info(f"ℹ️ Melewati instalasi {skipped_count} package (tqdm)")  
    
    # Gunakan filtered_packages untuk instalasi
    packages = filtered_packages
    
    if not packages:
        if logger: logger.info("ℹ️ Tidak ada package yang perlu diinstall")
        return True, stats
    
    # Dapatkan tracker
    tracker_key = 'dependency_installer_tracker'
    tracker = ui_components.get(tracker_key)
    
    # Setup progress
    if progress_bar and hasattr(progress_bar, 'max'):
        progress_bar.max = len(packages)
        progress_bar.value = 0
        # Pastikan progress bar terlihat
        if hasattr(progress_bar, 'layout') and hasattr(progress_bar.layout, 'visibility'):
            progress_bar.layout.visibility = 'visible'
        
    if progress_label:
        progress_label.value = "Memulai instalasi packages..."
        # Pastikan label terlihat
        if hasattr(progress_label, 'layout') and hasattr(progress_label.layout, 'visibility'):
            progress_label.layout.visibility = 'visible'
    
    # Track start time
    start_time = time.time()
    
    # Setup tracker jika tersedia
    if tracker:
        tracker.set_total(len(packages))
        tracker.current = 0
        tracker.desc = "Memulai instalasi packages..."
    
    # Install packages satu per satu
    for i, package in enumerate(packages):
        # Extract package name for display
        package_name = package.split('=')[0].split('>')[0].split('<')[0].strip()
        
        # Update progress label
        if progress_label:
            progress_label.value = f"Menginstall {package_name}... ({i+1}/{len(packages)})"
        
        # Display info di status output
        if status_output:
            with status_output:
                display(create_status_indicator('info', f"Menginstall {package}"))
        
        # Install package
        success, error_msg = install_single_package(package)
        
        # Update stats
        if success:
            stats['success'] += 1
            if status_output:
                with status_output:
                    display(create_status_indicator('success', f"{package} berhasil diinstall"))
        else:
            stats['failed'] += 1
            stats['errors'].append((package, error_msg))
            if status_output:
                with status_output:
                    display(create_status_indicator('error', f"Gagal menginstall {package}: {error_msg}"))
        
        # Update progress tracker jika tersedia
        if tracker:
            tracker.update(1, f"Installed {i+1}/{len(packages)}: {package_name}")
        # Jika tidak ada tracker, update progress bar langsung
        elif progress_bar and hasattr(progress_bar, 'value'):
            progress_bar.value = i + 1
            if hasattr(progress_bar, 'description'):
                percentage = int((i + 1) / len(packages) * 100)
                progress_bar.description = f"Proses: {percentage}%"
        
        # Custom status update jika disediakan
        if update_status:
            update_status(i+1, len(packages), package_name, success)
    
    # Calculate duration
    stats['duration'] = time.time() - start_time
    
    # Complete progress
    if tracker:
        tracker.complete(f"Instalasi selesai: {stats['success']}/{stats['total']} berhasil")
    elif progress_bar:
        progress_bar.value = len(packages)
        if hasattr(progress_bar, 'description'):
            progress_bar.description = f"Proses: 100%"
    
    # Update progress label
    if progress_label:
        progress_label.value = f"Instalasi selesai dalam {stats['duration']:.1f} detik"
        
    # Log completion
    if logger:
        logger.info(f"✅ Instalasi selesai dalam {stats['duration']:.1f} detik: "
                   f"{stats['success']}/{stats['total']} berhasil, {stats['failed']} gagal")
    
    return stats['failed'] == 0, stats