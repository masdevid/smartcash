"""
File: smartcash/ui/setup/dependency_installer/utils/package_installer.py
Deskripsi: Utilitas untuk menginstal package yang dibutuhkan
"""

from typing import Dict, Any, List, Set
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def install_package(package: str, ui_components: Dict[str, Any] = None) -> bool:
    """Install package dengan pip
    
    Args:
        package: Package yang akan diinstall
        ui_components: Dictionary komponen UI untuk logging
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Log instalasi
        if ui_components and 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](f"📦 Menginstall {package}...", "info")
        
        # Jalankan pip install
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        # Cek hasil instalasi
        if process.returncode == 0:
            if ui_components and 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](f"✅ Berhasil menginstall {package}", "success")
            return True
        else:
            if ui_components and 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](f"❌ Gagal menginstall {package}: {stderr}", "error")
            return False
    
    except Exception as e:
        # Log error
        if ui_components and 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](f"❌ Error saat menginstall {package}: {str(e)}", "error")
        return False

def install_packages_parallel(packages: List[str], ui_components: Dict[str, Any], max_workers: int = 3) -> Dict[str, bool]:
    """Install multiple packages secara parallel dengan ThreadPoolExecutor
    
    Args:
        packages: List package yang akan diinstall
        ui_components: Dictionary komponen UI
        max_workers: Jumlah maksimum worker thread
        
    Returns:
        Dictionary hasil instalasi {package: success}
    """
    results = {}
    total_packages = len(packages)
    
    # Log mulai instalasi
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"🚀 Memulai instalasi {total_packages} packages...", "info")
    
    # Reset progress bar
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, f"Menginstall {total_packages} packages...", True)
    
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.visibility = 'visible'
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit semua tugas instalasi
            future_to_package = {
                executor.submit(install_package, package, ui_components): package 
                for package in packages
            }
            
            # Proses hasil instalasi seiring selesai
            completed = 0
            for future in as_completed(future_to_package):
                package = future_to_package[future]
                try:
                    success = future.result()
                    results[package] = success
                    
                    # Update progress
                    completed += 1
                    progress_percent = int((completed / total_packages) * 100)
                    
                    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                        color = "#28a745" if success else "#dc3545"
                        ui_components['update_progress'](
                            'step', 
                            progress_percent, 
                            f"Menginstall package ({completed}/{total_packages})", 
                            color
                        )
                    
                except Exception as e:
                    # Log error dan tandai package sebagai gagal
                    if 'log_message' in ui_components and callable(ui_components['log_message']):
                        ui_components['log_message'](f"❌ Error saat menginstall {package}: {str(e)}", "error")
                    results[package] = False
        
        # Hitung statistik
        success_count = sum(1 for success in results.values() if success)
        
        # Log hasil instalasi
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](
                f"📊 Hasil instalasi: {success_count}/{total_packages} packages berhasil diinstall",
                "info" if success_count == total_packages else "warning"
            )
        
        # Update status panel
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            if success_count == total_packages:
                ui_components['update_status_panel']("success", f"Semua {total_packages} packages berhasil diinstall")
            else:
                ui_components['update_status_panel'](
                    "warning", 
                    f"{success_count}/{total_packages} packages berhasil diinstall"
                )
        
        # Update progress selesai
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('step', 100, "Instalasi selesai", "#28a745")
        
        return results
    
    except Exception as e:
        # Log error
        error_message = f"Gagal menginstall packages: {str(e)}"
        
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](f"❌ {error_message}", "error")
        
        # Update status panel
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel']("danger", error_message)
        
        # Tandai error pada progress
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](error_message)
        
        return {package: False for package in packages}

def install_required_packages(ui_components: Dict[str, Any]) -> bool:
    """Install package yang dibutuhkan berdasarkan hasil analisis
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika semua package berhasil diinstall, False jika ada yang gagal
    """
    # Cek apakah ada hasil analisis
    if 'analysis_categories' not in ui_components:
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message']("⚠️ Tidak ada hasil analisis package, jalankan analisis terlebih dahulu", "warning")
        return False
    
    # Dapatkan package yang perlu diinstall atau diupgrade
    missing_packages = ui_components['analysis_categories'].get('missing', [])
    upgrade_packages = ui_components['analysis_categories'].get('upgrade', [])
    
    # Gabungkan package yang perlu diinstall
    packages_to_install = missing_packages + upgrade_packages
    
    # Cek apakah ada package yang perlu diinstall
    if not packages_to_install:
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message']("✅ Semua package sudah terinstall dengan benar", "success")
        
        # Update status panel
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel']("success", "Semua package sudah terinstall dengan benar")
        
        return True
    
    # Log instalasi
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"🔄 Menginstall {len(packages_to_install)} package...", "info")
        
    # Install package secara parallel
    results = install_packages_parallel(packages_to_install, ui_components)
    
    # Cek apakah semua package berhasil diinstall
    all_success = all(results.values())
    
    return all_success
