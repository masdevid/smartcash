"""
File: smartcash/ui/setup/dependency_installer/utils/package_installer.py
Deskripsi: Utilitas untuk menginstal package yang dibutuhkan
"""

from typing import Dict, Any, List, Set
import subprocess
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import fungsi UI dan konstanta terpusat
from smartcash.ui.setup.dependency_installer.utils.ui_utils import update_status_panel, update_package_status
from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config, get_package_status

# Setup logger
logger = logging.getLogger(__name__)

def install_package(package: str, ui_components: Dict[str, Any] = None) -> bool:
    """Install package dengan pip
    
    Args:
        package: Package yang akan diinstall
        ui_components: Dictionary komponen UI untuk logging
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Dapatkan konfigurasi untuk level info
        info_config = get_status_config('info')
        success_config = get_status_config('success')
        error_config = get_status_config('error')
        
        # Log instalasi
        if ui_components and 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](f"{info_config['emoji']} Menginstall {package}...", "info")
        
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
                ui_components['log_message'](f"{success_config['emoji']} Berhasil menginstall {package}", "success")
            return True
        else:
            if ui_components and 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](f"{error_config['emoji']} Gagal menginstall {package}: {stderr}", "error")
            return False
    
    except Exception as e:
        # Log error
        if ui_components and 'log_message' in ui_components and callable(ui_components['log_message']):
            error_config = get_status_config('error')
            ui_components['log_message'](f"{error_config['emoji']} Error saat menginstall {package}: {str(e)}", "error")
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
    
    # Dapatkan konfigurasi untuk level info
    info_config = get_status_config('info')
    success_config = get_status_config('success')
    warning_config = get_status_config('warning')
    error_config = get_status_config('error')
    
    # Log mulai instalasi dengan emoji dan highlight
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"{info_config['emoji']} Memulai instalasi {total_packages} packages...", "info")
    
    # Tampilkan progress tracker untuk operasi install
    if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
        ui_components['show_for_operation']('install')
    
    # Reset progress bar dengan pesan yang jelas
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, f"Menginstall {total_packages} packages...", True)
    
    # Inisialisasi progress untuk overall dan step
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('overall', 0, f"Memulai instalasi {total_packages} packages", info_config['border'])
        ui_components['update_progress']('step', 0, "Mempersiapkan instalasi...", info_config['border'])
    
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
                    
                    # Update status package di UI
                    # Cari package_key berdasarkan nama package
                    for category in ui_components.get('categories', []):
                        for pkg in category.get('packages', []):
                            if pkg.get('name', '').lower() == package.lower() or pkg.get('key', '') == package:
                                # Update status widget dengan icon dan warna yang sesuai
                                status_type = "success" if success else "error"
                                message = "Terinstall" if success else "Gagal install"
                                update_package_status(ui_components, pkg['key'], status_type, message)
                    
                    # Update progress overall
                    completed += 1
                    progress_percent = int((completed / total_packages) * 100)
                    
                    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                        # Update step progress dengan warna sesuai status
                        color = success_config['border'] if success else error_config['border']
                        ui_components['update_progress'](
                            'step', 
                            progress_percent, 
                            f"Menginstall package ({completed}/{total_packages}): {package}", 
                            color
                        )
                        
                        # Update overall progress
                        ui_components['update_progress'](
                            'overall',
                            progress_percent,
                            f"Progress instalasi: {completed}/{total_packages}",
                            info_config['border']
                        )
                        
                        # Jika ada current progress, update juga
                        if 'current' in ui_components.get('active_bars', []):
                            status = "✅ Berhasil" if success else "❌ Gagal"
                            ui_components['update_progress'](
                                'current',
                                100 if success else 0,
                                f"{package}: {status}",
                                color
                            )
                    
                except Exception as e:
                    # Log error dan tandai package sebagai gagal
                    if 'log_message' in ui_components and callable(ui_components['log_message']):
                        ui_components['log_message'](f"❌ Error saat menginstall {package}: {str(e)}", "error")
                    results[package] = False
        
        # Hitung statistik
        success_count = sum(1 for success in results.values() if success)
        
        # Log hasil instalasi dengan emoji dan highlight
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            level = "success" if success_count == total_packages else "warning"
            icon = "✅" if success_count == total_packages else "⚠️"
            ui_components['log_message'](
                f"{icon} Hasil instalasi: {success_count}/{total_packages} packages berhasil diinstall",
                level
            )
        
        # Update status panel dengan pesan yang jelas menggunakan fungsi yang telah diperbarui
        if success_count == total_packages:
            update_status_panel(ui_components, "success", f"✅ Semua {total_packages} packages berhasil diinstall")
        else:
            update_status_panel(ui_components, "warning", f"⚠️ {success_count}/{total_packages} packages berhasil diinstall")
        
        # Complete operation jika semua berhasil, atau update progress selesai
        if success_count == total_packages and 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
            ui_components['complete_operation'](f"Semua {total_packages} packages berhasil diinstall")
        elif 'update_progress' in ui_components and callable(ui_components['update_progress']):
            # Update step progress selesai
            ui_components['update_progress']('step', 100, "Instalasi selesai", success_config['border'])
            # Update overall progress selesai
            ui_components['update_progress']('overall', 100, f"{success_count}/{total_packages} packages berhasil diinstall", success_config['border'])
        
        return results
    
    except Exception as e:
        # Log error dengan format yang jelas
        error_message = f"Gagal menginstall packages: {str(e)}"
        
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](f"❌ {error_message}", "error")
        
        # Update status panel dengan pesan error menggunakan fungsi yang telah diperbarui
        update_status_panel(ui_components, "error", f"❌ {error_message}")
        
        # Update status semua package yang belum diproses menjadi error
        for category in ui_components.get('categories', []):
            for pkg in category.get('packages', []):
                if pkg.get('name', '').lower() in packages or pkg.get('key', '') in packages:
                    if pkg.get('key', '') not in [k for k, v in results.items() if v]:
                        update_package_status(ui_components, pkg['key'], "error", "Gagal install")
        
        # Tandai error pada progress tracker
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](error_message)
        
        return {package: False for package in packages}

def install_required_packages(ui_components: Dict[str, Any]) -> None:
    """Install package yang dibutuhkan berdasarkan hasil analisis
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Import logger untuk keperluan logging
    from smartcash.common.logger import get_logger
    from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import analyze_installed_packages
    logger = get_logger('dependency_installer')
    
    # Log mulai proses dengan format yang jelas
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        info_config = get_status_config('info')
        ui_components['log_message'](f"{info_config['emoji']} 🔍 Memeriksa package yang perlu diinstall...", "info")
    
    # Jika hasil analisis tidak ada atau telah direset, jalankan analisis terlebih dahulu
    if 'analysis_result' not in ui_components or not ui_components['analysis_result']:
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            info_config = get_status_config('info')
            ui_components['log_message'](f"{info_config['emoji']} 🔄 Menjalankan analisis ulang untuk memastikan status terbaru...", "info")
        
        # Update status panel
        update_status_panel(ui_components, "info", f"{info_config['emoji']} 🔄 Menjalankan analisis ulang...")
        
        # Jalankan analisis package
        try:
            analyze_installed_packages(ui_components)
        except Exception as e:
            # Dapatkan konfigurasi untuk level error
            error_config = get_status_config('error')
            
            error_message = f"{error_config['emoji']} Gagal menjalankan analisis: {str(e)}"
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](error_message, "error")
            
            # Update status panel dengan pesan error
            update_status_panel(ui_components, "error", error_message)
            
            # Tandai error pada progress tracker
            if 'error_operation' in ui_components and callable(ui_components['error_operation']):
                ui_components['error_operation'](error_message)
            
            return
    
    # Cek apakah ada hasil analisis setelah menjalankan analisis ulang
    if 'analysis_result' not in ui_components or not ui_components['analysis_result']:
        # Dapatkan konfigurasi untuk level error
        error_config = get_status_config('error')
        
        error_message = f"{error_config['emoji']} Tidak ada hasil analisis. Silakan jalankan analisis terlebih dahulu."
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](error_message, "error")
        
        # Update status panel dengan pesan error
        update_status_panel(ui_components, "error", error_message)
        
        # Tandai error pada progress tracker
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](error_message)
        
        return
    
    # Dapatkan package yang perlu diinstall
    analysis_result = ui_components['analysis_result']
    packages_to_install = analysis_result.get('missing_packages', [])
    
    # Jika tidak ada package yang perlu diinstall
    if not packages_to_install:
        # Dapatkan konfigurasi untuk level success
        success_config = get_status_config('success')
        
        success_message = f"{success_config['emoji']} Semua package sudah terinstall dengan benar"
        # Tidak ada package yang perlu diinstall
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](success_message, "success")
        
        # Update status panel dengan pesan sukses menggunakan fungsi yang telah diperbarui
        update_status_panel(ui_components, "success", success_message)
        
        # Update status semua package menjadi sukses
        for category in ui_components.get('categories', []):
            for pkg in category.get('packages', []):
                update_package_status(ui_components, pkg['key'], "success", "Terinstall")
        
        # Complete operation jika tersedia
        if 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
            ui_components['complete_operation']("Semua package sudah terinstall dengan benar")
        
        return
    
    # Log jumlah package yang akan diinstall dengan format yang jelas
    package_count = len(packages_to_install)
    logger.info(f"Installing {package_count} packages")
    
    # Dapatkan konfigurasi untuk level info
    info_config = get_status_config('info')
    
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"{info_config['emoji']} Menemukan {package_count} package yang perlu diinstall", "info")
    
    # Tampilkan progress tracker untuk operasi install
    if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
        ui_components['show_for_operation']('install')
    
    # Reset progress bar dengan pesan yang jelas
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, f"Bersiap menginstall {package_count} packages...", True)
    
    # Install package secara parallel dengan progress tracking
    install_packages_parallel(packages_to_install, ui_components)
