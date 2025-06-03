"""
File: smartcash/ui/setup/dependency_installer/utils/package_installer.py
Deskripsi: Utilitas untuk menginstal package yang dibutuhkan
"""

from typing import Dict, Any, List, Set
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from smartcash.ui.setup.dependency_installer.utils.ui_utils import update_status_panel, update_package_status

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
    
    # Log mulai instalasi dengan emoji dan highlight
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"🚀 Memulai instalasi {total_packages} packages...", "info")
    
    # Tampilkan progress tracker untuk operasi install
    if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
        ui_components['show_for_operation']('install')
    
    # Reset progress bar dengan pesan yang jelas
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, f"Menginstall {total_packages} packages...", True)
    
    # Inisialisasi progress untuk overall dan step
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('overall', 0, f"Memulai instalasi {total_packages} packages", "#17a2b8")
        ui_components['update_progress']('step', 0, "Mempersiapkan instalasi...", "#17a2b8")
    
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
                    
                    # Update progress untuk step dan overall
                    completed += 1
                    progress_percent = int((completed / total_packages) * 100)
                    
                    # Update status package di UI
                    # Cari package_key berdasarkan nama package
                    for category in ui_components.get('categories', []):
                        for pkg in category.get('packages', []):
                            if pkg.get('name', '').lower() == package.lower() or pkg.get('key', '') == package:
                                # Update status widget dengan icon dan warna yang sesuai
                                status_type = "success" if success else "error"
                                message = "Terinstall" if success else "Gagal install"
                                update_package_status(ui_components, pkg['key'], status_type, message)
                    
                    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                        # Update step progress dengan warna sesuai status
                        color = "#28a745" if success else "#dc3545"
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
                            "#17a2b8"
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
            update_status_panel("success", f"✅ Semua {total_packages} packages berhasil diinstall", ui_components)
        else:
            update_status_panel("warning", f"⚠️ {success_count}/{total_packages} packages berhasil diinstall", ui_components)
        
        # Complete operation jika semua berhasil, atau update progress selesai
        if success_count == total_packages and 'complete_operation' in ui_components and callable(ui_components['complete_operation']):
            ui_components['complete_operation'](f"Semua {total_packages} packages berhasil diinstall")
        elif 'update_progress' in ui_components and callable(ui_components['update_progress']):
            # Update step progress selesai
            ui_components['update_progress']('step', 100, "Instalasi selesai", "#28a745")
            # Update overall progress selesai
            ui_components['update_progress']('overall', 100, f"{success_count}/{total_packages} packages berhasil diinstall", "#28a745")
        
        return results
    
    except Exception as e:
        # Log error dengan format yang jelas
        error_message = f"Gagal menginstall packages: {str(e)}"
        
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](f"❌ {error_message}", "error")
        
        # Update status panel dengan pesan error menggunakan fungsi yang telah diperbarui
        update_status_panel("error", f"❌ {error_message}", ui_components)
        
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
        ui_components['log_message']("🔍 Memeriksa package yang perlu diinstall...", "info")
    
    # Jika hasil analisis tidak ada atau telah direset, jalankan analisis terlebih dahulu
    if 'analysis_result' not in ui_components or not ui_components['analysis_result']:
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message']("🔄 Menjalankan analisis ulang untuk memastikan status terbaru...", "info")
        
        # Update status panel
        update_status_panel("info", "🔄 Menjalankan analisis ulang...", ui_components)
        
        # Jalankan analisis package
        try:
            analyze_installed_packages(ui_components)
        except Exception as e:
            error_message = f"❌ Gagal menjalankan analisis: {str(e)}"
            if 'log_message' in ui_components and callable(ui_components['log_message']):
                ui_components['log_message'](error_message, "error")
            
            # Update status panel dengan pesan error
            update_status_panel("error", error_message, ui_components)
            
            # Tandai error pada progress tracker
            if 'error_operation' in ui_components and callable(ui_components['error_operation']):
                ui_components['error_operation'](error_message)
            
            return
    
    # Cek apakah ada hasil analisis setelah menjalankan analisis ulang
    if 'analysis_result' not in ui_components or not ui_components['analysis_result']:
        error_message = "❌ Tidak ada hasil analisis. Silakan jalankan analisis terlebih dahulu."
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](error_message, "error")
        
        # Update status panel dengan pesan error
        update_status_panel("error", error_message, ui_components)
        
        # Tandai error pada progress tracker
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](error_message)
        
        return
    
    # Dapatkan package yang perlu diinstall
    analysis_result = ui_components['analysis_result']
    packages_to_install = analysis_result.get('missing_packages', [])
    
    # Jika tidak ada package yang perlu diinstall
    if not packages_to_install:
        success_message = "✅ Semua package sudah terinstall dengan benar"
        # Tidak ada package yang perlu diinstall
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](success_message, "success")
        
        # Update status panel dengan pesan sukses menggunakan fungsi yang telah diperbarui
        update_status_panel("success", success_message, ui_components)
        
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
    
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](f"📦 Menemukan {package_count} package yang perlu diinstall", "info")
    
    # Tampilkan progress tracker untuk operasi install
    if 'show_for_operation' in ui_components and callable(ui_components['show_for_operation']):
        ui_components['show_for_operation']('install')
    
    # Reset progress bar dengan pesan yang jelas
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, f"Bersiap menginstall {package_count} packages...", True)
    
    # Install package secara parallel dengan progress tracking
    install_packages_parallel(packages_to_install, ui_components)
