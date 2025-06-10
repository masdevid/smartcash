"""
File: smartcash/ui/setup/dependency/handlers/dependency_handlers.py
Deskripsi: Handler utama untuk modul dependency installer dengan absolute imports
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import sys
import subprocess
import pkg_resources
import importlib
import time

from smartcash.ui.setup.dependency.utils.ui_utils import log_to_ui_safe, update_status_panel
from smartcash.ui.setup.dependency.utils.button_manager import disable_all_buttons, enable_all_buttons, set_button_processing_state
from smartcash.ui.setup.dependency.utils.progress_utils import create_operation_context, update_package_status_by_name, batch_update_package_status, ProgressSteps

class DependencyHandlers:
    """Kelas utama untuk mengelola semua handler dependency installer"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Inisialisasi handler dengan komponen UI dan konfigurasi"""
        self.ui_components = ui_components
        self.config = config
        self.packages = config.get('packages', {})
        self.is_running = False
        self.max_workers = config.get('max_workers', 4)
        
        # Setup handler untuk tombol
        self._setup_button_handlers()
    
    def _setup_button_handlers(self):
        """Setup handler untuk semua tombol"""
        buttons = self.ui_components.get('buttons', {})
        
        # Handler untuk tombol analisis
        analyze_button = buttons.get('analyze')
        if analyze_button:
            analyze_button.on_click(lambda b: self.handle_analyze())
        
        # Handler untuk tombol instalasi
        install_button = buttons.get('install')
        if install_button:
            install_button.on_click(lambda b: self.handle_install())
        
        # Handler untuk tombol cek status
        check_button = buttons.get('check')
        if check_button:
            check_button.on_click(lambda b: self.handle_check_status())
        
        # Handler untuk tombol reset
        reset_button = buttons.get('reset')
        if reset_button:
            reset_button.on_click(lambda b: self.handle_reset())
    
    def handle_analyze(self):
        """Handler untuk tombol analisis"""
        if self.is_running:
            return
        
        self.is_running = True
        try:
            with create_operation_context(self.ui_components, "analisis paket") as ctx:
                log_to_ui_safe(self.ui_components, "🔍 Memulai analisis paket...", "info")
                ctx.stepped_progress(ProgressSteps.ANALYZE_INIT, "Memulai analisis paket", 0)
                
                # Analisis status paket
                package_status = self._analyze_packages()
                
                # Update status paket di UI
                batch_update_package_status(self.ui_components, package_status)
                
                # Hitung statistik
                total = len(package_status)
                installed = sum(1 for status in package_status.values() if status == 'installed')
                not_installed = total - installed
                
                # Log hasil
                log_to_ui_safe(
                    self.ui_components, 
                    f"✅ Analisis selesai: {installed} paket terinstall, {not_installed} paket belum terinstall", 
                    "success"
                )
                ctx.stepped_progress(ProgressSteps.ANALYZE_COMPLETE, "Analisis selesai", 100)
        except Exception as e:
            log_to_ui_safe(self.ui_components, f"❌ Error saat analisis: {str(e)}", "error")
            update_status_panel(self.ui_components, f"❌ Error saat analisis: {str(e)}", "error")
        finally:
            self.is_running = False
    
    def handle_install(self):
        """Handler untuk tombol instalasi"""
        if self.is_running:
            return
        
        self.is_running = True
        try:
            with create_operation_context(self.ui_components, "instalasi paket") as ctx:
                log_to_ui_safe(self.ui_components, "🔄 Memulai instalasi paket...", "info")
                ctx.stepped_progress(ProgressSteps.INSTALL_INIT, "Memulai instalasi paket", 0)
                
                # Dapatkan paket yang belum terinstall
                package_status = self._analyze_packages()
                packages_to_install = [pkg for pkg, status in package_status.items() 
                                      if status == 'not_installed']
                
                if not packages_to_install:
                    log_to_ui_safe(self.ui_components, "✅ Semua paket sudah terinstall!", "success")
                    ctx.stepped_progress(ProgressSteps.INSTALL_COMPLETE, "Semua paket sudah terinstall", 100)
                    return
                
                # Log paket yang akan diinstall
                log_to_ui_safe(
                    self.ui_components, 
                    f"🔄 Akan menginstall {len(packages_to_install)} paket: {', '.join(packages_to_install)}", 
                    "info"
                )
                
                # Install paket secara paralel
                ctx.stepped_progress(ProgressSteps.INSTALL_START, "Memulai instalasi paket", 10)
                results = self._install_packages(packages_to_install, ctx)
                
                # Hitung statistik
                success_count = sum(1 for _, success in results if success)
                failed_count = len(results) - success_count
                
                # Log hasil
                if failed_count == 0:
                    log_to_ui_safe(
                        self.ui_components, 
                        f"✅ Instalasi selesai: {success_count} paket berhasil diinstall", 
                        "success"
                    )
                else:
                    log_to_ui_safe(
                        self.ui_components, 
                        f"⚠️ Instalasi selesai dengan warning: {success_count} paket berhasil, {failed_count} paket gagal", 
                        "warning"
                    )
                
                # Update status paket di UI
                self._update_package_status_after_install(results)
                
                ctx.stepped_progress(ProgressSteps.INSTALL_COMPLETE, "Instalasi selesai", 100)
        except Exception as e:
            log_to_ui_safe(self.ui_components, f"❌ Error saat instalasi: {str(e)}", "error")
            update_status_panel(self.ui_components, f"❌ Error saat instalasi: {str(e)}", "error")
        finally:
            self.is_running = False
    
    def handle_check_status(self):
        """Handler untuk tombol cek status"""
        if self.is_running:
            return
        
        self.is_running = True
        try:
            with create_operation_context(self.ui_components, "pengecekan status") as ctx:
                log_to_ui_safe(self.ui_components, "🔍 Memeriksa status paket...", "info")
                ctx.stepped_progress(ProgressSteps.CHECK_INIT, "Memulai pengecekan status", 0)
                
                # Cek status paket
                package_status = self._analyze_packages()
                
                # Update status paket di UI
                batch_update_package_status(self.ui_components, package_status)
                
                # Hitung statistik
                total = len(package_status)
                installed = sum(1 for status in package_status.values() if status == 'installed')
                not_installed = total - installed
                
                # Log hasil
                if not_installed == 0:
                    log_to_ui_safe(
                        self.ui_components, 
                        f"✅ Semua paket ({total}) sudah terinstall dengan baik", 
                        "success"
                    )
                else:
                    log_to_ui_safe(
                        self.ui_components, 
                        f"⚠️ Status paket: {installed} terinstall, {not_installed} belum terinstall", 
                        "warning"
                    )
                
                ctx.stepped_progress(ProgressSteps.CHECK_COMPLETE, "Pengecekan selesai", 100)
        except Exception as e:
            log_to_ui_safe(self.ui_components, f"❌ Error saat pengecekan: {str(e)}", "error")
            update_status_panel(self.ui_components, f"❌ Error saat pengecekan: {str(e)}", "error")
        finally:
            self.is_running = False
    
    def handle_reset(self):
        """Handler untuk tombol reset"""
        if self.is_running:
            return
        
        try:
            # Reset status panel
            update_status_panel(self.ui_components, "🔄 UI direset", "info")
            
            # Reset progress bar jika ada
            progress_callback = self.ui_components.get('progress_callback')
            if progress_callback:
                progress_callback(message="Siap", progress=0)
            
            # Reset status paket di UI
            package_status = {pkg: "" for pkg in self.packages.keys()}
            batch_update_package_status(self.ui_components, package_status)
            
            # Clear log jika ada
            log_output = self.ui_components.get('log_output')
            if log_output and hasattr(log_output, 'clear_output'):
                log_output.clear_output()
                with log_output:
                    print("Log direset")
            
            log_to_ui_safe(self.ui_components, "🔄 UI telah direset", "info")
        except Exception as e:
            log_to_ui_safe(self.ui_components, f"❌ Error saat reset: {str(e)}", "error")
    
    def _analyze_packages(self) -> Dict[str, str]:
        """Analisis status instalasi paket
        
        Returns:
            Dict[str, str]: Dictionary dengan nama paket sebagai key dan status sebagai value
                            Status: 'installed' atau 'not_installed'
        """
        result = {}
        
        for pkg_name, pkg_info in self.packages.items():
            # Update status di UI
            update_package_status_by_name(self.ui_components, pkg_name, 'processing')
            
            # Log
            log_to_ui_safe(self.ui_components, f"🔍 Memeriksa paket {pkg_name}...", "info")
            
            # Cek apakah paket terinstall
            is_installed = self._check_package_installed(pkg_name, pkg_info)
            status = 'installed' if is_installed else 'not_installed'
            
            # Update status di UI
            update_package_status_by_name(self.ui_components, pkg_name, status)
            
            # Simpan hasil
            result[pkg_name] = status
            
            # Delay kecil untuk UI responsiveness
            time.sleep(0.1)
        
        return result
    
    def _check_package_installed(self, pkg_name: str, pkg_info: Dict[str, Any]) -> bool:
        """Cek apakah paket terinstall
        
        Args:
            pkg_name: Nama paket
            pkg_info: Informasi paket dari konfigurasi
            
        Returns:
            bool: True jika paket terinstall, False jika tidak
        """
        try:
            # Cek dengan importlib
            if pkg_info.get('check_import', False):
                module_name = pkg_info.get('import_name', pkg_name)
                importlib.import_module(module_name)
                return True
            
            # Cek dengan pkg_resources
            else:
                pkg_resources.get_distribution(pkg_name)
                return True
        except (ImportError, pkg_resources.DistributionNotFound):
            return False
    
    def _install_packages(self, packages_to_install: List[str], ctx) -> List[Tuple[str, bool]]:
        """Install paket secara paralel
        
        Args:
            packages_to_install: List nama paket yang akan diinstall
            ctx: Context operation untuk update progress
            
        Returns:
            List[Tuple[str, bool]]: List tuple (nama_paket, sukses)
        """
        results = []
        total_packages = len(packages_to_install)
        
        # Gunakan ThreadPoolExecutor untuk instalasi paralel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit semua tugas instalasi
            future_to_pkg = {executor.submit(self._install_single_package, pkg): pkg 
                             for pkg in packages_to_install}
            
            # Proses hasil setiap tugas selesai
            completed = 0
            for future in future_to_pkg:
                pkg_name = future_to_pkg[future]
                try:
                    # Update status di UI
                    update_package_status_by_name(self.ui_components, pkg_name, 'processing')
                    
                    # Tunggu hasil instalasi
                    success = future.result()
                    
                    # Simpan hasil
                    results.append((pkg_name, success))
                    
                    # Update progress
                    completed += 1
                    progress = int(10 + (completed / total_packages) * 90)
                    ctx.stepped_progress(
                        ProgressSteps.INSTALL_START,
                        f"Menginstall paket ({completed}/{total_packages})",
                        progress
                    )
                    
                except Exception as e:
                    # Tangani error
                    log_to_ui_safe(self.ui_components, f"❌ Error saat menginstall {pkg_name}: {str(e)}", "error")
                    results.append((pkg_name, False))
                    
                    # Update progress
                    completed += 1
                    progress = int(10 + (completed / total_packages) * 90)
                    ctx.stepped_progress(
                        ProgressSteps.INSTALL_START,
                        f"Menginstall paket ({completed}/{total_packages})",
                        progress
                    )
        
        return results
    
    def _install_single_package(self, pkg_name: str) -> bool:
        """Install satu paket dengan pip
        
        Args:
            pkg_name: Nama paket yang akan diinstall
            
        Returns:
            bool: True jika instalasi berhasil, False jika gagal
        """
        try:
            # Log
            log_to_ui_safe(self.ui_components, f"🔄 Menginstall paket {pkg_name}...", "info")
            
            # Install dengan pip
            cmd = [sys.executable, "-m", "pip", "install", pkg_name]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Ambil output
            stdout, stderr = process.communicate()
            
            # Cek hasil
            if process.returncode == 0:
                log_to_ui_safe(self.ui_components, f"✅ Paket {pkg_name} berhasil diinstall", "success")
                return True
            else:
                log_to_ui_safe(self.ui_components, f"❌ Gagal menginstall {pkg_name}: {stderr}", "error")
                return False
                
        except Exception as e:
            log_to_ui_safe(self.ui_components, f"❌ Error saat menginstall {pkg_name}: {str(e)}", "error")
            return False
    
    def _update_package_status_after_install(self, results: List[Tuple[str, bool]]) -> None:
        """Update status paket di UI setelah instalasi
        
        Args:
            results: List tuple (nama_paket, sukses)
        """
        for pkg_name, success in results:
            status = 'installed' if success else 'error'
            update_package_status_by_name(self.ui_components, pkg_name, status)
