"""
File: smartcash/ui/setup/dependency/handlers/status_check_handler_refactored.py
Deskripsi: Refactored status check handler untuk memeriksa status dependensi

Fitur Utama:
- Pengecekan status dependensi yang lebih efisien
- Pelacakan progress yang lebih baik
- Error handling yang lebih kuat
- Kode yang lebih modular dan mudah dipelihara
"""

from dataclasses import dataclass
import logging
import sys
from typing import Dict, Any, List, Optional, Tuple

# Core imports
from smartcash.common import get_logger
from smartcash.ui.setup.dependency.utils import (
    LogLevel, with_logging, requires, log_to_ui_safe, with_button_context,
    create_operation_context, update_status_panel, batch_check_packages_status,
    parse_package_requirement, get_installed_packages_dict
)
from smartcash.ui.setup.dependency.utils.system_info_utils import check_system_requirements

# Setup logger
logger = get_logger(__name__)

@dataclass
class StatusCheckResult:
    """Hasil pengecekan status dependensi"""
    package_status: Dict[str, Dict[str, Any]]
    system_requirements: Dict[str, Any]
    all_requirements_met: bool = False

class StatusChecker:
    """Kelas utama untuk menangani pengecekan status dependensi"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        self.ui = ui_components
        self.config = config
        self.logger = ui_components.get('logger', logger)
    
    @classmethod
    def setup(cls, ui_components: Dict[str, Any], config: Dict[str, Any]) -> 'StatusChecker':
        """Factory method untuk inisialisasi checker"""
        checker = cls(ui_components, config)
        checker._setup_handlers()
        return checker
    
    def _setup_handlers(self) -> None:
        """Setup semua handler yang diperlukan"""
        if 'check_status_button' in self.ui:
            self.ui['check_status_button'].on_click(
                lambda b: with_button_context(self.ui, 'check_status_button')(self.execute_status_check)
            )
    
    @with_logging("Status Check", LogLevel.INFO)
    def execute_status_check(self, *args) -> StatusCheckResult:
        """
        Eksekusi pengecekan status dependensi
        
        Returns:
            StatusCheckResult: Hasil pengecekan status
        """
        ctx = create_operation_context(self.ui, 'status_check')
        
        try:
            # Inisialisasi progress
            self._init_progress()
            
            # Dapatkan daftar paket yang akan diperiksa
            packages = self._get_packages_to_check()
            
            # Periksa status tiap paket
            package_status = self._check_packages_status(packages, ctx)
            
            # Periksa persyaratan sistem
            system_requirements = self._check_system_requirements()
            
            # Hasilkan ringkasan
            result = StatusCheckResult(
                package_status=package_status,
                system_requirements=system_requirements,
                all_requirements_met=system_requirements.get('all_requirements_met', False)
            )
            
            # Tampilkan ringkasan
            self._show_status_summary(result, ctx)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Gagal mengecek status: {str(e)}", exc_info=True)
            update_status_panel(self.ui, f"⚠️ Gagal mengecek status: {str(e)}", "error")
            raise
        finally:
            # Pastikan progress ditutup
            self._complete_progress()
    
    def _init_progress(self) -> None:
        """Inisialisasi progress tracker"""
        if progress_tracker := self.ui.get('progress_tracker'):
            progress_tracker.show("Status Check", ["🔍 Analisis", "📊 Evaluasi", "✅ Verifikasi"])
            progress_tracker.update(0, "Memulai pengecekan status...")
    
    def _complete_progress(self) -> None:
        """Tandai progress selesai"""
        if progress_tracker := self.ui.get('progress_tracker'):
            progress_tracker.complete("Pengecekan status selesai")
    
    def _get_packages_to_check(self) -> List[Dict[str, Any]]:
        """Dapatkan daftar paket yang akan diperiksa"""
        packages = self.config.get('packages', [])
        self.logger.info(f"Memeriksa status {len(packages)} paket")
        return packages
    
    def _check_packages_status(self, packages: List[Dict[str, Any]], ctx) -> Dict[str, Dict[str, Any]]:
        """Periksa status tiap paket"""
        package_status = {}
        total_packages = len(packages)
        
        for idx, pkg in enumerate(packages, 1):
            pkg_name = pkg.get('name', 'unknown')
            try:
                # Update progress
                progress = int((idx / total_packages) * 100)
                if progress_tracker := self.ui.get('progress_tracker'):
                    progress_tracker.update(progress, f"Memeriksa {pkg_name}...")
                
                # Periksa status paket
                pkg_status = self._check_single_package(pkg)
                package_status[pkg_name] = pkg_status
                
                # Update UI
                self._update_package_ui(pkg_name, pkg_status)
                
            except Exception as e:
                self.logger.error(f"Gagal memeriksa {pkg_name}: {str(e)}", exc_info=True)
                package_status[pkg_name] = {
                    'installed': False,
                    'version': None,
                    'error': str(e)
                }
        
        return package_status
    
    def _check_single_package(self, pkg: Dict[str, Any]) -> Dict[str, Any]:
        """Periksa status satu paket"""
        pkg_name = pkg.get('pip_name', pkg.get('name', ''))
        package_name = pkg_name.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
        
        try:
            # Dapatkan daftar paket yang terinstall
            installed_packages = get_installed_packages_dict()
            
            # Periksa apakah paket terinstall
            is_installed = package_name in installed_packages
            
            # Dapatkan informasi detail paket
            detailed_info = self._get_detailed_package_info(package_name, installed_packages)
            
            return {
                'name': pkg.get('name', package_name),
                'pip_name': pkg_name,
                'package_name': package_name,
                'category': pkg.get('category', 'other'),
                'installed': is_installed,
                'version': installed_packages.get(package_name) if is_installed else None,
                'required_version': pkg.get('version', ''),
                **detailed_info
            }
            
        except Exception as e:
            self.logger.error(f"Gagal memeriksa {pkg_name}: {str(e)}", exc_info=True)
            return {
                'name': pkg.get('name', package_name),
                'pip_name': pkg_name,
                'package_name': package_name,
                'category': pkg.get('category', 'other'),
                'installed': False,
                'version': None,
                'error': str(e)
            }
    
    def _get_detailed_package_info(self, package_name: str, installed_packages: Dict[str, str]) -> Dict[str, Any]:
        """Dapatkan informasi detail paket"""
        try:
            if package_name not in installed_packages:
                return {'installed': False}
                
            # Dapatkan versi yang terinstall
            version = installed_packages[package_name]
            
            # Dapatkan info detail menggunakan pip show
            detailed_info = {}
            try:
                import subprocess
                import json
                
                # Gunakan pip show untuk mendapatkan info detail
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'show', '--no-color', '--no-python-version-warning', package_name],
                    capture_output=True, text=True, check=False
                )
                
                if result.returncode == 0:
                    # Parse output pip show
                    for line in result.stdout.split('\n'):
                        if ': ' in line:
                            key, value = line.split(': ', 1)
                            detailed_info[key.lower().replace('-', '_')] = value.strip()
            except Exception as e:
                self.logger.warning(f"Tidak dapat mendapatkan info detail untuk {package_name}: {str(e)}")
            
            return {
                'installed': True,
                'version': version,
                'location': detailed_info.get('location'),
                'requires': detailed_info.get('requires', '').split(', ') if detailed_info.get('requires') else [],
                'summary': detailed_info.get('summary', '')
            }
            
        except Exception as e:
            self.logger.error(f"Gagal mendapatkan info detail paket {package_name}: {str(e)}", exc_info=True)
            return {'installed': False, 'error': str(e)}
    
    def _check_system_requirements(self) -> Dict[str, Any]:
        """Periksa persyaratan sistem"""
        try:
            requirements = check_system_requirements()
            # Pastikan format return sesuai dengan yang diharapkan
            return {
                'python_version_ok': requirements.get('python_version_ok', False),
                'platform_supported': requirements.get('platform_supported', False),
                'memory_sufficient': requirements.get('memory_sufficient', False),
                'all_requirements_met': requirements.get('all_requirements_met', False)
            }
        except Exception as e:
            self.logger.error(f"Gagal memeriksa persyaratan sistem: {str(e)}")
            return {
                'python_version_ok': False,
                'platform_supported': False,
                'memory_sufficient': False,
                'all_requirements_met': False
            }
    
    def _update_package_ui(self, pkg_name: str, pkg_status: Dict[str, Any]) -> None:
        """Perbarui UI berdasarkan status paket"""
        # Implementasi pembaruan UI disesuaikan dengan framework UI yang digunakan
        if pkg_status.get('installed'):
            status_text = f"✅ {pkg_name} (v{pkg_status.get('version', '?')})"
        else:
            status_text = f"❌ {pkg_name} (Tidak terinstal)"
        
        log_to_ui_safe(self.ui, status_text, "info")
    
    def _show_status_summary(self, result: StatusCheckResult, ctx) -> None:
        """Tampilkan ringkasan status"""
        # Hitung ringkasan per kategori
        category_summary = {}
        for pkg_info in result.package_status.values():
            category = pkg_info.get('category', 'other')
            if category not in category_summary:
                category_summary[category] = {'total': 0, 'installed': 0}
            category_summary[category]['total'] += 1
            if pkg_info.get('installed'):
                category_summary[category]['installed'] += 1
        
        # Tampilkan ringkasan
        for category, stats in category_summary.items():
            percentage = (stats['installed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            summary_msg = f"📋 {category}: {stats['installed']}/{stats['total']} ({percentage:.1f}%)"
            log_to_ui_safe(self.ui, summary_msg, "info")
        
        # Tampilkan peringatan jika ada persyaratan sistem yang tidak terpenuhi
        if not result.system_requirements.get('all_requirements_met', True):
            log_to_ui_safe(self.ui, "⚠️ Beberapa persyaratan sistem tidak terpenuhi", "warning")
            
            requirement_checks = [
                ('python_version_ok', 'Versi Python tidak didukung'),
                ('memory_sufficient', 'Memori tidak mencukupi'),
                ('platform_supported', 'Platform tidak didukung')
            ]
            
            for check, msg in requirement_checks:
                if not result.system_requirements.get(check, True):
                    log_to_ui_safe(self.ui, f"  • {msg}", "warning")

# Fungsi kompatibilitas
def setup_status_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Fungsi utama untuk setup status check handler (kompatibilitas ke belakang)
    
    Args:
        ui_components: Komponen UI
        config: Konfigurasi
    """
    return StatusChecker.setup(ui_components, config)
