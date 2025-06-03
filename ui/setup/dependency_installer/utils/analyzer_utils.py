"""
File: smartcash/ui/setup/dependency_installer/utils/analyzer_utils.py
Deskripsi: Utilitas untuk menganalisis package yang terinstall
"""

from typing import Dict, Any, List, Tuple, Set
import pkg_resources
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.setup.dependency_installer.utils.package_utils import get_project_requirements, get_installed_packages

def get_installed_package_versions() -> Dict[str, str]:
    """Mendapatkan daftar package yang terinstall dengan versinya
    
    Returns:
        Dictionary nama package dan versinya
    """
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def parse_requirement(req_str: str) -> Tuple[str, str]:
    """Parse requirement string menjadi nama package dan versi
    
    Args:
        req_str: String requirement (contoh: 'numpy>=1.18.0')
        
    Returns:
        Tuple (nama package, versi)
    """
    # Hapus komentar dan whitespace
    req_str = req_str.strip()
    if '#' in req_str:
        req_str = req_str.split('#')[0].strip()
    
    # Jika kosong, return tuple kosong
    if not req_str:
        return "", ""
    
    try:
        # Coba parse dengan pkg_resources
        req = pkg_resources.Requirement.parse(req_str)
        return req.name.lower(), str(req.specifier) if req.specifier else ""
    except Exception:
        # Fallback ke regex jika pkg_resources gagal
        # Parse requirement string dengan regex
        match = re.match(r'^([a-zA-Z0-9_\-\.]+)([<>=!~]+.+)?$', req_str)
        
        if match:
            package_name = match.group(1).lower()
            version_spec = match.group(2) if match.group(2) else ""
            return package_name, version_spec
        
        # Jika tidak ada spesifikasi versi, gunakan nama package saja
        return req_str.lower(), ""

def check_version_compatibility(installed_version: str, required_version: str) -> bool:
    """Cek kompatibilitas versi package
    
    Args:
        installed_version: Versi yang terinstall
        required_version: Spesifikasi versi yang diperlukan
        
    Returns:
        True jika kompatibel, False jika tidak
    """
    if not required_version:
        return True
    
    try:
        # Gunakan pkg_resources untuk cek kompatibilitas
        return pkg_resources.parse_version(installed_version) in pkg_resources.Requirement.parse(f"dummy{required_version}").specifier
    except Exception:
        # Fallback ke perbandingan string sederhana jika parsing gagal
        return installed_version == required_version.lstrip("=")

def analyze_requirements(requirements: List[str], installed_packages: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Analisis status requirements berdasarkan package yang terinstall
    
    Args:
        requirements: List requirement string
        installed_packages: Dictionary package terinstall
        
    Returns:
        Dictionary status requirement
    """
    result = {}
    
    for req in requirements:
        package_name, version_spec = parse_requirement(req)
        
        if package_name in installed_packages:
            installed_version = installed_packages[package_name]
            is_compatible = check_version_compatibility(installed_version, version_spec)
            
            result[req] = {
                'installed': True,
                'package_name': package_name,
                'installed_version': installed_version,
                'required_version': version_spec,
                'compatible': is_compatible
            }
        else:
            result[req] = {
                'installed': False,
                'package_name': package_name,
                'installed_version': None,
                'required_version': version_spec,
                'compatible': False
            }
    
    return result

def analyze_installed_packages(ui_components: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Analisis package yang sudah terinstall dengan pendekatan modular dan DRY
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil analisis
    """
    # Log analisis dimulai
    if 'log_message' in ui_components and callable(ui_components['log_message']) and not ui_components.get('suppress_logs', False):
        ui_components['log_message']("🔍 Menganalisis packages terinstall...", "info")
    
    # Update progress
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('step', 10, "Mendapatkan daftar package terinstall...", "#17a2b8")
    
    try:
        # Dapatkan daftar package terinstall menggunakan fungsi dari package_utils.py
        installed_packages_set = get_installed_packages()
        installed_versions = get_installed_package_versions()
        
        # Update progress
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('step', 30, "Mendapatkan requirements...", "#17a2b8")
        
        # Dapatkan requirements menggunakan fungsi dari package_utils.py
        requirements = get_project_requirements('smartcash')
        
        # Update progress
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('step', 50, "Menganalisis requirements...", "#17a2b8")
        
        # Analisis requirements
        analysis_result = analyze_requirements(requirements, installed_versions)
        
        # Update progress
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('step', 70, "Menyimpan hasil analisis...", "#17a2b8")
        
        # Simpan hasil analisis ke ui_components
        ui_components['analysis_result'] = analysis_result
        
        # Hitung statistik
        total_packages = len(analysis_result)
        installed_packages_count = sum(1 for pkg in analysis_result.values() if pkg['installed'])
        compatible_packages = sum(1 for pkg in analysis_result.values() if pkg['compatible'])
        
        # Kategorikan package untuk UI
        missing_packages = [info['package_name'] for req, info in analysis_result.items() if not info['installed']]
        upgrade_packages = [info['package_name'] for req, info in analysis_result.items() if info['installed'] and not info['compatible']]
        ok_packages = [info['package_name'] for req, info in analysis_result.items() if info['installed'] and info['compatible']]
        
        # Simpan hasil kategorisasi ke ui_components untuk digunakan oleh installer
        ui_components['analysis_categories'] = {
            'missing': missing_packages,
            'upgrade': upgrade_packages,
            'ok': ok_packages
        }
        
        # Update progress
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('step', 90, "Menampilkan hasil analisis...", "#17a2b8")
        
        # Log hasil analisis
        if 'log_message' in ui_components and callable(ui_components['log_message']) and not ui_components.get('suppress_logs', False):
            ui_components['log_message'](
                f"📊 Hasil analisis: {installed_packages_count}/{total_packages} terinstall, {compatible_packages}/{total_packages} kompatibel",
                "info"
            )
        
        # Update status panel
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            if missing_packages or upgrade_packages:
                status_type = "warning"
                message = f"Ada {len(missing_packages)} package yang belum terinstall dan {len(upgrade_packages)} package yang perlu diupgrade"
            else:
                status_type = "success"
                message = f"Semua {len(ok_packages)} package sudah terinstall dengan benar"
            
            ui_components['update_status_panel'](status_type, message)
        
        # Update progress selesai
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('step', 100, "Analisis selesai", "#28a745")
        
        return analysis_result
    
    except Exception as e:
        # Log error
        error_message = f"Gagal menganalisis packages: {str(e)}"
        
        if 'log_message' in ui_components and callable(ui_components['log_message']) and not ui_components.get('suppress_logs', False):
            ui_components['log_message'](f"❌ {error_message}", "error")
        
        # Update status panel
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            ui_components['update_status_panel']("danger", error_message)
        
        # Tandai error pada progress
        if 'error_operation' in ui_components and callable(ui_components['error_operation']):
            ui_components['error_operation'](error_message)
        
        return {}
