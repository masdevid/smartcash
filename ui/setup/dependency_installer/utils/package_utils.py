"""
File: smartcash/ui/setup/dependency_installer/utils/package_utils.py
Deskripsi: Utilitas untuk mengelola requirements package dan dependencies
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import importlib
import pkg_resources
import sys
from smartcash.ui.utils.ui_logger import log_to_ui

def get_project_requirements(project_name: str) -> List[str]:
    """
    Dapatkan requirements untuk project tertentu.
    
    Args:
        project_name: Nama project ('smartcash', 'yolov5')
        
    Returns:
        List requirements
    """
    # Default requirements fallback
    default_requirements = {
        'smartcash': [
            "pyyaml>=6.0", 
            "termcolor>=2.0.0", 
            "roboflow>=0.2.29", 
            "ultralytics>=8.0.0", 
            "seaborn>=0.11.2", 
            "pillow>=8.0.0"
        ],
        'yolov5': [
            "matplotlib>=3.3", 
            "numpy>=1.18.5", 
            "opencv-python>=4.1.2", 
            "pillow>=8.0.0", 
            "pyyaml>=5.3.1", 
            "requests>=2.23.0", 
            "scipy>=1.4.1",
            "torch>=1.7.0", 
            "torchvision>=0.8.1"
        ]
    }
    
    # Paket yang akan dilewati (skip) saat instalasi
    skip_packages = ['tqdm']
    
    # Coba baca dari file requirements.txt
    potential_paths = [
        Path(f'{project_name}/requirements.txt'),
        Path.cwd() / f'{project_name}/requirements.txt',
        Path.cwd() / 'requirements.txt'
    ]
    
    for path in potential_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    requirements = []
                    for line in f:
                        line = line.strip()
                        # Skip comments dan baris kosong
                        if line and not line.startswith('#'):
                            # Ambil hanya package name dan versi
                            package_line = line.split('#')[0].strip()
                            
                            # Skip tqdm dan paket lain yang perlu dilewati
                            should_skip = False
                            for skip_pkg in skip_packages:
                                if package_line.lower().startswith(skip_pkg.lower()):
                                    should_skip = True
                                    break
                            
                            if not should_skip:
                                requirements.append(package_line)
                    if requirements:
                        return requirements
            except Exception:
                pass
    
    # Return default requirements jika tidak ada file
    return default_requirements.get(project_name, [])

def get_package_groups() -> Dict[str, Any]:
    """
    Dapatkan definisi grup package dengan dependencies.
    
    Returns:
        Dictionary grup package dengan dependencies
    """
    return {
        'yolov5_req': lambda: get_project_requirements('yolov5'),
        'torch_req': ['torch>=1.7.0', 'torchvision>=0.8.1', 'torchaudio>=0.7.0'],
        'albumentations_req': ['opencv-python>=4.1.2', 'albumentations>=1.0.0'],
        'notebook_req': ['ipywidgets>=7.0.0', 'tqdm>=4.0.0'],
        'smartcash_req': lambda: get_project_requirements('smartcash'),
        'matplotlib_req': ['matplotlib>=3.0.0', 'pandas>=1.0.0', 'seaborn>=0.11.0']
    }

def get_package_categories() -> List[Dict[str, Any]]:
    """
    Dapatkan kategori package untuk UI.
    
    Returns:
        List kategori package dengan metadata
    """
    return [
        {
            'name': "Core Packages",
            'key': 'core',
            'description': "Package inti untuk SmartCash",
            'icon': "📊",
            'packages': [
                {'name': "YOLOv5 Requirements", 'key': 'yolov5_req', 'default': True, 
                 'description': "Dependencies YOLOv5 (numpy, opencv, torch, etc)"},
                {'name': "SmartCash Utils", 'key': 'smartcash_req', 'default': True, 
                 'description': "Utility packages (pyyaml, termcolor, etc)"}
            ]
        },
        {
            'name': "AI & ML Packages",
            'key': 'ml',
            'description': "Package ML untuk model",
            'icon': "🧠",
            'packages': [
                {'name': "PyTorch", 'key': 'torch_req', 'default': True, 
                 'description': "Deep learning framework"},
                {'name': "Albumentations", 'key': 'albumentations_req', 'default': True, 
                 'description': "Augmentasi gambar untuk training"}
            ]
        },
        {
            'name': "Visualization Packages",
            'key': 'viz',
            'description': "Package untuk visualisasi",
            'icon': "📈",
            'packages': [
                {'name': "Matplotlib & Pandas", 'key': 'matplotlib_req', 'default': True, 
                 'description': "Visualisasi data dan plot"},
                {'name': "Jupyter Tools", 'key': 'notebook_req', 'default': True, 
                 'description': "Widget dan tools Jupyter/Colab"}
            ]
        }
    ]

def analyze_installed_packages(ui_components: Dict[str, Any]) -> None:
    """
    Analisis package yang sudah terinstall dan update UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    from smartcash.ui.utils.constants import COLORS
    
    # Log info ke UI
    log_to_ui(ui_components, "Menganalisis paket yang terinstal...", "info", "🔍")
    
    # Dapatkan package groups
    package_groups = get_package_groups()
    
    # Dapatkan package categories untuk UI
    package_categories = get_package_categories()
    
    # Cek setiap kategori dan package
    for category in package_categories:
        for package_info in category['packages']:
            package_key = package_info['key']
            
            # Dapatkan daftar package untuk key ini
            packages_to_check = package_groups.get(package_key, [])
            if callable(packages_to_check):
                packages_to_check = packages_to_check()
            
            # Status widget untuk package ini
            status_widget = ui_components.get(f"{package_key}_status")
            if not status_widget:
                continue
            
            # Cek apakah semua package terinstall
            all_installed = True
            missing_packages = []
            
            for package_req in packages_to_check:
                # Parse package name dan versi
                package_name = package_req.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
                
                # Cek apakah package terinstall
                try:
                    importlib.import_module(package_name)
                except ImportError:
                    try:
                        # Coba cek dengan pkg_resources
                        pkg_resources.get_distribution(package_name)
                    except pkg_resources.DistributionNotFound:
                        all_installed = False
                        missing_packages.append(package_name)
            
            # Update status widget
            if all_installed:
                status_widget.value = f"<div style='width:100px;color:{COLORS['success']}'>✅ Terinstall</div>"
            else:
                status_widget.value = f"<div style='width:100px;color:{COLORS['warning']}'>⚠️ Belum lengkap</div>"
                # Log info ke UI tentang paket yang belum terinstal
                if missing_packages:
                    log_to_ui(ui_components, f"Paket yang belum terinstal untuk {package_info['name']}: {', '.join(missing_packages)}", "warning", "⚠️")

    # Log ringkasan analisis
    total_packages = 0
    total_installed = 0
    total_missing = 0
    
    for category in package_categories:
        for package_info in category['packages']:
            package_key = package_info['key']
            packages_to_check = package_groups.get(package_key, [])
            if callable(packages_to_check):
                packages_to_check = packages_to_check()
            
            total_packages += len(packages_to_check)
            
            # Hitung paket yang terinstal dan yang belum
            for package_req in packages_to_check:
                package_name = package_req.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
                try:
                    importlib.import_module(package_name)
                    total_installed += 1
                except ImportError:
                    try:
                        pkg_resources.get_distribution(package_name)
                        total_installed += 1
                    except pkg_resources.DistributionNotFound:
                        total_missing += 1
    
    # Log ringkasan ke UI
    if total_missing > 0:
        log_to_ui(ui_components, f"Analisis selesai: {total_installed}/{total_packages} paket terinstal, {total_missing} paket perlu diinstal", "info", "📊")
    else:
        log_to_ui(ui_components, f"Semua paket ({total_packages}) sudah terinstal dengan baik", "success", "✅")


def parse_custom_packages(custom_packages_text: str) -> List[str]:
    """
    Parse teks custom packages menjadi list.
    
    Args:
        custom_packages_text: Teks custom packages
        
    Returns:
        List package requirements
    """
    if not custom_packages_text:
        return []
    
    # Split berdasarkan baris baru dan filter baris kosong
    packages = []
    for line in custom_packages_text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            packages.append(line)
    
    return packages


def get_installed_packages() -> Set[str]:
    """
    Dapatkan daftar package yang sudah terinstall.
    
    Returns:
        Set nama package yang terinstall
    """
    installed_packages = set()
    
    # Metode 1: Gunakan pkg_resources
    try:
        installed_packages.update([pkg.key for pkg in pkg_resources.working_set])
    except Exception:
        pass
    
    # Metode 2: Cek sys.modules untuk package yang sudah diimport
    common_packages = {
        'numpy', 'pandas', 'matplotlib', 'torch', 'tensorflow', 'sklearn', 
        'scipy', 'cv2', 'opencv_python', 'pillow', 'pil', 'requests', 
        'bs4', 'beautifulsoup4', 'seaborn', 'plotly', 'ipywidgets', 'tqdm'
    }
    
    for pkg in common_packages:
        if pkg in sys.modules:
            installed_packages.add(pkg)
    
    return installed_packages


def check_missing_packages(required_packages: List[str], installed_packages: Set[str]) -> List[str]:
    """
    Cek package yang belum terinstall.
    
    Args:
        required_packages: List package yang dibutuhkan
        installed_packages: Set package yang sudah terinstall
        
    Returns:
        List package yang belum terinstall
    """
    missing_packages = []
    
    for pkg_req in required_packages:
        # Parse package name (tanpa versi)
        pkg_name = pkg_req.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
        pkg_name = pkg_name.lower()
        
        # Cek apakah package sudah terinstall
        if pkg_name not in installed_packages:
            # Cek alias umum
            if pkg_name == 'opencv-python' and ('cv2' in installed_packages or 'opencv_python' in installed_packages):
                continue
            if pkg_name == 'pillow' and 'pil' in installed_packages:
                continue
            if pkg_name == 'beautifulsoup4' and 'bs4' in installed_packages:
                continue
            
            # Tambahkan ke daftar missing jika belum terinstall
            missing_packages.append(pkg_req)
    
    return missing_packages
