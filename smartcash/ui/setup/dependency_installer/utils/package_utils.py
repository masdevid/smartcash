"""
File: smartcash/ui/setup/dependency_installer/utils/package_utils.py
Deskripsi: Utilitas untuk mengelola requirements package dan dependencies
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import importlib
import pkg_resources
import sys
import logging
from smartcash.ui.utils.ui_logger import log_to_ui

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

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
            except Exception as e:
                # Log error ke UI jika ada
                if 'ui_components' in locals():
                    log_to_ui(ui_components, f"Error membaca requirements.txt: {str(e)}", "error", "❌")
    
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
    Analisis package yang sudah terinstall dan update UI dengan penanganan error yang lebih baik.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        # Get package groups dan categories
        package_groups = get_package_groups()
        package_categories = get_package_categories()
        
        # Get installed packages dengan penanganan error yang lebih baik
        installed_packages = get_installed_packages()
        
        # Inisialisasi counter untuk statistik
        total_packages = 0
        total_installed = 0
        total_missing = 0
        all_missing_packages = []
        
        # Analisis setiap kategori dan package
        for category in package_categories:
            for package_info in category['packages']:
                package_key = package_info['key']
                package_name = package_info['name']
                package_description = package_info['description']
                
                # Get packages untuk key ini dengan penanganan error
                try:
                    packages_to_check = package_groups.get(package_key, [])
                    if callable(packages_to_check):
                        packages_to_check = packages_to_check()
                    
                    # Tambahkan ke total packages
                    total_packages += len(packages_to_check)
                    
                    # Cek apakah semua package sudah terinstall dengan penanganan error yang lebih baik
                    missing_packages = check_missing_packages(packages_to_check, installed_packages)
                    total_missing += len(missing_packages)
                    total_installed += len(packages_to_check) - len(missing_packages)
                    all_missing_packages.extend(missing_packages)
                    
                    # Update UI status
                    if package_key in ui_components:
                        # Update tooltip dengan informasi package
                        tooltip = f"{package_description}\n\nPackages: {', '.join(packages_to_check)}"
                        if missing_packages:
                            tooltip += f"\n\nMissing: {', '.join(missing_packages)}"
                        
                        # Update status icon dan warna
                        if not missing_packages:
                            # Semua package terinstall
                            status_icon = "✅"
                            status_color = "#28a745"  # green
                            status_text = f"{package_name} (Terinstall)"
                        else:
                            # Ada package yang belum terinstall
                            status_icon = "❌"
                            status_color = "#dc3545"  # red
                            status_text = f"{package_name} ({len(missing_packages)}/{len(packages_to_check)} perlu diinstall)"
                        
                        # Update UI components dengan penanganan error
                        try:
                            if hasattr(ui_components[package_key], 'description'):
                                ui_components[package_key].description = status_text
                            
                            # Update tooltip jika ada
                            tooltip_key = f"{package_key}_tooltip"
                            if tooltip_key in ui_components:
                                ui_components[tooltip_key].value = tooltip
                            
                            # Update status icon jika ada
                            status_key = f"{package_key}_status"
                            if status_key in ui_components:
                                ui_components[status_key].value = status_icon
                            
                            # Update warna jika ada
                            color_key = f"{package_key}_color"
                            if color_key in ui_components:
                                ui_components[color_key].value = status_color
                        except Exception as ui_error:
                            # Log error update UI tapi jangan gagalkan analisis
                            if 'log_message' in ui_components and callable(ui_components['log_message']):
                                ui_components['log_message'](f"⚠️ Error update UI untuk {package_key}: {str(ui_error)}", "warning")
                except Exception as pkg_error:
                    # Log error analisis package tapi jangan gagalkan seluruh analisis
                    if 'log_message' in ui_components and callable(ui_components['log_message']):
                        ui_components['log_message'](f"⚠️ Error analisis package {package_key}: {str(pkg_error)}", "warning")
        
        # Log ringkasan ke UI dengan penanganan error
        try:
            if total_missing > 0:
                # Gunakan log_message jika tersedia untuk memastikan log hanya muncul di UI
                if 'log_message' in ui_components and callable(ui_components['log_message']):
                    ui_components['log_message'](f"📊 Analisis selesai: {total_installed}/{total_packages} paket terinstal, {total_missing} paket perlu diinstal", "info")
                else:
                    log_to_ui(ui_components, f"📊 Analisis selesai: {total_installed}/{total_packages} paket terinstal, {total_missing} paket perlu diinstal", "info", "📊")
                
                # Update progress jika tersedia
                if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                    ui_components['update_progress']('overall', 0, f"📊 Analisis selesai: {total_missing} paket perlu diinstal")
            else:
                # Gunakan log_message jika tersedia untuk memastikan log hanya muncul di UI
                if 'log_message' in ui_components and callable(ui_components['log_message']):
                    ui_components['log_message'](f"✅ Semua paket ({total_packages}) sudah terinstal dengan baik", "success")
                else:
                    log_to_ui(ui_components, f"✅ Semua paket ({total_packages}) sudah terinstal dengan baik", "success", "✅")
                
                # Update progress jika tersedia
                if 'update_progress' in ui_components and callable(ui_components['update_progress']):
                    ui_components['update_progress']('overall', 100, "✅ Semua paket sudah terinstal")
        except Exception as log_error:
            # Fallback ke logging biasa jika gagal log ke UI
            logging.error(f"Error logging ringkasan: {str(log_error)}")
        
        # Return untuk testing dan debugging
        return {
            'total_packages': total_packages,
            'total_installed': total_installed,
            'total_missing': total_missing,
            'missing_packages': all_missing_packages
        }
    except Exception as e:
        # Tangkap semua error dan log ke UI
        error_message = f"❌ Gagal mendeteksi package: {str(e)}"
        
        # Gunakan log_message jika tersedia untuk memastikan log hanya muncul di UI
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](error_message, "error")
        else:
            # Fallback ke log_to_ui
            log_to_ui(ui_components, error_message, "error", "❌")
        
        # Update progress jika tersedia
        if 'update_progress' in ui_components and callable(ui_components['update_progress']):
            ui_components['update_progress']('overall', 0, error_message)
        
        # Return untuk testing dan debugging
        return {
            'total_packages': 0,
            'total_installed': 0,
            'total_missing': 0,
            'missing_packages': [],
            'error': str(e)
        }


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
    Dapatkan daftar package yang sudah terinstall dengan metode yang lebih robust.
    
    Returns:
        Set nama package yang terinstall
    """
    installed_packages = set()
    
    # Metode 1: Gunakan pkg_resources (paling akurat)
    try:
        installed_packages.update([pkg.key.lower() for pkg in pkg_resources.working_set])
    except Exception:
        pass
    
    # Metode 2: Gunakan importlib untuk cek package umum
    common_packages = {
        'numpy', 'pandas', 'matplotlib', 'torch', 'torchvision', 'torchaudio', 'tensorflow', 'sklearn', 
        'scikit-learn', 'scipy', 'cv2', 'opencv-python', 'opencv_python', 'pillow', 'pil', 'requests', 
        'bs4', 'beautifulsoup4', 'seaborn', 'plotly', 'ipywidgets', 'tqdm', 'albumentations',
        'roboflow', 'ultralytics', 'termcolor', 'pyyaml', 'yaml'
    }
    
    # Tambahkan alias package
    package_aliases = {
        'opencv-python': ['cv2', 'opencv_python'],
        'pillow': ['pil', 'PIL'],
        'scikit-learn': ['sklearn'],
        'beautifulsoup4': ['bs4'],
        'pyyaml': ['yaml']
    }
    
    # Cek package dengan importlib
    for pkg in common_packages:
        try:
            # Coba import package
            importlib.import_module(pkg)
            installed_packages.add(pkg.lower())
            
            # Tambahkan alias jika ada
            for main_pkg, aliases in package_aliases.items():
                if pkg.lower() == main_pkg.lower() or pkg.lower() in [alias.lower() for alias in aliases]:
                    installed_packages.add(main_pkg.lower())
                    for alias in aliases:
                        installed_packages.add(alias.lower())
        except ImportError:
            # Gagal import, coba metode lain
            pass
    
    # Metode 3: Cek sys.modules untuk package yang sudah diimport
    for pkg in sys.modules:
        pkg_base = pkg.split('.')[0].lower()
        installed_packages.add(pkg_base)
    
    return installed_packages


def check_missing_packages(required_packages: List[str], installed_packages: Set[str]) -> List[str]:
    """
    Cek package yang belum terinstall dengan metode yang lebih robust.
    
    Args:
        required_packages: List package yang dibutuhkan
        installed_packages: Set package yang sudah terinstall (lowercase)
        
    Returns:
        List package yang belum terinstall
    """
    missing_packages = []
    
    # Definisi mapping package untuk menangani nama modul yang berbeda dari nama package
    package_module_map = {
        'opencv-python': ['cv2', 'opencv_python', 'opencv-python'],
        'pillow': ['pil', 'PIL', 'pillow'],
        'beautifulsoup4': ['bs4', 'beautifulsoup4'],
        'scikit-learn': ['sklearn', 'scikit-learn', 'scikit_learn'],
        'pyyaml': ['yaml', 'pyyaml'],
        'matplotlib': ['matplotlib', 'matplotlib.pyplot', 'mpl'],
        'torchvision': ['torchvision', 'torch'],
        'torchaudio': ['torchaudio', 'torch'],
        'ipywidgets': ['ipywidgets', 'ipython', 'IPython']
    }
    
    for pkg_req in required_packages:
        try:
            # Parse package name (tanpa versi)
            pkg_name = pkg_req.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
            pkg_name_lower = pkg_name.lower()
            
            # Cek apakah package atau aliasnya sudah terinstall
            is_installed = False
            
            # Cek langsung di installed_packages
            if pkg_name_lower in installed_packages:
                is_installed = True
            else:
                # Cek dengan package_module_map
                for main_pkg, aliases in package_module_map.items():
                    if pkg_name_lower == main_pkg.lower():
                        # Jika package utama ada di map, cek semua aliasnya
                        for alias in aliases:
                            if alias.lower() in installed_packages:
                                is_installed = True
                                break
                    elif pkg_name_lower in [alias.lower() for alias in aliases]:
                        # Jika package adalah alias, cek package utama dan alias lainnya
                        if main_pkg.lower() in installed_packages:
                            is_installed = True
                            break
                        for alias in aliases:
                            if alias.lower() in installed_packages:
                                is_installed = True
                                break
            
            # Tambahkan ke daftar missing jika belum terinstall
            if not is_installed:
                missing_packages.append(pkg_req)
        except Exception as e:
            # Jika ada error saat parsing, tambahkan ke missing packages untuk aman
            logging.debug(f"Error saat cek package {pkg_req}: {str(e)}")
            missing_packages.append(pkg_req)
    
    return missing_packages
