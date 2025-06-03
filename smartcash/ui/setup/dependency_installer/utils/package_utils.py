"""
File: smartcash/ui/setup/dependency_installer/utils/package_utils.py
Deskripsi: Utilitas untuk mengelola requirements package dan dependencies dengan pendekatan DRY dan one-liner style
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable, Union, Tuple
import importlib
import pkg_resources
import sys
import logging
from smartcash.ui.utils.ui_logger import log_to_ui

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

def get_default_requirements(project_name: str) -> List[str]:
    """Mendapatkan default requirements untuk project tertentu
    
    Args:
        project_name: Nama project ('smartcash', 'yolov5')
        
    Returns:
        List default requirements
    """
    requirements_map = {
        'smartcash': ["pyyaml>=6.0", "termcolor>=2.0.0", "roboflow>=0.2.29", "ultralytics>=8.0.0", "seaborn>=0.11.2", "pillow>=8.0.0"],
        'yolov5': ["matplotlib>=3.3", "numpy>=1.18.5", "opencv-python>=4.1.2", "pillow>=8.0.0", "pyyaml>=5.3.1", "requests>=2.23.0", "scipy>=1.4.1", "torch>=1.7.0", "torchvision>=0.8.1"]
    }
    return requirements_map.get(project_name, [])

def get_project_requirements(project_name: str) -> List[str]:
    """Dapatkan requirements untuk project tertentu
    
    Args:
        project_name: Nama project ('smartcash', 'yolov5')
        
    Returns:
        List requirements
    """
    skip_packages = ['tqdm']  # Paket yang akan dilewati (skip) saat instalasi
    potential_paths = [Path(f'{project_name}/requirements.txt'), Path.cwd() / f'{project_name}/requirements.txt', Path.cwd() / 'requirements.txt']
    
    for path in potential_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    requirements = [line.split('#')[0].strip() for line in f if line.strip() and not line.strip().startswith('#')]
                    requirements = [pkg for pkg in requirements if not any(pkg.lower().startswith(skip_pkg.lower()) for skip_pkg in skip_packages)]
                    if requirements: return requirements
            except Exception as e:
                if 'ui_components' in locals(): log_to_ui(ui_components, f"Error membaca requirements.txt: {str(e)}", "error", "❌")
    
    return get_default_requirements(project_name)

def get_package_groups() -> Dict[str, Any]:
    """Dapatkan dictionary grup package dengan dependencies
    
    Returns:
        Dictionary grup package dengan dependencies
    """
    return {
        'yolov5_req': lambda: get_project_requirements('yolov5'),
        'torch_req': ["torch>=1.7.0", "torchvision>=0.8.1", "torchaudio>=0.7.0"],
        'smartcash_req': lambda: get_project_requirements('smartcash'),
        'efficientnet_req': ["efficientnet-pytorch>=0.7.0"],
        'roboflow_req': ["roboflow>=0.2.29"],
        'ui_req': ["ipywidgets>=7.6.0", "ipython>=7.0.0"],
        'utils_req': ["pyyaml>=6.0", "termcolor>=2.0.0", "tqdm>=4.64.0"]
    }

def create_package_info(name: str, key: str, description: str, icon: str) -> Dict[str, str]:
    """Helper untuk membuat info package dengan format yang konsisten
    
    Args:
        name: Nama package
        key: Key package
        description: Deskripsi package
        icon: Ikon package
        
    Returns:
        Dictionary info package
    """
    return {'name': name, 'key': key, 'description': description, 'icon': icon}

def create_category(name: str, key: str, description: str, icon: str, packages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Helper untuk membuat kategori dengan format yang konsisten
    
    Args:
        name: Nama kategori
        key: Key kategori
        description: Deskripsi kategori
        icon: Ikon kategori
        packages: List package dalam kategori
        
    Returns:
        Dictionary kategori
    """
    return {'name': name, 'key': key, 'description': description, 'icon': icon, 'packages': packages}

def get_package_categories() -> List[Dict[str, Any]]:
    """Dapatkan list kategori package untuk UI
    
    Returns:
        List kategori package
    """
    # Definisikan package info menggunakan helper function
    core_packages = [
        create_package_info("YOLOv5 Requirements", 'yolov5_req', "Package yang diperlukan untuk menjalankan YOLOv5", "📦"),
        create_package_info("SmartCash Utils", 'smartcash_req', "Package utilitas untuk SmartCash", "🧰")
    ]
    
    dl_packages = [
        create_package_info("PyTorch", 'torch_req', "PyTorch dan komponen terkait", "🔥"),
        create_package_info("EfficientNet", 'efficientnet_req', "EfficientNet untuk backbone", "🏋️")
    ]
    
    tools_packages = [
        create_package_info("Roboflow", 'roboflow_req', "Roboflow untuk dataset management", "🤖"),
        create_package_info("UI Components", 'ui_req', "Package untuk UI components", "🖥️"),
        create_package_info("Utilities", 'utils_req', "Package utilitas umum", "🧰")
    ]
    
    # Buat kategori menggunakan helper function
    return [
        create_category("Core Packages", 'core', "Package inti yang diperlukan untuk menjalankan SmartCash", "🔧", core_packages),
        create_category("Deep Learning", 'dl', "Package deep learning", "🧠", dl_packages),
        create_category("Tools & Utilities", 'tools', "Package tools dan utilitas", "🛠️", tools_packages)
    ]

def safe_ui_call(ui_components: Dict[str, Any], function_key: str, *args, **kwargs) -> Any:
    """Fungsi helper untuk memanggil fungsi UI dengan penanganan error
    
    Args:
        ui_components: Dictionary komponen UI
        function_key: Kunci fungsi yang akan dipanggil
        *args: Argumen untuk fungsi
        **kwargs: Keyword argumen untuk fungsi
        
    Returns:
        Hasil fungsi atau None jika gagal
    """
    if function_key in ui_components and callable(ui_components[function_key]):
        try:
            return ui_components[function_key](*args, **kwargs)
        except Exception as e:
            logging.debug(f"Error memanggil {function_key}: {str(e)}")
    return None

def update_package_ui(package_ui: Any, num_missing: int, num_packages: int) -> None:
    """Update UI komponen package dengan status instalasi
    
    Args:
        package_ui: Komponen UI package
        num_missing: Jumlah package yang belum terinstall
        num_packages: Total jumlah package
    """
    if hasattr(package_ui, 'badge') and hasattr(package_ui.badge, 'description'):
        if num_missing == 0:
            package_ui.badge.description = "✅"
            package_ui.badge.button_style = "success"
            if hasattr(package_ui, 'tooltip'): package_ui.tooltip = f"Semua {num_packages} package sudah terinstall"
        else:
            package_ui.badge.description = f"⚠️ {num_missing}/{num_packages}"
            package_ui.badge.button_style = "warning"
            if hasattr(package_ui, 'tooltip'): package_ui.tooltip = f"{num_missing} dari {num_packages} package belum terinstall"

def parse_custom_packages(custom_packages_text: str) -> List[str]:
    """Parse teks custom packages menjadi list
    
    Args:
        custom_packages_text: Teks custom packages
        
    Returns:
        List package requirements
    """
    return [line.strip() for line in custom_packages_text.split('\n') if line.strip() and not line.strip().startswith('#')]

def get_installed_packages() -> Set[str]:
    """Dapatkan daftar package yang sudah terinstall dengan metode yang lebih robust
    
    Returns:
        Set nama package yang terinstall
    """
    installed_packages = set(pkg.key.lower() for pkg in pkg_resources.working_set)
    common_packages = {
        'numpy', 'pandas', 'matplotlib', 'torch', 'torchvision', 'torchaudio', 'tensorflow', 'sklearn', 
        'scikit-learn', 'scipy', 'cv2', 'opencv-python', 'opencv_python', 'pillow', 'pil', 'requests', 
        'bs4', 'beautifulsoup4', 'seaborn', 'plotly', 'ipywidgets', 'tqdm', 'albumentations',
        'roboflow', 'ultralytics', 'termcolor', 'pyyaml', 'yaml'
    }
    package_aliases = {
        'opencv-python': ['cv2', 'opencv_python'],
        'pillow': ['pil', 'PIL'],
        'scikit-learn': ['sklearn'],
        'beautifulsoup4': ['bs4'],
        'pyyaml': ['yaml']
    }
    installed_packages.update(pkg.lower() for pkg in common_packages)
    for main_pkg, aliases in package_aliases.items():
        if main_pkg.lower() in installed_packages:
            installed_packages.update(alias.lower() for alias in aliases)
    installed_packages.update(pkg.split('.')[0].lower() for pkg in sys.modules)
    return installed_packages

def check_missing_packages(required_packages: List[str], installed_packages: Set[str]) -> List[str]:
    """Cek package yang belum terinstall dengan metode yang lebih robust
    
    Args:
        required_packages: List package yang dibutuhkan
        installed_packages: Set package yang sudah terinstall (lowercase)
        
    Returns:
        List package yang belum terinstall
    """
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
    missing_packages = []
    for pkg_req in required_packages:
        pkg_name = pkg_req.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
        pkg_name_lower = pkg_name.lower()
        if pkg_name_lower not in installed_packages:
            for main_pkg, aliases in package_module_map.items():
                if pkg_name_lower == main_pkg.lower() or pkg_name_lower in [alias.lower() for alias in aliases]:
                    if any(alias.lower() in installed_packages for alias in aliases) or main_pkg.lower() in installed_packages:
                        break
            else:
                missing_packages.append(pkg_req)
    return missing_packages

def analyze_installed_packages(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Analisis package yang sudah terinstall dengan pendekatan DRY dan one-liner style
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary hasil analisis
    """
    try:
        # Dapatkan data yang diperlukan
        package_groups = get_package_groups()
        package_categories = get_package_categories()
        installed_packages = get_installed_packages()
        
        # Setup UI untuk analisis
        safe_ui_call(ui_components, 'log_message', "🔍 Menganalisis package yang terinstall...", "info")
        safe_ui_call(ui_components, 'reset_progress_bar', 0, "Menganalisis package...", True)
        if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'): 
            ui_components['progress_container'].layout.visibility = 'visible'
        
        # Inisialisasi counter
        total_packages, total_installed, total_missing = 0, 0, 0
        all_missing_packages = []
        
        # Analisis setiap kategori dan package
        for category in package_categories:
            for package_info in category['packages']:
                package_key = package_info['key']
                
                try:
                    # Dapatkan dan cek package
                    packages_to_check = package_groups.get(package_key, [])
                    if callable(packages_to_check): packages_to_check = packages_to_check()
                    missing_packages = check_missing_packages(packages_to_check, installed_packages)
                    
                    # Update counter
                    num_packages, num_missing = len(packages_to_check), len(missing_packages)
                    num_installed = num_packages - num_missing
                    total_packages += num_packages
                    total_installed += num_installed
                    total_missing += num_missing
                    if missing_packages: all_missing_packages.extend(missing_packages)
                    
                    # Update UI package jika tersedia
                    if package_key in ui_components: update_package_ui(ui_components[package_key], num_missing, num_packages)
                except Exception as pkg_error:
                    safe_ui_call(ui_components, 'log_message', f"❌ Error saat menganalisis {package_info['name']}: {str(pkg_error)}", "error")
        
        # Log ringkasan hasil analisis
        try:
            if total_missing > 0:
                summary_message = f"📊 Analisis selesai: {total_installed}/{total_packages} paket terinstal, {total_missing} paket perlu diinstal"
                if not safe_ui_call(ui_components, 'log_message', summary_message, "info"):
                    log_to_ui(ui_components, summary_message, "info", "📊")
                safe_ui_call(ui_components, 'update_progress', 'overall', 0, f"📊 Analisis selesai: {total_missing} paket perlu diinstal")
            else:
                success_message = f"✅ Semua paket ({total_packages}) sudah terinstal dengan baik"
                if not safe_ui_call(ui_components, 'log_message', success_message, "success"):
                    log_to_ui(ui_components, success_message, "success", "✅")
                safe_ui_call(ui_components, 'update_progress', 'overall', 100, "✅ Semua paket sudah terinstal")
        except Exception as log_error:
            logging.error(f"Error logging ringkasan: {str(log_error)}")
        
        # Return hasil analisis
        return {
            'total_packages': total_packages,
            'total_installed': total_installed,
            'total_missing': total_missing,
            'missing_packages': all_missing_packages
        }
    except Exception as e:
        # Handle error
        error_message = f"❌ Gagal mendeteksi package: {str(e)}"
        if not safe_ui_call(ui_components, 'log_message', error_message, "error"):
            log_to_ui(ui_components, error_message, "error", "❌")
        safe_ui_call(ui_components, 'update_progress', 'overall', 0, error_message)
        
        return {'total_packages': 0, 'total_installed': 0, 'total_missing': 0, 'missing_packages': [], 'error': str(e)}

def parse_custom_packages(custom_packages_text: str) -> List[str]:
    """Parse teks custom packages menjadi list
    
    Args:
        custom_packages_text: Teks custom packages
        
    Returns:
        List package requirements
    """
    if not custom_packages_text: return []
    return [line.strip() for line in custom_packages_text.split('\n') if line.strip() and not line.strip().startswith('#')]


def get_installed_packages() -> Set[str]:
    """Dapatkan daftar package yang sudah terinstall dengan metode yang lebih robust
    
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
    
    # Cek package dengan importlib dan tambahkan alias
    for pkg in common_packages:
        try:
            importlib.import_module(pkg)
            installed_packages.add(pkg.lower())
            # Tambahkan alias jika ada
            for main_pkg, aliases in package_aliases.items():
                if pkg.lower() == main_pkg.lower() or pkg.lower() in [alias.lower() for alias in aliases]:
                    installed_packages.add(main_pkg.lower())
                    installed_packages.update([alias.lower() for alias in aliases])
        except ImportError:
            pass
    
    # Metode 3: Cek sys.modules untuk package yang sudah diimport
    installed_packages.update([pkg.split('.')[0].lower() for pkg in sys.modules])
    
    return installed_packages


def check_missing_packages(required_packages: List[str], installed_packages: Set[str]) -> List[str]:
    """Cek package yang belum terinstall dengan metode yang lebih robust
    
    Args:
        required_packages: List package yang dibutuhkan
        installed_packages: Set package yang sudah terinstall (lowercase)
        
    Returns:
        List package yang belum terinstall
    """
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
    
    missing_packages = []
    
    for pkg_req in required_packages:
        try:
            # Parse package name (tanpa versi)
            pkg_name = pkg_req.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
            pkg_name_lower = pkg_name.lower()
            
            # Cek apakah package atau aliasnya sudah terinstall
            is_installed = pkg_name_lower in installed_packages
            
            # Jika belum terinstall, cek dengan package_module_map
            if not is_installed:
                for main_pkg, aliases in package_module_map.items():
                    # Cek jika package utama atau aliasnya terinstall
                    if (pkg_name_lower == main_pkg.lower() and any(alias.lower() in installed_packages for alias in aliases)) or \
                       (pkg_name_lower in [alias.lower() for alias in aliases] and \
                        (main_pkg.lower() in installed_packages or any(alias.lower() in installed_packages for alias in aliases))):
                        is_installed = True
                        break
            
            # Tambahkan ke daftar missing jika belum terinstall
            if not is_installed: 
                missing_packages.append(pkg_req)
        except Exception as e:
            logging.debug(f"Error saat cek package {pkg_req}: {str(e)}")
            missing_packages.append(pkg_req)
    
    return missing_packages
