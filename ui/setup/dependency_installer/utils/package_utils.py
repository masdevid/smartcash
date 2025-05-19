"""
File: smartcash/ui/setup/dependency_installer/utils/package_utils.py
Deskripsi: Utilitas untuk mengelola requirements package dan dependencies
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib
import pkg_resources

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
            "torchvision>=0.8.1", 
            # tqdm dikomentari karena akan dilewati
            # "tqdm>=4.64.0"
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
        'torch_req': ['torch', 'torchvision', 'torchaudio'],
        'albumentations_req': ['albumentations>=1.0.0'],
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
