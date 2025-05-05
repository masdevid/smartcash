"""
File: smartcash/ui/setup/utils/package_requirements.py
Deskripsi: Utilitas untuk mengelola requirements package dan dependencies
"""

from pathlib import Path
from typing import Dict, List, Any

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