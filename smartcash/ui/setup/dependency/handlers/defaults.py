"""
File: smartcash/ui/setup/dependency/handlers/defaults.py
Deskripsi: Default configuration untuk dependency management system
"""

from typing import Dict, Any, List, TypedDict
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

# Package Configuration Schema
class PackageConfig(TypedDict, total=False):
    key: str
    name: str
    description: str
    pip_name: str
    default: bool
    required: bool
    installed: bool
    version: str
    latest_version: str
    update_available: bool

class CategoryConfig(TypedDict, total=False):
    name: str
    icon: str
    description: str
    packages: List[PackageConfig]

# Default Package Categories - TIDAK BISA DIHAPUS, HANYA BISA DIUBAH STATUS
DEFAULT_CATEGORIES: List[CategoryConfig] = [
    {
        'name': 'Core Requirements',
        'icon': '🔧',
        'description': 'Paket wajib untuk menjalankan SmartCash',
        'packages': [
            {
                'key': 'ipywidgets',
                'name': 'IPyWidgets',
                'description': 'Interactive widgets untuk Jupyter',
                'pip_name': 'ipywidgets>=8.1.0',
                'default': True,
                'required': True,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            },
            {
                'key': 'pyyaml',
                'name': 'PyYAML',
                'description': 'Parser untuk file konfigurasi YAML',
                'pip_name': 'pyyaml>=6.0.0',
                'default': True,
                'required': True,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            },
            {
                'key': 'requests',
                'name': 'Requests',
                'description': 'HTTP library untuk Python',
                'pip_name': 'requests>=2.31.0',
                'default': True,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            }
        ]
    },
    {
        'name': 'Deep Learning',
        'icon': '🧠',
        'description': 'Framework dan library untuk deep learning',
        'packages': [
            {
                'key': 'torch',
                'name': 'PyTorch',
                'description': 'Deep learning framework utama',
                'pip_name': 'torch>=2.2.0',
                'default': True,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            },
            {
                'key': 'torchvision',
                'name': 'TorchVision',
                'description': 'Computer vision library untuk PyTorch',
                'pip_name': 'torchvision>=0.17.0',
                'default': True,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            },
            {
                'key': 'ultralytics',
                'name': 'Ultralytics YOLO',
                'description': 'YOLOv8 implementation terbaru',
                'pip_name': 'ultralytics>=8.1.0',
                'default': True,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            }
        ]
    },
    {
        'name': 'Data Processing',
        'icon': '📊',
        'description': 'Library untuk preprocessing dan augmentasi data',
        'packages': [
            {
                'key': 'opencv',
                'name': 'OpenCV',
                'description': 'Computer vision dan image processing',
                'pip_name': 'opencv-python>=4.9.0',
                'default': True,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            },
            {
                'key': 'albumentations',
                'name': 'Albumentations',
                'description': 'Fast image augmentation library',
                'pip_name': 'albumentations>=1.4.0',
                'default': True,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            },
            {
                'key': 'pillow',
                'name': 'Pillow',
                'description': 'Python Imaging Library',
                'pip_name': 'pillow>=10.2.0',
                'default': True,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            }
        ]
    },
    {
        'name': 'Visualization',
        'icon': '📈',
        'description': 'Library untuk visualisasi dan plotting',
        'packages': [
            {
                'key': 'matplotlib',
                'name': 'Matplotlib',
                'description': 'Plotting library untuk Python',
                'pip_name': 'matplotlib>=3.8.0',
                'default': False,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            },
            {
                'key': 'seaborn',
                'name': 'Seaborn',
                'description': 'Statistical data visualization',
                'pip_name': 'seaborn>=0.13.0',
                'default': False,
                'required': False,
                'installed': False,
                'version': '',
                'latest_version': '',
                'update_available': False
            }
        ]
    }
]

# Installation Options
DEFAULT_INSTALL_OPTIONS: Dict[str, Any] = {
    'use_venv': True,
    'venv_path': '.venv',
    'python_path': 'python',
    'package_manager': 'pip',
    'upgrade_strategy': 'eager',
    'timeout': 300,
    'retries': 3,
    'parallel_workers': 4,
    'force_reinstall': False,
    'use_cache': True,
    'trusted_hosts': ['pypi.org', 'files.pythonhosted.org']
}

# UI Settings
DEFAULT_UI_SETTINGS: Dict[str, Any] = {
    'auto_check_updates': True,
    'show_progress': True,
    'log_level': 'info',
    'compact_view': False
}

# Main Default Configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    'module_name': 'dependency',
    'version': '1.0.0',
    'created_by': 'SmartCash',
    'categories': DEFAULT_CATEGORIES,
    'selected_packages': [],  # Akan diisi dengan package yang default=True
    'custom_packages': '',
    'install_options': DEFAULT_INSTALL_OPTIONS,
    'ui_settings': DEFAULT_UI_SETTINGS
}

def get_default_dependency_config() -> Dict[str, Any]:
    """Mengembalikan konfigurasi default untuk dependency management
    
    Returns:
        Dict[str, Any]: Konfigurasi default lengkap
    """
    config = DEFAULT_CONFIG.copy()
    
    # Auto-populate selected_packages dengan package yang default=True
    selected_packages = []
    for category in config['categories']:
        for package in category['packages']:
            if package.get('default', False):
                selected_packages.append(package['key'])
    
    config['selected_packages'] = selected_packages
    
    logger.info(f"📋 Loaded default config dengan {len(selected_packages)} package terpilih")
    return config

def get_all_packages() -> List[PackageConfig]:
    """Mengembalikan semua package dari semua kategori
    
    Returns:
        List[PackageConfig]: Daftar semua package
    """
    all_packages = []
    for category in DEFAULT_CATEGORIES:
        all_packages.extend(category['packages'])
    return all_packages

def get_package_by_key(package_key: str) -> PackageConfig:
    """Mencari package berdasarkan key
    
    Args:
        package_key: Key package yang dicari
        
    Returns:
        PackageConfig: Package yang ditemukan atau None
    """
    for package in get_all_packages():
        if package['key'] == package_key:
            return package
    return None

def is_default_package(package_key: str) -> bool:
    """Mengecek apakah package adalah default package (tidak bisa dihapus)
    
    Args:
        package_key: Key package yang dicek
        
    Returns:
        bool: True jika package adalah default package
    """
    package = get_package_by_key(package_key)
    return package.get('default', False) if package else False