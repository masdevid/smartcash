"""
File: smartcash/ui/setup/dependency/handlers/defaults.py
Deskripsi: Default configuration untuk dependency installer dengan struktur yang sama dengan YAML
"""

from typing import Dict, Any, List
import copy

# Package categories and their configurations
PACKAGE_CATEGORIES = [
    {
        'name': 'Core Requirements',
        'icon': '🔧',
        'description': 'Package inti SmartCash',
        'packages': [
            {
                'key': 'ipywidgets',
                'name': 'IPython Widgets',
                'description': 'Core utilities dan helpers untuk UI',
                'pip_name': 'ipywidgets>=8.1.0',
                'default': True
            },
            {
                'key': 'notebook_deps',
                'name': 'Notebook Dependencies',
                'description': 'IPython dan Jupyter dependencies',
                'pip_name': 'ipython>=8.12.0',
                'default': True
            },
            {
                'key': 'albumentations',
                'name': 'Albumentations',
                'description': 'Augmentation library',
                'pip_name': 'albumentations>=1.4.0',
                'default': True
            },
            {
                'key': 'yaml_parser',
                'name': 'YAML Parser',
                'description': 'Configuration file parsing',
                'pip_name': 'pyyaml>=6.0.0',
                'default': True
            }
        ]
    },
    {
        'name': 'ML/AI Libraries',
        'icon': '🤖',
        'description': 'Machine Learning frameworks',
        'packages': [
            {
                'key': 'pytorch',
                'name': 'PyTorch',
                'description': 'Deep learning framework utama',
                'pip_name': 'torch>=2.2.0',
                'default': True
            },
            {
                'key': 'torchvision',
                'name': 'TorchVision',
                'description': 'Computer vision untuk PyTorch',
                'pip_name': 'torchvision>=0.17.0',
                'default': True
            },
            {
                'key': 'ultralytics',
                'name': 'Ultralytics',
                'description': 'YOLO implementation terbaru',
                'pip_name': 'ultralytics>=8.1.0',
                'default': True
            },
            {
                'key': 'timm',
                'name': 'Timm',
                'description': 'Library untuk model vision transformer dan CNN',
                'pip_name': 'timm>=0.9.12',
                'default': True
            },
            {
                'key': 'scikit_learn',
                'name': 'scikit-learn',
                'description': 'Machine learning library untuk klasifikasi dan evaluasi',
                'pip_name': 'scikit-learn>=1.5.0',
                'default': True
            }
        ]
    },
    {
        'name': 'Data Processing',
        'icon': '📊',
        'description': 'Data manipulation tools',
        'packages': [
            {
                'key': 'pandas',
                'name': 'Pandas',
                'description': 'Data manipulation dan analysis',
                'pip_name': 'pandas>=2.1.0',
                'default': True
            },
            {
                'key': 'numpy',
                'name': 'NumPy',
                'description': 'Numerical computing foundation',
                'pip_name': 'numpy>=1.24.0,<2.0.0',
                'default': True
            },
            {
                'key': 'opencv',
                'name': 'OpenCV',
                'description': 'Computer vision library',
                'pip_name': 'opencv-python>=4.8.0',
                'default': True
            },
            {
                'key': 'pillow',
                'name': 'Pillow',
                'description': 'Python Imaging Library',
                'pip_name': 'Pillow>=10.0.0',
                'default': True
            },
            {
                'key': 'matplotlib',
                'name': 'Matplotlib',
                'description': 'Plotting dan visualization',
                'pip_name': 'matplotlib>=3.8.0',
                'default': True
            },
            {
                'key': 'scipy',
                'name': 'SciPy',
                'description': 'Scientific computing library',
                'pip_name': 'scipy>=1.12.0',
                'default': True
            }
        ]
    }
]

def get_default_selected_packages() -> List[str]:
    """Get list of package keys that are selected by default"""
    return [
        pkg['key']
        for category in PACKAGE_CATEGORIES
        for pkg in category['packages']
        if pkg.get('default', False)
    ]

# Default configuration yang selaras dengan dependency_config.yaml
DEFAULT_CONFIG = {
    'module_name': 'dependency',
    'version': '1.0.0',
    'created_by': 'SmartCash',
    'description': 'Dependency installer configuration untuk SmartCash project',
    
    # Package selections
    'selected_packages': get_default_selected_packages(),
    'custom_packages': '',
    'auto_analyze': True,
    
    # Installation settings
    'installation': {
        'parallel_workers': 3,
        'force_reinstall': False,
        'use_cache': True,
        'timeout': 300,
        'max_retries': 2,
        'retry_delay': 1.0
    },
    
    # Analysis settings
    'analysis': {
        'check_compatibility': True,
        'include_dev_deps': False,
        'batch_size': 10,
        'detailed_info': True
    },
    
    # UI settings
    'ui_settings': {
        'auto_analyze_on_render': True,
        'show_progress': True,
        'log_level': 'info',
        'compact_view': False
    },
    
    # Advanced settings
    'advanced': {
        'pip_extra_args': [],
        'environment_variables': {},
        'pre_install_commands': [],
        'post_install_commands': []
    }
}

def get_default_dependency_config() -> Dict[str, Any]:
    """Get complete default config untuk dependency installer"""
    return copy.deepcopy(DEFAULT_CONFIG)

def get_minimal_config() -> Dict[str, Any]:
    """Get minimal config untuk basic functionality"""
    config = get_default_dependency_config()
    return {
        'module_name': config['module_name'],
        'selected_packages': config['selected_packages'],
        'installation': config['installation'],
        'analysis': config['analysis']
    }

def get_environment_specific_config() -> Dict[str, Any]:
    """Get config yang disesuaikan dengan environment"""
    config = get_default_dependency_config()
    
    # Sesuaikan dengan environment Google Colab
    config['installation']['parallel_workers'] = 2  # Lebih konservatif di Colab
    config['installation']['timeout'] = 600  # Timeout lebih lama
    config['ui_settings']['compact_view'] = True  # Compact view di Colab
    
    return config

def get_package_categories() -> List[Dict[str, Any]]:
    """Get package categories with their configurations"""
    return copy.deepcopy(PACKAGE_CATEGORIES)