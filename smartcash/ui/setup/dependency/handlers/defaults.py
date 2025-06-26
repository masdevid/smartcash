"""
Default Configuration for SmartCash Dependency Management.

This module provides the default package configurations and settings used
throughout the dependency management system. It serves as the source of truth
for package definitions and default configurations.

Note: All configuration handling logic is delegated to config_handler.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

# Configuration versions
CONFIG_VERSION = '1.0.0'
CONFIG_SCHEMA_VERSION = '1.0.0'

# Package manager constants
DEFAULT_PYTHON_PATH = 'python'
DEFAULT_VENV_PATH = '.venv'
DEFAULT_PACKAGE_MANAGER = 'pip'
DEFAULT_UPGRADE_STRATEGY = 'eager'
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_RETRIES = 3
DEFAULT_HTTP_RETRIES = 3
DEFAULT_MAX_WORKERS = 4  # Will be adjusted based on CPU count
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_LOG_FILE = 'dependency_installer.log'
DEFAULT_TRUSTED_HOSTS = ['pypi.org', 'files.pythonhosted.org']

class PackageConfig(TypedDict, total=False):
    """Configuration schema for an individual package."""
    key: str
    name: str
    description: str
    pip_name: str
    default: bool
    required: bool
    installed: bool
    min_version: Optional[str]
    max_version: Optional[str]

class PackageCategory(TypedDict, total=False):
    """Schema for a category of packages."""
    name: str
    icon: str
    description: str
    packages: List[PackageConfig]

# Core Requirements packages
CORE_PACKAGES: List[PackageConfig] = [
    {
        'key': 'ipywidgets',
        'name': 'IPython Widgets',
        'description': 'Core utilities dan helpers untuk UI',
        'pip_name': 'ipywidgets>=8.1.0',
        'default': True,
        'required': True
    },
    {
        'key': 'notebook_deps',
        'name': 'Notebook Dependencies',
        'description': 'IPython dan Jupyter dependencies',
        'pip_name': 'ipython>=8.12.0',
        'default': True,
        'required': True
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

# ML/AI Libraries packages
ML_AI_PACKAGES: List[PackageConfig] = [
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

# Data Processing packages
DATA_PROCESSING_PACKAGES: List[PackageConfig] = [
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

# Package categories with their respective packages
PACKAGE_CATEGORIES: List[PackageCategory] = [
    {
        'name': 'Core Requirements',
        'icon': '🔧',
        'description': 'Package inti SmartCash',
        'packages': CORE_PACKAGES
    },
    {
        'name': 'ML/AI Libraries',
        'icon': '🤖',
        'description': 'Machine Learning frameworks',
        'packages': ML_AI_PACKAGES
    },
    {
        'name': 'Data Processing',
        'icon': '📊',
        'description': 'Data manipulation tools',
        'packages': DATA_PROCESSING_PACKAGES
    }
]

def get_default_dependencies() -> Dict[str, Dict[str, Any]]:
    """Generate default dependencies configuration from PACKAGE_CATEGORIES."""
    dependencies = {}
    for category in PACKAGE_CATEGORIES:
        for pkg in category.get('packages', []):
            key = pkg.get('key', pkg.get('name', '').lower())
            if key:  # Only add if we have a valid key
                dependencies[key] = {
                    'required': pkg.get('required', False),
                    'version': pkg.get('pip_name', '').split('>=')[-1] if '>=' in pkg.get('pip_name', '') else 'latest',
                    'default': pkg.get('default', False)
                }
    return dependencies

# Default configuration that aligns with dependency_config.yaml
DEFAULT_CONFIG: Dict[str, Any] = {
    'module_name': 'dependency',
    'version': CONFIG_VERSION,
    'schema_version': CONFIG_SCHEMA_VERSION,
    'created_by': 'SmartCash',
    'description': 'Dependency installer configuration untuk SmartCash project',
    
    # Required fields for validation
    'dependencies': get_default_dependencies(),
    'install_options': {
        'use_venv': True,
        'venv_path': DEFAULT_VENV_PATH,
        'python_path': DEFAULT_PYTHON_PATH,
        'package_manager': DEFAULT_PACKAGE_MANAGER,
        'upgrade_strategy': DEFAULT_UPGRADE_STRATEGY,
        'timeout': DEFAULT_TIMEOUT,
        'retries': 2,
        'http_retries': DEFAULT_HTTP_RETRIES,
        'prefer_binary': False,
        'trusted_hosts': DEFAULT_TRUSTED_HOSTS,
        'extra_index_urls': [],
        'constraints': [],
        'parallel_workers': 3,
        'force_reinstall': False,
        'use_cache': True,
        'max_workers': DEFAULT_MAX_WORKERS
    },
    
    # Package selections
    'selected_packages': [pkg['key'] for cat in PACKAGE_CATEGORIES for pkg in cat['packages'] if pkg.get('default', False)],
    'custom_packages': '',
    'auto_analyze': True,
    
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
        'log_level': DEFAULT_LOG_LEVEL,
        'log_file': DEFAULT_LOG_FILE,
        'compact_view': False
    },
    
    # Advanced settings
    'advanced': {
        'pip_extra_args': [],
        'environment_variables': {},
        'pre_install_commands': [],
        'post_install_commands': []
    },
    
    # Package categories for UI
    'categories': PACKAGE_CATEGORIES
}
