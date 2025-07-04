"""
File: smartcash/ui/setup/dependency/configs/dependency_defaults.py
Deskripsi: Default configuration untuk dependency management
"""

from typing import Dict, Any, List

def get_default_dependency_config() -> Dict[str, Any]:
    """Get default dependency configuration"""
    return {
        'module_name': 'dependency',
        'version': '2.0.0',
        'created_by': 'SmartCash',
        'description': 'Dependency management dengan tab-based UI dan persistent config',
        
        # Selected packages - bisa ditambah/dikurangi
        'selected_packages': [],
        
        # Custom packages string
        'custom_packages': '',
        
        # Install options
        'install_options': {
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
            'trusted_hosts': [
                'pypi.org',
                'files.pythonhosted.org'
            ]
        },
        
        # UI settings
        'ui_settings': {
            'auto_check_updates': True,
            'show_progress': True,
            'log_level': 'info',
            'compact_view': False,
            'default_tab': 0  # 0: package categories, 1: custom packages
        }
    }

def get_default_package_categories() -> Dict[str, Dict[str, Any]]:
    """Get default package categories untuk SmartCash"""
    return {
        'computer_vision': {
            'name': 'Computer Vision',
            'description': 'Packages untuk computer vision dan image processing',
            'icon': '👁️',
            'color': '#4CAF50',
            'packages': [
                {
                    'name': 'opencv-python',
                    'version': '>=4.8.0',
                    'description': 'OpenCV untuk computer vision',
                    'size': '~45MB',
                    'is_default': True
                },
                {
                    'name': 'pillow',
                    'version': '>=9.0.0',
                    'description': 'Python Imaging Library',
                    'size': '~3MB',
                    'is_default': True
                },
                {
                    'name': 'scikit-image',
                    'version': '>=0.20.0',
                    'description': 'Image processing library',
                    'size': '~12MB',
                    'is_default': False
                }
            ]
        },
        
        'machine_learning': {
            'name': 'Machine Learning',
            'description': 'Packages untuk machine learning dan deep learning',
            'icon': '🤖',
            'color': '#2196F3',
            'packages': [
                {
                    'name': 'torch',
                    'version': '>=2.0.0',
                    'description': 'PyTorch framework',
                    'size': '~800MB',
                    'is_default': True
                },
                {
                    'name': 'torchvision',
                    'version': '>=0.15.0',
                    'description': 'PyTorch vision library',
                    'size': '~25MB',
                    'is_default': True
                },
                {
                    'name': 'ultralytics',
                    'version': '>=8.0.0',
                    'description': 'YOLOv8 implementation',
                    'size': '~15MB',
                    'is_default': True
                },
                {
                    'name': 'efficientnet-pytorch',
                    'version': '>=0.7.0',
                    'description': 'EfficientNet implementation',
                    'size': '~5MB',
                    'is_default': True
                }
            ]
        },
        
        'data_science': {
            'name': 'Data Science',
            'description': 'Packages untuk data analysis dan visualization',
            'icon': '📊',
            'color': '#FF9800',
            'packages': [
                {
                    'name': 'numpy',
                    'version': '>=1.24.0',
                    'description': 'Numerical computing library',
                    'size': '~15MB',
                    'is_default': True
                },
                {
                    'name': 'pandas',
                    'version': '>=2.0.0',
                    'description': 'Data manipulation library',
                    'size': '~35MB',
                    'is_default': True
                },
                {
                    'name': 'matplotlib',
                    'version': '>=3.7.0',
                    'description': 'Plotting library',
                    'size': '~40MB',
                    'is_default': True
                },
                {
                    'name': 'seaborn',
                    'version': '>=0.12.0',
                    'description': 'Statistical visualization',
                    'size': '~5MB',
                    'is_default': False
                }
            ]
        },
        
        'utilities': {
            'name': 'Utilities',
            'description': 'Packages utilitas dan helper functions',
            'icon': '🔧',
            'color': '#9C27B0',
            'packages': [
                {
                    'name': 'tqdm',
                    'version': '>=4.65.0',
                    'description': 'Progress bar library',
                    'size': '~200KB',
                    'is_default': True
                },
                {
                    'name': 'pyyaml',
                    'version': '>=6.0',
                    'description': 'YAML parser',
                    'size': '~600KB',
                    'is_default': True
                },
                {
                    'name': 'requests',
                    'version': '>=2.31.0',
                    'description': 'HTTP library',
                    'size': '~500KB',
                    'is_default': False
                }
            ]
        },
        
        'jupyter': {
            'name': 'Jupyter',
            'description': 'Packages untuk Jupyter notebook dan widgets',
            'icon': '📓',
            'color': '#F44336',
            'packages': [
                {
                    'name': 'ipywidgets',
                    'version': '>=8.0.0',
                    'description': 'Interactive widgets',
                    'size': '~3MB',
                    'is_default': True
                },
                {
                    'name': 'jupyter',
                    'version': '>=1.0.0',
                    'description': 'Jupyter notebook',
                    'size': '~50MB',
                    'is_default': False
                }
            ]
        }
    }

def get_package_status_options() -> Dict[str, Dict[str, Any]]:
    """Get package status options dengan styling"""
    return {
        'installed': {
            'icon': '✅',
            'color': '#4CAF50',
            'text': 'Installed',
            'bg_color': '#E8F5E8'
        },
        'checking': {
            'icon': '🔄',
            'color': '#FF9800',
            'text': 'Checking...',
            'bg_color': '#FFF3E0'
        },
        'installing': {
            'icon': '📥',
            'color': '#2196F3',
            'text': 'Installing...',
            'bg_color': '#E3F2FD'
        },
        'updating': {
            'icon': '⬆️',
            'color': '#9C27B0',
            'text': 'Updating...',
            'bg_color': '#F3E5F5'
        },
        'not_installed': {
            'icon': '❌',
            'color': '#F44336',
            'text': 'Not Installed',
            'bg_color': '#FFEBEE'
        },
        'update_available': {
            'icon': '🔄',
            'color': '#FF5722',
            'text': 'Update Available',
            'bg_color': '#FFF8E1'
        },
        'error': {
            'icon': '⚠️',
            'color': '#F44336',
            'text': 'Error',
            'bg_color': '#FFEBEE'
        }
    }

def get_button_actions() -> Dict[str, Dict[str, Any]]:
    """Get button actions dengan styling"""
    return {
        'install': {
            'text': 'Install',
            'icon': '📥',
            'color': '#4CAF50',
            'disabled_color': '#CCCCCC'
        },
        'update': {
            'text': 'Update',
            'icon': '⬆️',
            'color': '#FF9800',
            'disabled_color': '#CCCCCC'
        },
        'uninstall': {
            'text': 'Uninstall',
            'icon': '🗑️',
            'color': '#F44336',
            'disabled_color': '#CCCCCC'
        },
        'check': {
            'text': 'Check',
            'icon': '🔍',
            'color': '#2196F3',
            'disabled_color': '#CCCCCC'
        }
    }