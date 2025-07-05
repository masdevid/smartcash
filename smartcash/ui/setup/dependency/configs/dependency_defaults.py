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
        'core_requirements': {
            'name': 'Core Requirements',
            'description': 'Package inti SmartCash',
            'icon': '🔧',
            'color': '#4CAF50',
            'packages': [
                {
                    'name': 'ipywidgets',
                    'version': '>=8.1.0',
                    'description': 'Core utilities dan helpers untuk UI',
                    'pip_name': 'ipywidgets>=8.1.0',
                    'is_default': True,
                    'size': '~5MB'
                },
                {
                    'name': 'notebook_deps',
                    'version': '>=8.12.0',
                    'description': 'IPython dan Jupyter dependencies',
                    'pip_name': 'ipython>=8.12.0',
                    'is_default': True,
                    'size': '~10MB'
                },
                {
                    'name': 'albumentations',
                    'version': '>=1.4.0',
                    'description': 'Augmentation library',
                    'pip_name': 'albumentations>=1.4.0',
                    'is_default': True,
                    'size': '~2MB'
                },
                {
                    'name': 'yaml_parser',
                    'version': '>=6.0.0',
                    'description': 'Configuration file parsing',
                    'pip_name': 'pyyaml>=6.0.0',
                    'is_default': True,
                    'size': '~1MB'
                }
            ]
        },
        
        'ml_ai_libraries': {
            'name': 'ML/AI Libraries',
            'description': 'Machine Learning frameworks',
            'icon': '🤖',
            'color': '#2196F3',
            'packages': [
                {
                    'name': 'pytorch',
                    'version': '>=2.2.0',
                    'description': 'Deep learning framework utama',
                    'pip_name': 'torch>=2.2.0',
                    'is_default': True,
                    'size': '~800MB'
                },
                {
                    'name': 'torchvision',
                    'version': '>=0.17.0',
                    'description': 'Computer vision untuk PyTorch',
                    'pip_name': 'torchvision>=0.17.0',
                    'is_default': True,
                    'size': '~25MB'
                },
                {
                    'name': 'ultralytics',
                    'version': '>=8.1.0',
                    'description': 'YOLO implementation terbaru',
                    'pip_name': 'ultralytics>=8.1.0',
                    'is_default': True,
                    'size': '~15MB'
                },
                {
                    'name': 'timm',
                    'version': '>=0.9.12',
                    'description': 'Library untuk model vision transformer dan CNN',
                    'pip_name': 'timm>=0.9.12',
                    'is_default': True,
                    'size': '~5MB'
                },
                {
                    'name': 'scikit_learn',
                    'version': '>=1.5.0',
                    'description': 'Machine learning library untuk klasifikasi dan evaluasi',
                    'pip_name': 'scikit-learn>=1.5.0',
                    'is_default': True,
                    'size': '~10MB'
                }
            ]
        },
        
        'data_processing': {
            'name': 'Data Processing',
            'description': 'Data manipulation tools',
            'icon': '📊',
            'color': '#9C27B0',
            'packages': [
                {
                    'name': 'pandas',
                    'version': '>=2.1.0',
                    'description': 'Data manipulation dan analysis',
                    'pip_name': 'pandas>=2.1.0',
                    'is_default': True,
                    'size': '~35MB'
                },
                {
                    'name': 'numpy',
                    'version': '>=1.24.0,<2.0.0',
                    'description': 'Numerical computing foundation',
                    'pip_name': 'numpy>=1.24.0,<2.0.0',
                    'is_default': True,
                    'size': '~15MB'
                },
                {
                    'name': 'opencv',
                    'version': '>=4.8.0',
                    'description': 'Computer vision library',
                    'pip_name': 'opencv-python>=4.8.0',
                    'is_default': True,
                    'size': '~45MB'
                },
                {
                    'name': 'pillow',
                    'version': '>=10.0.0',
                    'description': 'Python Imaging Library',
                    'pip_name': 'Pillow>=10.0.0',
                    'is_default': True,
                    'size': '~3MB'
                },
                {
                    'name': 'matplotlib',
                    'version': '>=3.8.0',
                    'description': 'Plotting dan visualization',
                    'pip_name': 'matplotlib>=3.8.0',
                    'is_default': True,
                    'size': '~40MB'
                },
                {
                    'name': 'scipy',
                    'version': '>=1.12.0',
                    'description': 'Scientific computing library',
                    'pip_name': 'scipy>=1.12.0',
                    'is_default': True,
                    'size': '~30MB'
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