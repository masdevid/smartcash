"""
File: smartcash/ui/setup/dependency/configs/dependency_defaults.py
Deskripsi: Default configuration untuk dependency management
"""

from typing import Dict, Any, List
import os

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
        
        # Uninstalled default packages (to track which defaults were uninstalled)
        'uninstalled_defaults': [],
        
        # Package categories - include reference to package categories
        'package_categories': get_default_package_categories(),
        
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
    base_categories = {
        'core_requirements': {
            'name': 'Kebutuhan Inti',
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
                    'name': 'ipython',
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
                    'name': 'pyyaml',
                    'version': '>=6.0.0',
                    'description': 'Configuration file parsing',
                    'pip_name': 'pyyaml>=6.0.0',
                    'is_default': True,
                    'size': '~1MB'
                },
                {
                    'name': 'python-dotenv',
                    'version': '>=1.0.1',
                    'description': 'Environment variable loading',
                    'pip_name': 'python-dotenv>=1.0.1',
                    'is_default': True,
                    'size': '~0.5MB'
                },
                {
                    'name': 'pytz',
                    'version': '>=2023.3',
                    'description': 'Timezone support',
                    'pip_name': 'pytz>=2023.3',
                    'is_default': True,
                    'size': '~0.5MB'
                }
            ]
        },
        
        'ml_ai_libraries': {
            'name': 'Pustaka ML/AI',
            'description': 'Framework Machine Learning',
            'icon': '🤖',
            'color': '#2196F3',
            'packages': [
                {
                    'name': 'torch',
                    'version': '>=2.1.0',
                    'description': 'Deep learning framework utama',
                    'pip_name': 'torch>=2.1.0',
                    'is_default': True,
                    'size': '~800MB'
                },
                {
                    'name': 'torchvision',
                    'version': '>=0.16.0',
                    'description': 'Computer vision untuk PyTorch',
                    'pip_name': 'torchvision>=0.16.0',
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
                    'name': 'scikit-learn',
                    'version': '>=1.5.0',
                    'description': 'Machine learning library untuk klasifikasi dan evaluasi',
                    'pip_name': 'scikit-learn>=1.5.0',
                    'is_default': True,
                    'size': '~10MB'
                },
                {
                    'name': 'tensorboard',
                    'version': '>=2.19.0',
                    'description': 'TensorBoard visualization tool',
                    'pip_name': 'tensorboard>=2.19.0',
                    'is_default': True,
                    'size': '~10MB'
                },
                {
                    'name': 'thop',
                    'version': '>=0.1.1',
                    'description': 'PyTorch model complexity analysis',
                    'pip_name': 'thop>=0.1.1',
                    'is_default': True,
                    'size': '~1MB'
                }
            ]
        },
        
        'data_processing': {
            'name': 'Pemrosesan Data',
            'description': 'Alat manipulasi data',
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
                    'name': 'opencv-python',
                    'version': '>=4.8.0',
                    'description': 'Computer vision library',
                    'pip_name': 'opencv-python>=4.8.0',
                    'is_default': True,
                    'size': '~45MB'
                },
                {
                    'name': 'Pillow',
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
                },
                {
                    'name': 'pybboxes',
                    'version': '>=0.1.6',
                    'description': 'Bounding box manipulation library',
                    'pip_name': 'pybboxes>=0.1.6',
                    'is_default': True,
                    'size': '~1MB'
                },
                {
                    'name': 'sahi',
                    'version': '>=0.11.21',
                    'description': 'Slicing aided hyper inference library',
                    'pip_name': 'sahi>=0.11.21',
                    'is_default': True,
                    'size': '~5MB'
                }
            ]
        },
        
        'custom_packages': {
            'name': 'Paket Kustom',
            'description': 'Paket dan repositori yang didefinisikan pengguna',
            'icon': '📦',
            'color': '#607D8B',
            'packages': []  # Dynamic packages loaded from config
        }
    }
    
    # Merge requirements.txt packages into appropriate categories
    requirements_categorized = categorize_requirements_packages()
    
    for category_key, req_packages in requirements_categorized.items():
        if category_key in base_categories and req_packages:
            existing_packages = base_categories[category_key]['packages']
            existing_names = {pkg['name'].lower().replace('_', '-') for pkg in existing_packages}
            
            # Add only unique packages from requirements.txt
            for req_pkg in req_packages:
                req_name = req_pkg['name'].lower().replace('_', '-')
                if req_name not in existing_names:
                    # Update description to indicate it's from requirements.txt
                    req_pkg['description'] = f"{req_pkg['description']} (from requirements.txt)"
                    req_pkg['source'] = 'requirements.txt'
                    existing_packages.append(req_pkg)
                    existing_names.add(req_name)
    
    # Add additional packages category for uncategorized packages
    additional_packages = requirements_categorized.get('additional_packages', [])
    if additional_packages:
        base_categories['additional_packages'] = {
            'name': 'Paket Tambahan',
            'description': 'Paket dari requirements.txt yang tidak dalam kategori standar',
            'icon': '📦',
            'color': '#607D8B',
            'packages': additional_packages
        }
    
    return base_categories

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
        },
        'uninstalled_default': {
            'icon': '⭐❌',
            'color': '#FF5722',
            'text': 'Default (Uninstalled)',
            'bg_color': '#FFEBE9'
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
        'check_status': {
            'text': 'Check Status',
            'icon': '🔍',
            'color': '#2196F3',
            'disabled_color': '#CCCCCC'
        }
    }

def parse_requirements_txt(requirements_path: str = None) -> List[Dict[str, Any]]:
    """Parse requirements.txt file and convert to package format.
    
    Args:
        requirements_path: Path to requirements.txt file (optional, auto-detects if None)
        
    Returns:
        List of package dictionaries
    """
    if requirements_path is None:
        # Auto-detect requirements.txt in project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '../../../../..')
        requirements_path = os.path.normpath(os.path.join(project_root, 'requirements.txt'))
    
    packages = []
    
    if not os.path.exists(requirements_path):
        return packages
    
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse package name and version
            # Handle different version specifiers: ==, >=, <=, >, <, ~=
            import re
            match = re.match(r'^([a-zA-Z0-9_-]+)([><=~!]+.+)?$', line)
            
            if match:
                package_name = match.group(1).replace('_', '-')
                version_spec = match.group(2) or ''
                
                packages.append({
                    'name': package_name,
                    'version': version_spec,
                    'description': f'Package from requirements.txt',
                    'pip_name': line,
                    'is_default': True,
                    'size': '~Unknown',
                    'source': 'requirements.txt'
                })
    
    except Exception as e:
        print(f"Warning: Could not parse requirements.txt: {e}")
    
    return packages

def categorize_requirements_packages() -> Dict[str, List[Dict[str, Any]]]:
    """Categorize packages from requirements.txt into appropriate groups.
    
    Returns:
        Dictionary with categorized packages for each group
    """
    requirements_packages = parse_requirements_txt()
    
    # Define package mappings based on their purpose
    core_packages = {
        'ipywidgets', 'ipython', 'pyyaml', 'python-dotenv', 'pytz', 'albumentations'
    }
    
    ml_ai_packages = {
        'torch', 'torchvision', 'ultralytics', 'timm', 'scikit-learn', 
        'tensorboard', 'thop'
    }
    
    data_processing_packages = {
        'pandas', 'numpy', 'opencv-python', 'pillow', 'matplotlib', 'scipy',
        'pybboxes', 'sahi'
    }
    
    categorized = {
        'core_requirements': [],
        'ml_ai_libraries': [],
        'data_processing': [],
        'additional_packages': []
    }
    
    for pkg in requirements_packages:
        pkg_name = pkg['name'].lower().replace('_', '-')
        
        if pkg_name in core_packages:
            categorized['core_requirements'].append(pkg)
        elif pkg_name in ml_ai_packages:
            categorized['ml_ai_libraries'].append(pkg)
        elif pkg_name in data_processing_packages:
            categorized['data_processing'].append(pkg)
        else:
            # Unknown packages go to additional packages
            categorized['additional_packages'].append(pkg)
    
    return categorized