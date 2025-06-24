"""
File: smartcash/ui/setup/dependency/utils/package_selector_utils.py
Deskripsi: Package selector utilities with improved spacing and justify alignment
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS

def create_package_selector_grid(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create package selector grid with improved spacing and justify alignment"""
    
    # Package categories and definitions
    package_categories = get_package_categories()
    
    checkboxes = {}
    category_widgets = []
    
    # Create category boxes with improved spacing
    for category in package_categories:
        category_widget = _create_category_widget_improved(category)
        checkboxes.update(category_widget['checkboxes'])
        category_widgets.append(category_widget['widget'])
    
    # Create main container with consistent spacing
    container = widgets.VBox(
        category_widgets,
        layout=widgets.Layout(
            width='100%',
            margin='0 0 20px 0',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='5px',
            overflow_y='auto',
            max_height='500px'
        )
    )
    
    return {
        'container': container,
        'checkboxes': checkboxes,
        'categories': package_categories
    }

def _create_category_widget_improved(category: Dict[str, Any]) -> Dict[str, Any]:
    """Create single category widget with improved spacing and justify alignment"""
    checkboxes = {}
    
    # Create checkboxes for each package in the category
    package_widgets = []
    for pkg in category['packages']:
        pkg_key = pkg['key']
        checkbox = widgets.Checkbox(
            value=pkg.get('default', False),
            description=pkg['name'],
            disabled=not pkg.get('optional', True),
            indent=False,
            layout=widgets.Layout(
                width='auto',
                margin='2px 0',
                padding='4px 8px',
                border_radius='4px',
                border='1px solid #e0e0e0',
                background_color='#f9f9f9'
            )
        )
        
        # Add tooltip if description exists
        if 'description' in pkg:
            checkbox.add_class('has-tooltip')
            checkbox.tooltip = pkg['description']
        
        checkboxes[pkg_key] = checkbox
        
        # Create status indicator
        status = widgets.HTML(
            value='',
            placeholder='',
            description='',
            layout=widgets.Layout(
                width='24px',
                height='24px',
                margin='0 0 0 8px',
                padding='0',
                display='flex',
                align_items='center',
                justify_content='center'
            )
        )
        
        # Create row with checkbox and status
        row = widgets.HBox(
            [checkbox, status],
            layout=widgets.Layout(
                width='100%',
                justify_content='space-between',
                align_items='center',
                margin='2px 0',
                padding='2px 0'
            )
        )
        
        # Store status widget reference in checkbox for easy updates
        checkbox.status = status
        package_widgets.append(row)
    
    # Create category header
    header = widgets.HTML(
        value=f"<b>{category['name']}</b>" + 
              (f"<p style='margin: 4px 0 8px 0; color: #666; font-size: 0.9em;'>{category.get('description', '')}</p>" 
               if 'description' in category else ''),
        layout=widgets.Layout(
            width='100%',
            margin='0 0 8px 0',
            padding='0 0 8px 0',
            border_bottom='1px solid #e0e0e0'
        )
    )
    
    # Create category container
    container = widgets.VBox(
        [header] + package_widgets,
        layout=widgets.Layout(
            width='100%',
            margin='0 0 20px 0',
            padding='12px',
            border='1px solid #e0e0e0',
            border_radius='6px',
            background='white'
        )
    )
    
    return {
        'widget': container,
        'checkboxes': checkboxes
    }

def get_package_categories() -> list[Dict[str, Any]]:
    """Get package categories with updated defaults (all essential packages checked)"""
    return [
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
                    'default': True  # Essential
                },
                {
                    'key': 'notebook_deps',
                    'name': 'Notebook Dependencies',
                    'description': 'IPython dan Jupyter dependencies',
                    'pip_name': 'ipython>=8.12.0',
                    'default': True  # Essential
                },
                {
                    'key': 'albumentations',
                    'name': 'Albumentations',
                    'description': 'Augmentation library',
                    'pip_name': 'albumentations>=1.4.0',
                    'default': True  # Essential
                },
                {
                    'key': 'yaml_parser',
                    'name': 'YAML Parser',
                    'description': 'Configuration file parsing',
                    'pip_name': 'pyyaml>=6.0.0',
                    'default': True  # Essential
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
                    'default': True  # Essential untuk ML
                },
                {
                    'key': 'torchvision',
                    'name': 'TorchVision',
                    'description': 'Computer vision untuk PyTorch',
                    'pip_name': 'torchvision>=0.17.0',
                    'default': True  # Essential untuk computer vision
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
                    'default': True  # Essential untuk data processing
                },
                {
                    'key': 'numpy',
                    'name': 'NumPy',
                    'description': 'Numerical computing foundation',
                    'pip_name': 'numpy>=1.24.0,<2.0.0',  # Sementara gunakan v1.x untuk kompatibilitas
                    'default': True  # Essential untuk numerical computing
                },
                {
                    'key': 'opencv',
                    'name': 'OpenCV',
                    'description': 'Computer vision library',
                    'pip_name': 'opencv-python>=4.8.0',
                    'default': True  # Essential untuk image processing
                },
                {
                    'key': 'pillow',
                    'name': 'Pillow',
                    'description': 'Python Imaging Library',
                    'pip_name': 'Pillow>=10.0.0',
                    'default': True  # Essential untuk image manipulation
                },
                {
                    'key': 'matplotlib',
                    'name': 'Matplotlib',
                    'description': 'Plotting dan visualization',
                    'pip_name': 'matplotlib>=3.8.0',
                    'default': True  # Essential untuk visualization
                },
                {
                    'key': 'scipy',
                    'name': 'SciPy',
                    'description': 'Scientific computing library',
                    'pip_name': 'scipy>=1.12.0',
                    'default': True  # Essential untuk scientific computing
                }
            ]
        }
    ]

def update_package_status(ui_components: Dict[str, Any], package_key: str, status: str, message: str = None) -> None:
    """Update status widget for package with improved styling"""
    if not hasattr(ui_components, 'get') or 'checkboxes' not in ui_components:
        return
        
    checkboxes = ui_components['checkboxes']
    if package_key not in checkboxes:
        return
        
    checkbox = checkboxes[package_key]
    if not hasattr(checkbox, 'status'):
        return
        
    status_widget = checkbox.status
    
    # Define status styles
    status_styles = {
        'checking': ('⏳', '#FFA500'),  # Orange
        'installing': ('⏳', '#FFA500'),  # Orange
        'installed': ('✓', '#4CAF50'),   # Green
        'error': ('✗', '#F44336'),       # Red
        'warning': ('!', '#FFC107'),     # Yellow
        'info': ('ℹ️', '#2196F3'),       # Blue
        'skipped': ('⏭️', '#9E9E9E')     # Grey
    }
    
    # Get status icon and color
    icon, color = status_styles.get(status.lower(), ('', '#9E9E9E'))
    
    # Update status widget
    status_widget.value = f'<div style="color: {color}; font-weight: bold;">{icon}</div>'
    
    # Add tooltip if message is provided
    if message:
        status_widget.tooltip = message
    
    # Update checkbox style based on status
    if status.lower() == 'installed':
        checkbox.layout.border = '1px solid #4CAF50'
        checkbox.layout.background_color = '#E8F5E9'  # Light green
    elif status.lower() == 'error':
        checkbox.layout.border = '1px solid #F44336'
        checkbox.layout.background_color = '#FFEBEE'  # Light red
    elif status.lower() == 'warning':
        checkbox.layout.border = '1px solid #FFC107'
        checkbox.layout.background_color = '#FFF8E1'  # Light yellow
    else:
        checkbox.layout.border = '1px solid #e0e0e0'
        checkbox.layout.background_color = '#f9f9f9'

def get_selected_packages(ui_components: Dict[str, Any]) -> list:
    """Get list of selected packages for installation"""
    if not hasattr(ui_components, 'get') or 'checkboxes' not in ui_components:
        return []
        
    checkboxes = ui_components['checkboxes']
    selected = []
    
    for pkg_key, checkbox in checkboxes.items():
        if checkbox.value:  # If checkbox is checked
            selected.append({
                'key': pkg_key,
                'name': checkbox.description,
                'optional': not checkbox.disabled
            })
    
    return selected

def reset_package_selections(ui_components: Dict[str, Any]) -> None:
    """Reset package selections to default values"""
    if not hasattr(ui_components, 'get') or 'checkboxes' not in ui_components:
        return
        
    checkboxes = ui_components['checkboxes']
    categories = get_package_categories()
    
    # Create a mapping of package keys to their default values
    defaults = {}
    for category in categories:
        for pkg in category['packages']:
            defaults[pkg['key']] = pkg.get('default', False)
    
    # Update checkboxes
    for pkg_key, checkbox in checkboxes.items():
        if pkg_key in defaults:
            checkbox.value = defaults[pkg_key]
            # Reset status
            if hasattr(checkbox, 'status'):
                checkbox.status.value = ''
                checkbox.status.tooltip = ''
            # Reset style
            checkbox.layout.border = '1px solid #e0e0e0'
            checkbox.layout.background_color = '#f9f9f9'
