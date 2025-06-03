"""
File: smartcash/ui/setup/dependency_installer/components/package_selector.py
Deskripsi: Package selector grid component dengan kategori packages
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS

def create_package_selector_grid(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create package selector grid dengan kategori"""
    
    # Package categories dan definitions
    package_categories = get_package_categories()
    
    checkboxes = {}
    category_widgets = []
    
    # Create category boxes
    for category in package_categories:
        category_widget, category_checkboxes = _create_category_widget(category)
        category_widgets.append(category_widget)
        checkboxes.update(category_checkboxes)
    
    # Grid container dengan responsive layout
    grid_container = widgets.HBox(
        category_widgets,
        layout=widgets.Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-between',
            align_items='stretch',
            width='100%',
            margin='10px 0',
            overflow='hidden'
        )
    )
    
    return {
        'container': grid_container,
        'categories': package_categories,
        'checkboxes': checkboxes,
        'category_widgets': category_widgets
    }

def _create_category_widget(category: Dict[str, Any]) -> tuple[widgets.VBox, Dict[str, Any]]:
    """Create single category widget dengan packages"""
    
    # Header kategori
    header = widgets.HTML(f"""
    <div style="padding:8px 0; border-bottom:1px solid {COLORS.get('border', '#ddd')}; margin-bottom:8px; text-align:center;">
        <h4 style="margin:0; color:{COLORS.get('primary', '#007bff')}">{category['icon']} {category['name']}</h4>
        <small style="color:{COLORS.get('muted', '#666')}">{category['description']}</small>
    </div>
    """)
    
    # Package checkboxes dan status widgets
    package_widgets = []
    category_checkboxes = {}
    
    for package in category['packages']:
        package_key = package['key']
        
        # Status widget untuk setiap package
        status_widget = widgets.HTML(
            f"<span style='color:{COLORS.get('info', '#17a2b8')};font-size:11px;'>🔍 Checking...</span>",
            layout=widgets.Layout(width='90px', margin='0')
        )
        
        # Checkbox untuk package
        checkbox = widgets.Checkbox(
            description=package['name'],
            value=package.get('default', True),
            tooltip=package.get('description', ''),
            layout=widgets.Layout(width='calc(100% - 80px)', margin='2px 0')
        )
        
        # Row container
        row = widgets.HBox(
            [checkbox, status_widget], 
            layout=widgets.Layout(
                width='90%',
                justify_content='space-between',
                align_items='center',
                margin='3px 0',
                padding='0',
                overflow='hidden'
            )
        )
        
        package_widgets.append(row)
        
        # Store references dengan naming convention
        category_checkboxes[package_key] = checkbox
        category_checkboxes[f"{package_key}_status"] = status_widget
    
    # Category container
    category_container = widgets.VBox(
        [header] + package_widgets, 
        layout=widgets.Layout(
            width='32%',
            max_width='32%',
            margin='0',
            padding='10px',
            border=f'1px solid {COLORS.get("border", "#ddd")}',
            border_radius='6px',
            overflow='hidden',
            box_sizing='border-box',
            flex_grow='1',
            display='flex',
            flex_direction='column'
        )
    )
    
    return category_container, category_checkboxes

def get_package_categories() -> list[Dict[str, Any]]:
    """Get package categories dengan definisi packages"""
    return [
        {
            'name': 'Core Requirements',
            'icon': '🔧',
            'description': 'Package inti SmartCash',
            'packages': [
                {
                    'key': 'smartcash_core',
                    'name': 'SmartCash Core',
                    'description': 'Core utilities dan helpers',
                    'pip_name': 'ipywidgets>=7.6.0',
                    'default': True
                },
                {
                    'key': 'notebook_deps',
                    'name': 'Notebook Dependencies',
                    'description': 'IPython dan Jupyter dependencies',
                    'pip_name': 'ipython>=7.0.0',
                    'default': True
                },
                {
                    'key': 'file_utils',
                    'name': 'File Utilities',
                    'description': 'Path handling dan file operations',
                    'pip_name': 'pathlib2>=2.3.0',
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
                    'description': 'Deep learning framework',
                    'pip_name': 'torch>=1.9.0',
                    'default': True
                },
                {
                    'key': 'torchvision',
                    'name': 'TorchVision',
                    'description': 'Computer vision untuk PyTorch',
                    'pip_name': 'torchvision>=0.10.0',
                    'default': True
                },
                {
                    'key': 'yolov5',
                    'name': 'YOLOv5',
                    'description': 'Object detection model',
                    'pip_name': 'yolov5>=6.0.0',
                    'default': True
                },
                {
                    'key': 'ultralytics',
                    'name': 'Ultralytics',
                    'description': 'YOLO implementation',
                    'pip_name': 'ultralytics>=8.0.0',
                    'default': False
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
                    'pip_name': 'pandas>=1.3.0',
                    'default': True
                },
                {
                    'key': 'numpy',
                    'name': 'NumPy',
                    'description': 'Numerical computing',
                    'pip_name': 'numpy>=1.21.0',
                    'default': True
                },
                {
                    'key': 'opencv',
                    'name': 'OpenCV',
                    'description': 'Computer vision library',
                    'pip_name': 'opencv-python>=4.5.0',
                    'default': True
                },
                {
                    'key': 'pillow',
                    'name': 'Pillow',
                    'description': 'Image processing library',
                    'pip_name': 'Pillow>=8.0.0',
                    'default': True
                },
                {
                    'key': 'matplotlib',
                    'name': 'Matplotlib',
                    'description': 'Plotting dan visualization',
                    'pip_name': 'matplotlib>=3.4.0',
                    'default': False
                }
            ]
        }
    ]

def update_package_status(ui_components: Dict[str, Any], package_key: str, status: str, message: str = None):
    """Update status widget untuk package tertentu"""
    status_widget_key = f"{package_key}_status"
    
    if status_widget_key in ui_components:
        status_widget = ui_components[status_widget_key]
        
        # Status icons dan colors
        status_config = {
            'checking': {'icon': '🔍', 'color': COLORS.get('info', '#17a2b8'), 'text': 'Checking...'},
            'installed': {'icon': '✅', 'color': COLORS.get('success', '#28a745'), 'text': 'Terinstall'},
            'missing': {'icon': '❌', 'color': COLORS.get('danger', '#dc3545'), 'text': 'Tidak terinstall'},
            'upgrade': {'icon': '⚠️', 'color': COLORS.get('warning', '#ffc107'), 'text': 'Perlu update'},
            'installing': {'icon': '⏳', 'color': COLORS.get('info', '#17a2b8'), 'text': 'Installing...'},
            'error': {'icon': '💥', 'color': COLORS.get('danger', '#dc3545'), 'text': 'Error'}
        }
        
        config = status_config.get(status, status_config['checking'])
        display_text = message or config['text']
        
        status_widget.value = f"<span style='color:{config['color']};font-size:11px;'>{config['icon']} {display_text}</span>"

def get_selected_packages(ui_components: Dict[str, Any]) -> list[str]:
    """Get list package yang dipilih untuk diinstall"""
    selected_packages = []
    package_categories = get_package_categories()
    
    for category in package_categories:
        for package in category['packages']:
            package_key = package['key']
            if package_key in ui_components:
                checkbox = ui_components[package_key]
                if checkbox and checkbox.value:
                    selected_packages.append(package['pip_name'])
    
    # Add custom packages
    if 'custom_packages' in ui_components:
        custom_text = ui_components['custom_packages'].value.strip()
        if custom_text:
            custom_packages = [pkg.strip() for pkg in custom_text.split('\n') if pkg.strip()]
            selected_packages.extend(custom_packages)
    
    return selected_packages

def reset_package_selections(ui_components: Dict[str, Any]):
    """Reset package selections ke default values"""
    package_categories = get_package_categories()
    
    for category in package_categories:
        for package in category['packages']:
            package_key = package['key']
            if package_key in ui_components:
                checkbox = ui_components[package_key]
                if checkbox:
                    checkbox.value = package.get('default', True)
            
            # Reset status
            update_package_status(ui_components, package_key, 'checking')
    
    # Clear custom packages
    if 'custom_packages' in ui_components:
        ui_components['custom_packages'].value = ''