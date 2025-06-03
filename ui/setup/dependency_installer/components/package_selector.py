"""
File: smartcash/ui/setup/dependency_installer/components/package_selector.py
Deskripsi: Package selector grid dengan improved spacing dan justify alignment
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import COLORS, ICONS

def create_package_selector_grid(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create package selector grid dengan improved spacing dan justify alignment"""
    
    # Package categories dan definitions
    package_categories = get_package_categories()
    
    checkboxes = {}
    category_widgets = []
    
    # Create category boxes dengan improved spacing
    for category in package_categories:
        category_widget, category_checkboxes = _create_category_widget_improved(category)
        category_widgets.append(category_widget)
        checkboxes.update(category_checkboxes)
    
    # Grid container dengan responsive layout dan better spacing
    grid_container = widgets.HBox(
        category_widgets,
        layout=widgets.Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-between',
            align_items='stretch',
            width='100%',
            margin='15px 0',
            padding='0 5px',
            gap='15px',  # CSS gap untuk better spacing
            overflow='hidden',
            box_sizing='border-box'
        )
    )
    
    return {
        'container': grid_container,
        'categories': package_categories,
        'checkboxes': checkboxes,
        'category_widgets': category_widgets
    }

def _create_category_widget_improved(category: Dict[str, Any]) -> tuple[widgets.VBox, Dict[str, Any]]:
    """Create single category widget dengan improved spacing dan justify alignment"""
    
    # Header kategori dengan better styling
    header = widgets.HTML(f"""
    <div style="padding: 12px 8px; border-bottom: 2px solid {COLORS.get('primary', '#007bff')}; 
                margin-bottom: 12px; text-align: center; background: linear-gradient(45deg, {COLORS.get('light', '#f8f9fa')}, #ffffff);">
        <h4 style="margin: 0 0 4px 0; color: {COLORS.get('primary', '#007bff')}; font-size: 15px; font-weight: 600;">
            {category['icon']} {category['name']}
        </h4>
        <small style="color: {COLORS.get('muted', '#666')}; font-size: 11px; line-height: 1.3;">
            {category['description']}
        </small>
    </div>
    """)
    
    # Package checkboxes dan status widgets dengan improved layout
    package_widgets = []
    category_checkboxes = {}
    
    for package in category['packages']:
        package_key = package['key']
        
        # Status widget untuk setiap package dengan better styling
        status_widget = widgets.HTML(
            f"""<span style='color: {COLORS.get('info', '#17a2b8')}; font-size: 11px; 
                        padding: 2px 6px; background: #f8f9fa; border-radius: 3px; 
                        border: 1px solid #e9ecef; white-space: nowrap;'>
                🔍 Checking...
                </span>""",
            layout=widgets.Layout(width='110px', margin='0', flex='0 0 auto')
        )
        
        # Checkbox untuk package dengan improved styling dan essential indicator
        package_name_display = package['name']
        if package.get('default', True):
            package_name_display = f"⭐ {package['name']}"  # Star untuk essential packages
        
        checkbox = widgets.Checkbox(
            description=package_name_display,
            value=package.get('default', True),
            tooltip=f"{'[ESSENTIAL] ' if package.get('default', True) else '[OPTIONAL] '}{package.get('description', '')} | {package.get('pip_name', '')}",
            layout=widgets.Layout(
                width='auto', 
                margin='0', 
                flex='1 1 auto',
                max_width='calc(100% - 120px)'
            ),
            style={'description_width': 'initial'}
        )
        
        # Row container dengan justify space-between dan proper alignment
        row = widgets.HBox(
            [checkbox, status_widget], 
            layout=widgets.Layout(
                width='100%',
                justify_content='space-between',
                align_items='center',
                margin='6px 0',
                padding='4px 6px',
                border_radius='4px',
                background_color='transparent',
                overflow='hidden',
                box_sizing='border-box'
            )
        )
        
        # Add hover effect dengan CSS-like styling
        row.add_class('package-row')
        
        package_widgets.append(row)
        
        # Store references dengan naming convention
        category_checkboxes[package_key] = checkbox
        category_checkboxes[f"{package_key}_status"] = status_widget
    
    # Category container dengan improved styling
    category_container = widgets.VBox(
        [header] + package_widgets, 
        layout=widgets.Layout(
            width='32%',
            max_width='32%',
            min_height='300px',
            margin='0',
            padding='12px',
            border=f'1px solid {COLORS.get("border", "#ddd")}',
            border_radius='8px',
            background_color='#fafbfc',
            overflow='hidden',
            box_sizing='border-box',
            flex_grow='1',
            display='flex',
            flex_direction='column'
        )
    )
    
    return category_container, category_checkboxes

def get_package_categories() -> list[Dict[str, Any]]:
    """Get package categories dengan updated defaults (semua essential packages checked)"""
    return [
        {
            'name': 'Core Requirements',
            'icon': '🔧',
            'description': 'Package inti SmartCash',
            'packages': [
                {
                    'key': 'smartcash_core',
                    'name': 'SmartCash Core',
                    'description': 'Core utilities dan helpers untuk UI',
                    'pip_name': 'ipywidgets>=7.6.0',
                    'default': True  # Essential
                },
                {
                    'key': 'notebook_deps',
                    'name': 'Notebook Dependencies',
                    'description': 'IPython dan Jupyter dependencies',
                    'pip_name': 'ipython>=7.0.0',
                    'default': True  # Essential
                },
                {
                    'key': 'file_utils',
                    'name': 'File Utilities',
                    'description': 'Path handling dan file operations',
                    'pip_name': 'pathlib2>=2.3.0',
                    'default': True  # Essential
                },
                {
                    'key': 'yaml_parser',
                    'name': 'YAML Parser',
                    'description': 'Configuration file parsing',
                    'pip_name': 'pyyaml>=5.4.0',
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
                    'pip_name': 'torch>=1.9.0',
                    'default': True  # Essential untuk ML
                },
                {
                    'key': 'torchvision',
                    'name': 'TorchVision',
                    'description': 'Computer vision untuk PyTorch',
                    'pip_name': 'torchvision>=0.10.0',
                    'default': True  # Essential untuk computer vision
                },
                {
                    'key': 'yolov5',
                    'name': 'YOLOv5',
                    'description': 'Object detection model',
                    'pip_name': 'yolov5>=6.0.0',
                    'default': True  # Essential untuk SmartCash detection
                },
                {
                    'key': 'ultralytics',
                    'name': 'Ultralytics',
                    'description': 'YOLO implementation terbaru',
                    'pip_name': 'ultralytics>=8.0.0',
                    'default': False  # Optional (alternative)
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
                    'default': True  # Essential untuk data processing
                },
                {
                    'key': 'numpy',
                    'name': 'NumPy',
                    'description': 'Numerical computing foundation',
                    'pip_name': 'numpy>=1.21.0',
                    'default': True  # Essential untuk numerical computing
                },
                {
                    'key': 'opencv',
                    'name': 'OpenCV',
                    'description': 'Computer vision library',
                    'pip_name': 'opencv-python>=4.5.0',
                    'default': True  # Essential untuk image processing
                },
                {
                    'key': 'pillow',
                    'name': 'Pillow',
                    'description': 'Python Imaging Library',
                    'pip_name': 'Pillow>=8.0.0',
                    'default': True  # Essential untuk image manipulation
                },
                {
                    'key': 'matplotlib',
                    'name': 'Matplotlib',
                    'description': 'Plotting dan visualization',
                    'pip_name': 'matplotlib>=3.4.0',
                    'default': True  # Essential untuk visualization
                }
            ]
        }
    ]

def update_package_status(ui_components: Dict[str, Any], package_key: str, status: str, message: str = None):
    """Update status widget untuk package dengan improved styling"""
    status_widget_key = f"{package_key}_status"
    
    if status_widget_key in ui_components:
        status_widget = ui_components[status_widget_key]
        
        # Status icons dan colors dengan improved styling
        status_config = {
            'checking': {
                'icon': '🔍', 
                'color': COLORS.get('info', '#17a2b8'), 
                'bg_color': '#e3f2fd',
                'border_color': '#90caf9',
                'text': 'Checking...'
            },
            'installed': {
                'icon': '✅', 
                'color': COLORS.get('success', '#28a745'), 
                'bg_color': '#e8f5e9',
                'border_color': '#81c784',
                'text': 'Terinstall'
            },
            'missing': {
                'icon': '❌', 
                'color': COLORS.get('danger', '#dc3545'), 
                'bg_color': '#ffebee',
                'border_color': '#e57373',
                'text': 'Tidak terinstall'
            },
            'upgrade': {
                'icon': '⚠️', 
                'color': COLORS.get('warning', '#ffc107'), 
                'bg_color': '#fff8e1',
                'border_color': '#ffb74d',
                'text': 'Perlu update'
            },
            'installing': {
                'icon': '⏳', 
                'color': COLORS.get('info', '#17a2b8'), 
                'bg_color': '#e1f5fe',
                'border_color': '#4fc3f7',
                'text': 'Installing...'
            },
            'error': {
                'icon': '💥', 
                'color': COLORS.get('danger', '#dc3545'), 
                'bg_color': '#ffebee',
                'border_color': '#f44336',
                'text': 'Error'
            }
        }
        
        config = status_config.get(status, status_config['checking'])
        display_text = message or config['text']
        
        # Improved status widget dengan better styling
        status_widget.value = f"""
        <span style='color: {config['color']}; font-size: 11px; font-weight: 500;
                     padding: 3px 8px; background: {config['bg_color']}; 
                     border-radius: 4px; border: 1px solid {config['border_color']}; 
                     white-space: nowrap; display: inline-block; text-align: center;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            {config['icon']} {display_text}
        </span>
        """

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
            
            # Reset status ke checking
            update_package_status(ui_components, package_key, 'checking')
    
    # Clear custom packages
    if 'custom_packages' in ui_components:
        ui_components['custom_packages'].value = ''