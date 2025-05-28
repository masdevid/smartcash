"""
File: smartcash/ui/setup/dependency_installer/components/dependency_installer_component.py
Deskripsi: Fixed dependency installer component menggunakan existing implementations
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_dependency_installer_ui(env=None, config=None) -> Dict[str, Any]:
    """Create UI components untuk dependency installer"""
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.dependencies_info import get_dependencies_info
    from smartcash.ui.setup.dependency_installer.utils.package_utils import get_package_categories
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    
    # Header
    header = create_header("📦 Instalasi Dependencies", "Setup package yang diperlukan untuk SmartCash")
    
    # Package categories
    package_categories = get_package_categories()
    checkboxes = {}
    category_boxes = []
    
    # Create category boxes
    for category in package_categories:
        category_box = create_category_box(category, checkboxes)
        category_boxes.append(category_box)
    
    # Packages container
    packages_container = widgets.HBox(
        category_boxes,
        layout=widgets.Layout(
            display='flex',
            flex_flow='row wrap',
            justify_content='space-between',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Custom packages
    custom_packages = widgets.Textarea(
        placeholder='Package tambahan (satu per baris)',
        layout=widgets.Layout(width='100%', height='80px')
    )
    
    custom_section = widgets.VBox([
        widgets.HTML(f"<h3>{ICONS.get('edit', '📝')} Custom Packages</h3>"),
        custom_packages
    ])
    
    # Install button
    install_button = widgets.Button(
        description='Mulai Instalasi',
        button_style='primary',
        icon='download',
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Progress tracking container
    progress_components = create_progress_tracking_container()
    
    # Status output
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'1px solid {COLORS["border"]}',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px'
        )
    )
    
    # Info box
    info_box = get_dependencies_info()
    
    # Main container
    main = widgets.VBox([
        header,
        packages_container,
        custom_section,
        widgets.HBox([install_button], layout=widgets.Layout(justify_content='center')),
        progress_components['container'],
        status,
        info_box
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # UI components
    ui_components = {
        'ui': main,
        'status': status,
        'install_button': install_button,
        'custom_packages': custom_packages,
        'progress_tracker': progress_components['tracker'],
        'module_name': 'dependency_installer',
        **checkboxes
    }
    
    return ui_components

def create_category_box(category: Dict[str, Any], checkboxes: Dict[str, Any]) -> widgets.VBox:
    """Create category box for packages"""
    from smartcash.ui.utils.constants import COLORS
    
    # Header
    header = widgets.HTML(f"""
    <div style="padding:5px 0">
        <h3 style="margin:5px 0">{category['icon']} {category['name']}</h3>
        <p style="margin:2px 0;color:{COLORS['muted']}">{category['description']}</p>
    </div>
    """)
    
    # Package rows
    package_rows = []
    for package in category['packages']:
        status_widget = widgets.HTML(f"<div style='width:100px;color:{COLORS['muted']}'>Checking...</div>")
        
        checkbox = widgets.Checkbox(
            description=package['name'],
            value=package['default'],
            tooltip=package['description'],
            layout=widgets.Layout(width='auto')
        )
        
        row = widgets.HBox([checkbox, status_widget], 
                          layout=widgets.Layout(justify_content='space-between'))
        package_rows.append(row)
        
        # Store references
        checkboxes[package['key']] = checkbox
        checkboxes[f"{package['key']}_status"] = status_widget
    
    # Category box
    return widgets.VBox([header] + package_rows, 
                       layout=widgets.Layout(
                           margin='10px 0',
                           padding='10px',
                           border=f'1px solid {COLORS["border"]}',
                           border_radius='5px',
                           width='31%',
                           min_width='250px'
                       ))