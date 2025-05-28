"""
File: smartcash/ui/setup/dependency_installer/components/dependency_installer_component.py
Deskripsi: Fixed dependency installer component menggunakan log accordion dan centered category items
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_dependency_installer_ui(env=None, config=None) -> Dict[str, Any]:
    """Create UI components untuk dependency installer dengan log accordion"""
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.info_boxes.dependencies_info import get_dependencies_info
    from smartcash.ui.setup.dependency_installer.utils.package_utils import get_package_categories
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    from smartcash.ui.components.log_accordion import create_log_accordion
    
    # Header
    header = create_header("📦 Instalasi Dependencies", "Setup package yang diperlukan untuk SmartCash")
    
    # Status panel
    status_panel = widgets.HTML(
        value=f"""
        <div style="padding:8px 12px; background-color:#d1ecf1; 
                   color:#0c5460; border-radius:4px; margin:10px 0;
                   border-left:4px solid #17a2b8;">
            <p style="margin:3px 0">ℹ️ Pilih packages yang akan diinstall dan klik "Mulai Instalasi"</p>
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Package categories
    package_categories = get_package_categories()
    checkboxes = {}
    category_boxes = []
    
    # Create category boxes dengan centered alignment
    for category in package_categories:
        category_box = create_category_box(category, checkboxes)
        category_boxes.append(category_box)
    
    # Packages container - 3 columns dengan proper alignment
    packages_container = widgets.HBox(
        category_boxes,
        layout=widgets.Layout(
            display='flex',
            flex_flow='row nowrap',
            justify_content='space-evenly',  # Changed to space-evenly for better distribution
            align_items='flex-start',  # Added for proper top alignment
            width='100%',
            margin='10px 0',
            overflow='hidden'
        )
    )
    
    # Custom packages section
    custom_packages = widgets.Textarea(
        placeholder='Package tambahan (satu per baris)',
        layout=widgets.Layout(width='100%', height='80px')
    )
    
    custom_section = widgets.VBox([
        widgets.HTML(f"<h3>{ICONS.get('edit', '📝')} Custom Packages</h3>"),
        custom_packages
    ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    
    # Install button
    install_button = widgets.Button(
        description='Mulai Instalasi',
        button_style='primary',
        icon='download',
        layout=widgets.Layout(margin='10px 0')
    )
    
    # Button container untuk center alignment
    button_container = widgets.HBox(
        [install_button],
        layout=widgets.Layout(
            justify_content='center',
            width='100%',
            margin='10px 0'
        )
    )
    
    # Progress tracking container
    progress_components = create_progress_tracking_container()
    
    # Log accordion untuk menggantikan status output
    log_components = create_log_accordion(
        module_name='dependency_installer',
        height='250px',
        width='100%'
    )
    
    # Info box
    info_box = get_dependencies_info()
    
    # Main container
    main = widgets.VBox([
        header,
        status_panel,
        packages_container,
        custom_section,
        button_container,
        progress_components['container'],
        log_components['log_accordion'],  # Menggunakan log accordion
        info_box
    ], layout=widgets.Layout(width='100%', max_width='100%', padding='10px', overflow='hidden'))
    
    # UI components
    ui_components = {
        'ui': main,
        'log_output': log_components['log_output'],  # Key untuk log output
        'status_panel': status_panel,
        'install_button': install_button,
        'custom_packages': custom_packages,
        'progress_tracker': progress_components['tracker'],
        'module_name': 'dependency_installer',
        **checkboxes
    }
    
    return ui_components

def create_category_box(category: Dict[str, Any], checkboxes: Dict[str, Any]) -> widgets.VBox:
    """Create category box dengan centered alignment untuk items"""
    from smartcash.ui.utils.constants import COLORS
    
    # Header dengan centered alignment
    header = widgets.HTML(f"""
    <div style="padding:8px 0; border-bottom:1px solid {COLORS['border']}; 
               margin-bottom:8px; text-align:center;">
        <h4 style="margin:0; color:{COLORS['primary']}">{category['icon']} {category['name']}</h4>
        <small style="color:{COLORS['muted']}">{category['description']}</small>
    </div>
    """)
    
    # Package checkboxes dengan centered alignment
    package_widgets = []
    for package in category['packages']:
        status_widget = widgets.HTML(
            f"<span style='color:{COLORS['muted']};font-size:11px;white-space:nowrap;text-align:center;'>Checking...</span>",
            layout=widgets.Layout(width='70px', margin='0', flex='0 0 auto')
        )
        
        checkbox = widgets.Checkbox(
            description=package['name'],
            value=package['default'],
            tooltip=package['description'],
            layout=widgets.Layout(
                flex='1 1 auto',
                margin='2px 0',
                max_width='calc(100% - 80px)'
            )
        )
        
        # Horizontal row dengan center-aligned content
        row = widgets.HBox([checkbox, status_widget], 
                          layout=widgets.Layout(
                              width='100%',
                              max_width='100%',
                              justify_content='space-between',
                              align_items='center',
                              margin='3px 0',
                              padding='0 8px',  # Added horizontal padding for better spacing
                              overflow='hidden',
                              box_sizing='border-box'
                          ))
        package_widgets.append(row)
        
        # Store references
        checkboxes[package['key']] = checkbox
        checkboxes[f"{package['key']}_status"] = status_widget
    
    # Category container dengan centered items
    return widgets.VBox([header] + package_widgets, 
                       layout=widgets.Layout(
                           width='300px',  # Fixed width untuk consistent sizing
                           max_width='300px',
                           margin='5px',
                           padding='12px',
                           border=f'1px solid {COLORS["border"]}',
                           border_radius='6px',
                           overflow='hidden',
                           align_items='stretch',  # Stretch items untuk consistent width
                           box_sizing='border-box'
                       ))