"""
File: smartcash/ui/setup/dependency/components/package_categories_tab.py
Deskripsi: Tab untuk package categories dengan card layout dan real-time status
"""

import ipywidgets as widgets
from typing import Dict, Any, List

from smartcash.ui.components.form_container import create_form_container, LayoutType
from ..configs.dependency_defaults import get_default_package_categories, get_package_status_options, get_button_actions
from ..utils.package_status_tracker import PackageStatusTracker

def create_package_categories_tab(config: Dict[str, Any], logger) -> widgets.VBox:
    """Create tab untuk package categories"""
    
    # Get package categories
    categories = get_default_package_categories()
    selected_packages = config.get('selected_packages', [])
    
    # Initialize status tracker
    status_tracker = PackageStatusTracker(config, logger)
    
    # Create category cards
    category_cards = []
    for category_key, category_info in categories.items():
        card = create_category_card(category_key, category_info, selected_packages, status_tracker, logger)
        category_cards.append(card)
    
    # Create form container with grid layout
    form_container = create_form_container(
        layout_type=LayoutType.GRID,
        grid_columns='repeat(auto-fit, minmax(400px, 1fr))',
        gap='20px',
        container_padding='20px'
    )
    
    # Add category cards to the grid
    for card in category_cards:
        form_container['add_item'](card, height='auto')
    
    # Create header
    header = widgets.HTML("""
    <div style="margin-bottom: 20px;">
        <h3 style="color: #333; margin: 0 0 10px 0;">📦 Package Categories</h3>
        <p style="color: #666; margin: 0;">Pilih packages dari kategori yang tersedia. Default packages ditandai dengan ⭐.</p>
    </div>
    """)
    
    # Create container
    container = widgets.VBox([
        header,
        form_container['container']
    ])
    
    # Store status tracker untuk access dari luar
    container.status_tracker = status_tracker
    
    return container

def create_category_card(category_key: str, category_info: Dict[str, Any], selected_packages: List[str], status_tracker: PackageStatusTracker, logger) -> widgets.VBox:
    """Create card untuk satu category"""
    
    icon = category_info.get('icon', '📦')
    name = category_info.get('name', category_key)
    description = category_info.get('description', '')
    color = category_info.get('color', '#333')
    packages = category_info.get('packages', [])
    
    # Create header
    header = widgets.HTML(f"""
    <div style="
        background: linear-gradient(135deg, {color}20, {color}10);
        border-left: 4px solid {color};
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 8px;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">{icon}</span>
            <div>
                <h4 style="margin: 0; color: {color};">{name}</h4>
                <p style="margin: 5px 0 0 0; color: #666; font-size: 14px;">{description}</p>
            </div>
        </div>
    </div>
    """)
    
    # Create package list
    package_widgets = []
    for pkg in packages:
        pkg_widget = create_package_widget(pkg, pkg['name'] in selected_packages, status_tracker, logger)
        package_widgets.append(pkg_widget)
    
    # Create card
    card = widgets.VBox([
        header,
        widgets.VBox(package_widgets, layout=widgets.Layout(gap='10px'))
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='10px',
        padding='0',
        background_color='white',
        box_shadow='0 2px 8px rgba(0,0,0,0.1)'
    ))
    
    return card

def create_package_widget(pkg: Dict[str, Any], is_selected: bool, status_tracker: PackageStatusTracker, logger) -> widgets.HBox:
    """Create widget untuk satu package dengan real-time status"""
    
    name = pkg.get('name', '')
    version = pkg.get('version', '')
    description = pkg.get('description', '')
    size = pkg.get('size', '')
    is_default = pkg.get('is_default', False)
    
    # Selection checkbox
    selection_checkbox = widgets.Checkbox(
        value=is_selected,
        description='',
        layout=widgets.Layout(width='30px')
    )
    
    # Default indicator
    default_indicator = '⭐' if is_default else ''
    
    # Package info
    info_html = widgets.HTML(f"""
    <div style="flex: 1; min-width: 0;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
            <span style="font-weight: bold; color: #333;">{name}</span>
            <span style="color: #666; font-size: 12px;">{version}</span>
            <span style="color: #ffa500;">{default_indicator}</span>
        </div>
        <div style="color: #666; font-size: 12px; margin-bottom: 3px;">{description}</div>
        <div style="color: #888; font-size: 11px;">{size}</div>
    </div>
    """)
    
    # Status widget dengan real-time updates
    status_widget = status_tracker.create_status_widget(name)
    
    # Action buttons
    action_buttons = create_package_action_buttons(pkg, status_tracker, logger)
    
    # Container
    container = widgets.HBox([
        selection_checkbox,
        info_html,
        status_widget,
        action_buttons
    ], layout=widgets.Layout(
        border='1px solid #ddd',
        border_radius='8px',
        padding='10px',
        background_color='#fafafa' if is_selected else 'white',
        align_items='center'
    ))
    
    # Update container style berdasarkan selection
    def on_selection_change(change):
        is_selected = change['new']
        border_color = '#4CAF50' if is_selected else '#ddd'
        bg_color = '#fafafa' if is_selected else 'white'
        
        container.layout.border = f'1px solid {border_color}'
        container.layout.background_color = bg_color
    
    selection_checkbox.observe(on_selection_change, names='value')
    
    # Store references
    container.package_info = pkg
    container.selection_checkbox = selection_checkbox
    container.status_widget = status_widget
    container.is_selected = is_selected
    
    return container

def create_package_action_buttons(pkg: Dict[str, Any], status_tracker: PackageStatusTracker, logger) -> widgets.HBox:
    """Create action buttons untuk package"""
    
    actions = get_button_actions()
    package_name = pkg['name']
    
    # Install button
    install_btn = widgets.Button(
        description=actions['install']['text'],
        button_style='primary',
        layout=widgets.Layout(width='80px', height='30px'),
        tooltip=f"Install {package_name}"
    )
    
    # Check button
    check_btn = widgets.Button(
        description=actions['check']['text'],
        button_style='info',
        layout=widgets.Layout(width='70px', height='30px'),
        tooltip=f"Check status {package_name}"
    )
    
    # Uninstall button
    uninstall_btn = widgets.Button(
        description=actions['uninstall']['text'],
        button_style='danger',
        layout=widgets.Layout(width='90px', height='30px'),
        tooltip=f"Uninstall {package_name}"
    )
    
    # Button handlers
    def on_install_click(btn):
        logger.info(f"📥 Installing {package_name}...")
        # Trigger install operation
        status_tracker.update_package_status(package_name, 'installing')
    
    def on_check_click(btn):
        logger.info(f"🔍 Checking {package_name}...")
        status_tracker.check_package_status_async(package_name)
    
    def on_uninstall_click(btn):
        logger.info(f"🗑️ Uninstalling {package_name}...")
        status_tracker.update_package_status(package_name, 'uninstalling')
    
    install_btn.on_click(on_install_click)
    check_btn.on_click(on_check_click)
    uninstall_btn.on_click(on_uninstall_click)
    
    return widgets.HBox([
        install_btn,
        check_btn,
        uninstall_btn
    ], layout=widgets.Layout(gap='5px'))