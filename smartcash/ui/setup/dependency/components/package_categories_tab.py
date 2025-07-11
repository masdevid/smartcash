"""
File: smartcash/ui/setup/dependency/components/package_categories_tab.py
Deskripsi: Tab untuk package categories dengan card layout dan real-time status
"""

import ipywidgets as widgets
from typing import Dict, Any, List

from smartcash.ui.components.form_container import create_form_container, LayoutType
from ..configs.dependency_defaults import get_default_package_categories, get_package_status_options, get_button_actions
from ..services.package_status_tracker import PackageStatusTracker

def create_package_categories_tab(config: Dict[str, Any], logger=None) -> widgets.VBox:
    """Create enhanced tab for package categories with full-width compact design."""
    
    # Get package categories and load custom packages
    categories = get_default_package_categories()
    selected_packages = config.get('selected_packages', [])
    
    # Load custom packages from config and add to custom_packages category
    categories = _load_custom_packages_to_categories(categories, config)
    
    # Initialize status tracker
    status_tracker = PackageStatusTracker(config, logger)
    
    # Create category cards with improved design
    category_cards = []
    for category_key, category_info in categories.items():
        card = create_enhanced_category_card(category_key, category_info, selected_packages, status_tracker, logger)
        category_cards.append(card)
    
    # Create full width container
    container = widgets.VBox([
        # Header
        widgets.HTML("""
        <div style="margin-bottom: 16px; padding: 0 16px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 24px;">📦</span>
                <div>
                    <h3 style="color: #333; margin: 0; font-size: 1.25rem;">Package Categories</h3>
                    <p style="color: #666; margin: 0; font-size: 0.9rem;">Select packages from predefined categories. Default packages (⭐) are recommended.</p>
                </div>
            </div>
        </div>
        """),
        
        # Category cards in a scrollable container
        widgets.VBox(
            category_cards,
            layout=widgets.Layout(
                width='100%',
                overflow_y='auto',
                padding='0 16px 16px 16px',
                gap='16px'
            )
        )
    ], layout=widgets.Layout(
        width='100%',
        height='100%',
        overflow='hidden'
    ))
    
    # Store status tracker for external access
    container.status_tracker = status_tracker
    container.category_cards = category_cards
    
    return container

def create_enhanced_category_card(category_key: str, category_info: Dict[str, Any], selected_packages: List[str], status_tracker: PackageStatusTracker, logger) -> widgets.VBox:
    """Create enhanced compact card for package category with full width layout."""
    
    icon = category_info.get('icon', '📦')
    name = category_info.get('name', category_key)
    description = category_info.get('description', '')
    color = category_info.get('color', '#2196F3')
    packages = category_info.get('packages', [])
    
    # Create compact header with better styling
    header = widgets.HTML(f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}08);
        border-left: 3px solid {color};
        padding: 10px 16px;
        border-radius: 6px 6px 0 0;
        border: 1px solid {color}30;
        border-bottom: none;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 18px;">{icon}</span>
            <div style="flex: 1; min-width: 0;">
                <h4 style="margin: 0; color: {color}; font-size: 1rem; font-weight: 600; line-height: 1.2;">
                    {name} <span style="color: #666; font-weight: 400; font-size: 0.9rem;">{len(packages)} packages</span>
                </h4>
                {f'<p style="margin: 2px 0 0 0; color: #666; font-size: 0.8rem; line-height: 1.2;">{description}</p>' if description else ''}
            </div>
        </div>
    </div>
    """)
    
    # Create compact package list in a grid layout (3 columns)
    package_widgets = []
    for pkg in packages:
        pkg_widget = create_compact_package_widget(pkg, pkg['name'] in selected_packages, status_tracker, logger)
        package_widgets.append(pkg_widget)
    
    # Create responsive 3-column grid layout for packages
    grid = widgets.GridBox(
        children=package_widgets,
        layout=widgets.Layout(
            width='100%',
            grid_template_columns='repeat(auto-fill, minmax(300px, 1fr))',
            grid_auto_rows='minmax(80px, max-content)',
            grid_gap='12px',
            padding='12px',
            overflow='hidden',
            align_items='stretch',
            justify_content='space-between'
        )
    )
    
    # Create card with full width and responsive design
    card = widgets.VBox([
        header,
        grid
    ], layout=widgets.Layout(
        width='100%',
        border='1px solid #e0e0e0',
        border_radius='8px',
        padding='0',
        background_color='white',
        margin='0 0 16px 0',
        overflow='hidden',
        min_width='0'  # Prevent horizontal overflow
    ))
    
    # Store metadata
    card.category_key = category_key
    card.category_info = category_info
    card.package_widgets = package_widgets
    
    return card

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

def create_compact_package_widget(pkg: Dict[str, Any], is_selected: bool, status_tracker: PackageStatusTracker, logger) -> widgets.VBox:
    """Create compact widget for package with name, version, and action buttons in a 3-column grid."""
    
    name = pkg.get('name', '')
    version = pkg.get('version', '')
    is_default = pkg.get('is_default', False)
    
    # Selection checkbox with smaller size
    selection_checkbox = widgets.Checkbox(
        value=is_selected,
        description='',
        layout=widgets.Layout(width='20px', height='20px', margin='0 4px 0 0')
    )
    
    # Package name and version in a single line with truncation
    name_html = widgets.HTML(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 4px;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 180px;
    ">
        <span style="
            font-weight: 500;
            color: #333;
            font-size: 0.8rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            display: inline-block;
            max-width: 150px;
        ">{name}</span>
        {f'<span style="color: #666; font-size: 0.7rem; white-space: nowrap;">v{version}</span>' if version else ''}
        {'<span style="color: #ff9800; font-size: 0.7rem; flex-shrink: 0;" title="Default package">⭐</span>' if is_default else ''}
    </div>
    """)
    
    # Create action buttons in a horizontal layout
    action_buttons = create_compact_action_buttons(pkg, status_tracker, logger)
    
    # Create main container with horizontal layout
    container = widgets.HBox(
        [
            selection_checkbox,
            widgets.VBox(
                [
                    name_html,
                    action_buttons  # Use action_buttons directly
                ],
                layout=widgets.Layout(
                    width='auto',
                    margin='0',
                    padding='0 4px',
                    align_items='flex-start'
                )
            )
        ],
        layout=widgets.Layout(
            width='100%',
            height='100%',
            min_height='50px',
            border='1px solid #e8e8e8',
            border_radius='6px',
            padding='8px',
            background_color='#f8fff8' if is_selected else 'white',
            margin='0',
            align_items='center',
            overflow='hidden'
        )
    )
    
    # Update container style on selection change
    def on_selection_change(change):
        is_selected = change['new']
        border_color = '#4CAF50' if is_selected else '#e8e8e8'
        bg_color = '#f8fff8' if is_selected else 'white'
        
        container.layout.border = f'1px solid {border_color}'
        container.layout.background_color = bg_color
    
    selection_checkbox.observe(on_selection_change, names='value')
    
    # Store references
    container.package_info = pkg
    container.selection_checkbox = selection_checkbox
    container.is_selected = is_selected
    
    return container

def create_compact_action_buttons(pkg: Dict[str, Any], status_tracker: PackageStatusTracker, logger) -> widgets.HBox:
    """Create compact action buttons for package in a 3-column grid.
    
    Args:
        pkg: Package information dictionary
        status_tracker: Package status tracker instance
        logger: Logger instance for logging
        
    Returns:
        HBox containing the action buttons
    """
    package_name = pkg['name']
    
    # Button styles
    button_style = {
        'button_width': '26px',
        'button_height': '24px',
        'font_size': '12px',
        'padding': '0',
        'margin': '0 2px',
        'button_color': '#f0f0f0',
        'button_hover': '#e0e0e0'
    }
    
    # Check if package name already has emoji
    import emoji
    name_has_emoji = any(char in emoji.EMOJI_DATA for char in package_name)
    
    # Install button with emoji (only if name doesn't have emoji)
    install_emoji = '⬇️' if not name_has_emoji else ''
    install_btn = widgets.Button(
        description=install_emoji,  # Only show emoji if name doesn't have one
        layout=widgets.Layout(
            width=button_style['button_width'],
            height=button_style['button_height'],
            padding=button_style['padding'],
            margin=button_style['margin'],
            min_width=button_style['button_width']
        ),
        tooltip=f"Install {package_name}",
        style={
            'button_color': button_style['button_color'],
            'font_size': button_style['font_size'],
            'font_weight': 'bold',
            'text_color': '#333',
            'border': '1px solid #ddd',
            'border_radius': '4px'
        }
    )
    
    # Check/Refresh button with emoji (only if name doesn't have emoji)
    check_emoji = '🔄' if not name_has_emoji else ''
    check_btn = widgets.Button(
        description=check_emoji,  # Only show emoji if name doesn't have one
        layout=widgets.Layout(
            width=button_style['button_width'],
            height=button_style['button_height'],
            padding=button_style['padding'],
            margin=button_style['margin'],
            min_width=button_style['button_width']
        ),
        tooltip=f"Check {package_name}",
        style={
            'button_color': button_style['button_color'],
            'font_size': button_style['font_size'],
            'font_weight': 'bold',
            'text_color': '#333',
            'border': '1px solid #ddd',
            'border_radius': '4px'
        }
    )
    
    # Button handlers
    def on_install_click(btn):
        if logger:
            logger.info(f"📥 Installing {package_name}...")
        status_tracker.update_package_status(package_name, 'installing')
    
    def on_check_click(btn):
        if logger:
            logger.info(f"🔍 Checking {package_name}...")
        status_tracker.check_package_status_async(package_name)
    
    install_btn.on_click(on_install_click)
    check_btn.on_click(on_check_click)
    
    return widgets.HBox([
        install_btn,
        check_btn
    ], layout=widgets.Layout(gap='4px', width='auto'))

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
    
    # Create layout with validated properties
    layout_kwargs = {
        'border': '1px solid #ddd',
        'border_radius': '8px',
        'padding': '10px',
        'background_color': '#fafafa' if is_selected else 'white'
    }
    
    # Create container with validated layout
    container = widgets.HBox(
        [selection_checkbox, info_html, status_widget, action_buttons],
        layout=widgets.Layout(**layout_kwargs)
    )
    
    # Update container style based on selection
    def on_selection_change(change):
        is_selected = change['new']
        border_color = '#4CAF50' if is_selected else '#ddd'
        bg_color = '#fafafa' if is_selected else 'white'
        
        # Create new layout with updated properties
        new_layout = container.layout.copy()
        new_layout.border = f'1px solid {border_color}'
        new_layout.background_color = bg_color
        
        # Apply the new layout
        container.layout = new_layout
    
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

def _load_custom_packages_to_categories(categories: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Load custom packages from config and add to custom_packages category."""
    
    # Get custom packages from config
    custom_packages_text = config.get('custom_packages', '')
    
    if custom_packages_text:
        custom_packages = []
        for line in custom_packages_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse package name (handle version specs)
                package_name = line.split('>')[0].split('<')[0].split('=')[0].strip()
                
                custom_packages.append({
                    'name': package_name,
                    'version': '',  # Version handled in pip_name
                    'description': f'Custom package: {line}',
                    'pip_name': line,
                    'is_default': False,  # Custom packages are not default
                    'size': '~Unknown'
                })
        
        # Update custom_packages category
        categories['custom_packages']['packages'] = custom_packages
    
    return categories