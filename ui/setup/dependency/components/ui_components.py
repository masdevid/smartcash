"""
File: smartcash/ui/setup/dependency/components/ui_components.py
Module: smartcash.ui.setup.dependency.components.ui_components
Description: Consolidated UI components for dependency management in SmartCash

This module provides UI components for managing Python package dependencies
in the SmartCash project, including package selection, custom package input,
and dependency resolution interfaces.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple
import ipywidgets as widgets

# Core UI components from central location
from smartcash.ui.components.header import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.card import create_card
from smartcash.ui.components.dialog import create_confirmation_area

# Local components for package-specific UIs
from smartcash.ui.setup.dependency.utils.ui_utils import create_package_checkbox

# Local component imports
from smartcash.ui.setup.dependency.components.summary_panel import DependencySummaryPanel

# ======================== MAIN UI ========================

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create main dependency management UI with consistent structure
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing UI components and state
    """
    logger = get_logger(__name__)
    
    try:
        config = config or {}
        ui_components = {}
        
        # === CORE COMPONENTS ===
        header = create_header(
            title="📦 Manajemen Dependensi",
            description="Kelola dependensi Python untuk proyek SmartCash",
            icon="settings"
        )
        
        # Status panel
        status_panel = create_status_panel()
        
        # Create action buttons with consistent styling
        action_buttons = create_action_buttons(
            primary_button={"label": "⚙️ Install Dependensi", "style": "primary"},
            secondary_buttons=[
                {"label": "🔄 Periksa Pembaruan", "style": "info"},
                {"label": "🗑️ Hapus Terpilih", "style": "danger"}
            ]
        )
        
        # Initialize button references with fallbacks
        install_btn = action_buttons.get('primary')
        check_updates_btn = action_buttons.get('secondary_0')
        uninstall_btn = action_buttons.get('secondary_1')
        
        # Fallback button creation if any are missing
        if not install_btn:
            install_btn = widgets.Button(
                description='⚙️ Install Dependensi',
                button_style='primary',
                layout=widgets.Layout(width='auto')
            )
        
        # Create main content sections
        categories = create_categories_section(config)
        custom_packages = create_custom_packages_section()
        
        # Create tab layout
        tabs = widgets.Tab(children=[categories, custom_packages])
        tabs.set_title(0, "📦 Paket")
        tabs.set_title(1, "➕ Kustom")
        tabs.selected_index = 0
        tabs.layout = widgets.Layout(width='100%', margin='10px 0')
        
        # Create save/reset buttons
        save_reset_buttons = create_save_reset_buttons()
        save_btn = save_reset_buttons.get('save_button')
        reset_btn = save_reset_buttons.get('reset_button')
        
        # Create button containers
        action_buttons_container = widgets.HBox(
            [install_btn, check_updates_btn, uninstall_btn],
            layout=widgets.Layout(
                justify_content='flex-end',
                margin='10px 0',
                width='100%'
            )
        )
        
        save_reset_container = widgets.HBox(
            [save_btn, reset_btn],
            layout=widgets.Layout(
                justify_content='flex-end',
                margin='10px 0',
                width='100%'
            )
        )
        
        # Create log accordion
        log_components = create_log_accordion()
        log_accordion = log_components.get('accordion')
        
        # Main content layout
        content = widgets.VBox([
            header,
            status_panel,
            widgets.HTML('<hr style="margin: 10px 0; border: 0.5px solid #e0e0e0;">'),
            tabs,
            widgets.HTML('<hr style="margin: 15px 0; border: 0.5px solid #e0e0e0;">'),
            action_buttons_container,
            save_reset_container
        ], layout=widgets.Layout(width='100%'))
        
        # Combine main content and logs
        layout = widgets.VBox([
            content,
            log_accordion
        ], layout=widgets.Layout(width='100%'))
        
        # Store all components
        ui_components.update({
            'ui': layout,
            'log_output': log_components.get('log_output'),
            'log_accordion': log_accordion,
            'header': header,
            'status_panel': status_panel,
            'tabs': tabs,
            'install_button': install_btn,
            'check_updates_button': check_updates_btn,
            'uninstall_button': uninstall_btn,
            'save_button': save_btn,
            'reset_button': reset_btn,
            'action_buttons_container': action_buttons_container,
            'save_reset_container': save_reset_container
        })
        
        # Initialize confirmation area
        ui_components['confirmation_area'] = create_confirmation_area(ui_components)
        
        return ui_components
        
    except Exception as e:
        logger.error(f"Error creating dependency UI: {str(e)}", exc_info=True)
        raise

# ======================== CATEGORIES ========================

def create_categories_section(
    config: Dict[str, Any],
    on_package_select: Optional[Callable[[str, bool], None]] = None
) -> widgets.Widget:
    """Create categories section with package lists
    
    Args:
        config: Configuration dictionary with categories and selected packages
        on_package_select: Callback when package selection changes
        
    Returns:
        Widget containing categories section
    """
    from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config
    
    default_config = get_default_dependency_config()
    categories = config.get("categories", default_config.get("categories", []))
    selected_packages = config.get("selected_packages", [])
    
    if not categories:
        return widgets.HTML(
            "<div style='padding: 10px; border-radius: 4px; color: #dc3545; background: #fff5f5;'>"
            "⚠️ Tidak ada kategori paket yang tersedia"
            "</div>"
        )
    
    def handle_package_select(package_key: str, is_checked: bool) -> None:
        if on_package_select:
            on_package_select(package_key, is_checked)
    
    # Create category sections
    category_sections = [
        _create_category_section(cat, selected_packages, handle_package_select)
        for cat in categories
    ]
    
    # Create a grid layout with max 3 columns
    grid_children = []
    row = []
    
    for i, section in enumerate(category_sections):
        row.append(section)
        if len(row) == 3 or i == len(category_sections) - 1:
            grid_children.append(
                widgets.HBox(
                    row,
                    layout=widgets.Layout(
                        width='100%',
                        justify_content='space-between',
                        margin='0 0 10px 0'
                    )
                )
            )
            row = []
    
    return widgets.VBox([
        widgets.HTML("<h3 style='margin: 0 0 12px 0;'>Kategori Paket</h3>"),
        widgets.VBox(grid_children, layout=widgets.Layout(width='100%'))
    ])

def _create_category_section(
    category: Dict[str, Any],
    selected_packages: List[str],
    on_package_select: Callable[[str, bool], None]
) -> widgets.Widget:
    """Create a single category section with its packages.
    
    Args:
        category: Dictionary containing category details
        selected_packages: List of selected package keys
        on_package_select: Callback function when a package is selected
        
    Returns:
        ipywidgets.Widget: A card widget containing the category's packages
    """
    if not category.get('packages'):
        return widgets.HTML(
            f"<p style='padding: 8px; color: #666;'>"
            f"Tidak ada paket dalam kategori {category.get('name', 'ini')}"
            "</p>"
        )
    
    package_items = [
        _create_package_item(
            pkg,
            is_checked=pkg['key'] in selected_packages,
            on_select=on_package_select
        )
        for pkg in category.get('packages', [])
    ]
    
    # Create a compact container for the category content
    category_content = widgets.VBox(
        package_items,
        layout=widgets.Layout(
            padding='4px 6px',
            width='100%',
            margin='2px 0',
            overflow_y='auto',
            max_height='200px',
            min_height='40px'
        )
    )
    
    # Create a compact card with grid layout
    card = widgets.VBox(
        [
            # Compact category header
            widgets.HTML(
                f"""
                <div style='
                    margin: 0 0 6px 0;
                    padding: 0;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                '>
                    <span style='font-size: 1em;'>{category.get('icon', '📦')}</span>
                    <h4 style='margin: 0; font-size: 0.95em; color: #2c3e50;'>{category.get('name', 'Kategori')}</h4>
                </div>
                """
            ),
            # Package items with scrolling
            category_content,
            # Compact description tooltip
            widgets.HTML(
                f"""
                <div style='
                    color: #888;
                    font-size: 0.75em;
                    margin-top: 4px;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    width: 100%;
                ' 
                title='{category.get('description', '')}'>
                    {category.get('description', '')}
                </div>
                """
            )
        ],
        layout=widgets.Layout(
            margin='0 8px 8px 0',
            width='32%',
            min_width='250px',
            border='1px solid #e8e8e8',
            border_radius='6px',
            padding='10px',
            background='#ffffff',
            box_shadow='0 1px 3px rgba(0,0,0,0.05)',
            _hover='box-shadow: 0 2px 5px rgba(0,0,0,0.1);',
            transition='all 0.15s ease-in-out',
            flex='1 1 30%',
            max_width='32%',
            min_height='200px',
            height='auto',
            overflow='hidden'
        )
    )
    
    card.add_class('category-card')
    
    return card

def _create_package_item(
    package: Dict[str, Any],
    is_checked: bool,
    on_select: Optional[Callable[[str, bool], None]] = None
) -> widgets.Widget:
    """Create a single package item with checkbox and description
    
    Args:
        package: Dictionary containing package details
        is_checked: Whether the package is selected
        on_select: Callback function when selection changes
        
    Returns:
        Widget containing the package item with enhanced styling
    """
    # Create a compact container for the package item
    package_container = widgets.HBox(
        layout=widgets.Layout(
            width='100%',
            padding='2px 4px',
            margin='1px 0',
            border_radius='3px',
            border='1px solid transparent',
            _hover='background-color: #f8f9fa;',
            transition='all 0.15s ease',
            align_items='center',
            height='28px',
            overflow='hidden'
        )
    )
    
    # Add custom class for styling
    package_container.add_class('package-item')
    if is_checked:
        package_container.add_style('background-color: #f0f7ff; border-color: #cce5ff;')
    
    # Create the checkbox with better styling
    checkbox = create_package_checkbox(
        package_name=package.get('name', ''),
        version=package.get('version', ''),
        is_installed=is_checked,
        on_change=lambda name, checked: on_select(package['key'], checked) if on_select else None
    )
    
    if on_select:
        def on_change(change):
            on_select(package['key'], change['new'])
        checkbox.observe(on_change, names='value')
    
    return widgets.HBox(
        [checkbox],
        layout=widgets.Layout(
            margin='5px 0',
            padding='5px 10px',
            border='1px solid #e0e0e0',
            border_radius='4px',
            width='100%',
            align_items='center'
        )
    )

# ======================== CUSTOM PACKAGES ========================

def create_custom_packages_section() -> widgets.Widget:
    """Create custom packages input section with package management controls.
    
    This function creates a UI section that allows users to:
    - Input package names with version specifications
    - Add/remove packages from a list
    - View currently added packages
    
    Returns:
        ipywidgets.VBox: A container widget with all the package management controls
    """
    # Create input field for package names with validation
    package_input = widgets.Text(
        placeholder='contoh: numpy>=1.20.0, pandas<2.0.0',
        description='Paket:',
        layout=widgets.Layout(width='100%'),
        style={'description_width': 'initial'}
    )
    
    # Create add button
    add_button = widgets.Button(
        description='Tambah',
        button_style='primary',
        icon='plus',
        layout=widgets.Layout(width='100px')
    )
    
    # Create package list
    package_list = widgets.VBox([], layout=widgets.Layout(width='100%'))
    
    # Create input row
    input_row = widgets.HBox(
        [package_input, add_button],
        layout=widgets.Layout(width='100%', justify_content='space-between')
    )
    
    # Create section
    section = widgets.VBox([
        widgets.HTML('<h3>Paket Kustom</h3>'),
        widgets.HTML('<p>Tambahkan paket Python dengan format: <code>nama-paket[spesifikasi-versi]</code></p>'),
        input_row,
        package_list
    ], layout=widgets.Layout(width='100%'))
    
    def on_add_package(button: widgets.Button) -> None:
        """Handle adding a new package to the list.
        
        Args:
            button: The button that triggered this event
        """
        package_spec = package_input.value.strip()
        if package_spec:
            # Create package item with remove button
            remove_btn = widgets.Button(
                description='',
                button_style='danger',
                layout=widgets.Layout(width='40px')
            )
            
            def on_remove(button: widgets.Button) -> None:
                """Remove the package from the list."""
                package_list.children = tuple(
                    child for child in package_list.children
                    if child != item
                )
            
            remove_btn.on_click(on_remove)
            
            item = widgets.HBox([
                widgets.Label(package_spec, layout=widgets.Layout(flex='1')),
                remove_btn
            ], layout=widgets.Layout(
                width='100%',
                justify_content='space-between',
                align_items='center',
                margin='2px 0',
                padding='4px',
                border='1px solid #e0e0e0',
                border_radius='4px'
            ))
            
            package_list.children += (item,)
            package_input.value = ''
    
    add_button.on_click(on_add_package)
    
    return section
