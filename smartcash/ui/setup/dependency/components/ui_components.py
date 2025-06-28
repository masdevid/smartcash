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
    config = config or {}
    ui_components = {}
    
    # === CORE COMPONENTS ===
    header = create_header(
        title="Manajemen Dependensi",
        description="Kelola dependensi Python untuk proyek SmartCash",
        icon="📦"
    )
    
    status_panel = create_status_panel(
        message="Siap mengelola dependensi",
        status_type="success"
    )
    
    # Create main sections
    categories = create_categories_section(config)
    custom_packages = create_custom_packages_section()
    
    # Create tab layout with consistent styling
    tabs = widgets.Tab(children=[categories, custom_packages])
    tabs.set_title(0, "📦 Paket")
    tabs.set_title(1, "➕ Kustom")
    tabs.selected_index = 0
    tabs.layout = widgets.Layout(width='100%', margin='10px 0')
    
    # Create action buttons with consistent styling and tooltips
    # Primary action button
    install_btn = widgets.Button(
        description='Install',
        icon='download',
        button_style='success',
        tooltip='Install selected packages',
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    # Secondary action buttons
    check_updates_btn = widgets.Button(
        description='Check Updates',
        icon='sync',
        button_style='info',
        tooltip='Check for package updates',
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    uninstall_btn = widgets.Button(
        description='Uninstall',
        icon='trash',
        button_style='danger',
        tooltip='Uninstall selected packages',
        layout=widgets.Layout(width='auto', margin='0 5px')
    )
    
    # Container for all action buttons
    action_components = {
        'install': install_btn,
        'check_updates': check_updates_btn,
        'uninstall': uninstall_btn
    }
    
    action_buttons = widgets.HBox(
        [install_btn, check_updates_btn, uninstall_btn],
        layout=widgets.Layout(
            margin='10px 0',
            justify_content='flex-start',
            width='100%',
            flex_flow='row wrap',
            align_items='center',
            gap='10px'
        )
    )
    
    # Create save/reset buttons with consistent styling
    save_reset_buttons = create_save_reset_buttons(
        save_label="💾 Simpan Konfigurasi",
        reset_label="🔄 Reset",
        button_width='auto',
        container_width='100%',
        save_tooltip="Simpan konfigurasi dependensi",
        reset_tooltip="Reset ke konfigurasi default"
    )
    
    # Use the container from save_reset_buttons
    save_reset = save_reset_buttons['container']
    
    # Update the container's layout
    save_reset.layout = widgets.Layout(
        margin='10px 0',
        justify_content='flex-end',
        width='100%'
    )
    
    # Create log accordion and get its components
    log_components = create_log_accordion(
        module_name="Dependency Management",
        height="300px"
    )
    log_accordion = log_components['log_accordion']
    
    # Main layout with consistent spacing
    layout = widgets.VBox(
        children=[
            header,
            status_panel,
            widgets.VBox([tabs], layout=widgets.Layout(width='100%', margin='10px 0')),
            action_buttons,
            save_reset,
            log_accordion
        ],
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            margin='10px 0'
        )
    )
    
    # Store components with required keys
    ui_components.update({
        'ui': layout,  # Main UI component
        'log_output': log_components.get('log_output'),
        'log_accordion': log_accordion,
        'header': header,
        'status_panel': status_panel,
        'tabs': tabs,
        'action_buttons': action_components,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset': save_reset,
        'status': status_panel.get('status', None),
        'confirmation_area': status_panel.get('confirmation_area', None)
    })
    
    return ui_components

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
    
    category_sections = [
        _create_category_section(cat, selected_packages, handle_package_select)
        for cat in categories
    ]
    
    return widgets.VBox([
        widgets.HTML("<h3>Kategori Paket</h3>"),
        widgets.VBox(category_sections, layout=widgets.Layout(width='100%'))
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
    
    # Create a container for the category content
    category_content = widgets.VBox(
        package_items,
        layout=widgets.Layout(
            padding='10px',
            width='100%',
            margin='5px 0'
        )
    )
    
    # Create a custom card using VBox with similar styling
    card = widgets.VBox(
        [
            widgets.HTML(
                f"<h4 style='margin: 0 0 10px 0; padding: 0;'>{category.get('icon', '')} {category.get('name', '')}</h4>"
            ),
            category_content,
            widgets.HTML(f"<div style='color: #666; font-size: 0.9em; margin-top: 10px;'>{category.get('description', '')}</div>")
        ],
        layout=widgets.Layout(
            margin='10px 0',
            width='100%',
            border='1px solid #e0e0e0',
            border_radius='6px',
            padding='15px',
            background='#ffffff'
        )
    )
    
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
        Widget containing the package item
    """
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
