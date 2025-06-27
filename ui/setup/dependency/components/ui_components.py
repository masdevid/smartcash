# =============================================================================
# File: smartcash/ui/setup/dependency/components/ui_components.py - FIXED
# Deskripsi: Main UI components dengan logger_bridge pattern yang lengkap
# =============================================================================

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, List, Optional
from smartcash.ui.components import (
    create_header, create_responsive_container, create_action_buttons,
    create_dual_progress_tracker, create_status_panel,
    create_text_input, create_checkbox, create_card, create_confirmation_area,
    create_action_section, create_summary_panel
)
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.utils.logger_bridge import UILoggerBridge

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create main UI components for dependency management with improved layout and organization"""
    if config is None:
        config = {}
    
    # Create header with icon and description
    header = create_header(
        title="📦 Dependency Management",
        description="Install, update, and manage Python packages",
        icon="🔌"
    )
    
    # Create status panel for system messages
    status_panel = create_status_panel(
        message="✅ Ready to manage dependencies",
        status_type="info"
    )
    
    # Create summary panel
    summary_panel = create_summary_panel()
    
    # Create progress tracker
    progress_tracker = create_dual_progress_tracker(
        total_steps=100,
        current_step=0,
        descriptions={
            'overall': 'Overall Progress',
            'current': 'Current Operation',
            'details': 'Preparing...'
        }
    )
    
    # Create collapsible categories section
    categories_section = _create_categories_section(config)
    
    # Create custom package section with save/reset buttons
    custom_section, custom_package_input, add_custom_btn, custom_packages_list = _create_custom_package_section_with_controls()
    
    # Create save status indicator
    save_status = widgets.HTML(
        value='<span style="color: #6c757d;">No changes to save</span>',
        layout=widgets.Layout(margin='10px 0 0 10px')
    )
    
    # Update save status when packages change
    def update_save_status():
        has_changes = True  # Implement actual change detection
        save_button = action_components.buttons.get('save_btn')
        if save_button:
            save_button.disabled = not has_changes
        save_status.value = (
            '<span style="color: #28a745;">Changes detected</span>'
            if has_changes else
            '<span style="color: #6c757d;">No changes to save</span>'
        )
    
    # Add change handlers
    if hasattr(custom_packages_list, 'observe'):
        custom_packages_list.observe(lambda _: update_save_status(), names='children')
    
    # Create action buttons with icons and tooltips
    action_buttons = create_action_buttons([
        {
            'id': 'install_btn',
            'label': '📦 Install',
            'style': 'success',
            'tooltip': 'Install selected packages',
            'icon': 'fa-download',
            'disabled': False
        },
        {
            'id': 'check_updates_btn',
            'label': '🔄 Check Updates',
            'style': 'info',
            'tooltip': 'Check for available updates',
            'icon': 'fa-sync',
            'disabled': False
        },
        {
            'id': 'uninstall_btn',
            'label': '🗑️ Uninstall',
            'style': 'danger',
            'tooltip': 'Uninstall selected packages',
            'icon': 'fa-trash',
            'disabled': False
        },
        {
            'id': 'save_btn',
            'label': '💾 Save',
            'style': 'primary',
            'tooltip': 'Save current configuration',
            'icon': 'fa-save',
            'disabled': False
        },
        {
            'id': 'reset_btn',
            'label': '🔄 Reset',
            'style': 'warning',
            'tooltip': 'Reset to saved configuration',
            'icon': 'fa-undo',
            'disabled': False
        }
    ])
    
    action_components = create_action_buttons(action_buttons)
    
    # Progress tracker
    progress_tracker = create_dual_progress_tracker("Overall Progress", "Current Operation")
    
    # Create log accordion for installation logs
    log_accordion = create_log_accordion(
        module_name="Dependency Management",
        height="300px",
        width="100%",
        show_timestamps=True,
        show_level_icons=True,
        auto_scroll=True
    )
    
    # Create a tabbed interface for better organization
    tabs = widgets.Tab()
    tabs.children = [
        widgets.VBox([
            widgets.HTML("<h3 style='margin-top: 0;'>Package Categories</h3>"),
            categories_section,
            widgets.HTML("<div style='margin: 10px 0;'></div>"),
            action_buttons
        ]),
        widgets.VBox([
            widgets.HTML("<h3 style='margin-top: 0;'>Custom Packages</h3>"),
            custom_section
        ])
    ]
    tabs.set_title(0, '📦 Packages')
    tabs.set_title(1, '➕ Custom')
    
    # Create main container with responsive layout
    main_container = widgets.VBox(
        [
            header,
            status_panel,
            tabs,
            widgets.HBox([
                action_components,
                save_status
            ], layout=widgets.Layout(
                justify_content='space-between',
                align_items='center',
                margin='15px 0'
            )),
            log_accordion
        ],
        layout=widgets.Layout(
            width='100%',
            max_width='1200px',
            margin='0 auto',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            box_shadow='0 2px 4px rgba(0,0,0,0.05)'
        )
    )
    
    # Extract log output widget from accordion
    log_output = None
    if hasattr(log_accordion, 'children') and len(log_accordion.children) > 1:
        if hasattr(log_accordion.children[1], 'children') and len(log_accordion.children[1].children) > 0:
            log_output = log_accordion.children[1].children[0]
    
    # Initialize UI components dictionary
    ui_components = {
        'ui': main_container,
        'header': header,
        'status_panel': status_panel,
        'tabs': tabs,
        'categories_section': categories_section,
        'custom_section': custom_section,
        'log_accordion': log_accordion,
        'log_output': log_output,
        'config': config,
        'action_buttons': action_buttons
    }
    
    # Map action buttons for easy access
    if hasattr(action_buttons, 'buttons') and isinstance(action_buttons.buttons, dict):
        ui_components.update({
            'install_btn': action_buttons.buttons.get('install_btn'),
            'check_updates_btn': action_buttons.buttons.get('check_updates_btn'),
            'uninstall_btn': action_buttons.buttons.get('uninstall_btn')
        })
    elif hasattr(action_buttons, 'children') and len(action_buttons.children) >= 3:
        ui_components.update({
            'install_btn': action_buttons.children[0],
            'check_updates_btn': action_buttons.children[1],
            'uninstall_btn': action_buttons.children[2]
        })
    
    # Extract package checkboxes from categories section
    ui_components.update(_extract_category_components(categories_section))
    
    # Add some global styles
    display(HTML("""
    <style>
        .widget-tab > .p-TabBar {
            margin-bottom: 15px;
        }
        .widget-tab > .p-TabBar .p-TabBar-tab {
            padding: 8px 16px;
            border: 1px solid #dee2e6;
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            margin-right: 4px;
            background: #f8f9fa;
        }
        .widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current {
            background: white;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
        }
        .widget-tab > .p-TabBar .p-TabBar-tab:hover:not(.p-mod-current) {
            background: #e9ecef;
        }
    </style>
    
    <script>
        // Add smooth scrolling for better UX
        document.addEventListener('DOMContentLoaded', function() {
            // Smooth scroll to log when new messages are added
            const observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes.length) {
                        const logOutput = document.querySelector('.output_subarea');
                        if (logOutput) {
                            logOutput.scrollTop = logOutput.scrollHeight;
                        }
                    }
                });
            });
            
            // Start observing the log output
            const logOutput = document.querySelector('.output_subarea');
            if (logOutput) {
                observer.observe(logOutput, { childList: true, subtree: true });
            }
        });
    </script>
    
    """))
    
    return ui_components

def _create_categories_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create package categories section with compact card groups"""
    from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config
    default_config = get_default_dependency_config()
    categories = config.get('categories', default_config.get('categories', []))
    selected_packages = config.get('selected_packages', [])
    
    if not categories:
        return widgets.HTML("⚠️ Tidak ada kategori package yang tersedia")
    
    category_widgets = []
    for category in categories:
        # Create collapsible section for each category
        category_name = category.get('name', 'Unknown')
        packages = category.get('packages', [])
        
        # Create checkboxes for packages in a grid
        package_checkboxes = []
        for pkg in packages:
            checkbox = _create_package_checkbox(pkg, selected_packages)
            package_checkboxes.append(checkbox)
        
        # Create a grid layout for checkboxes (2 columns)
        checkbox_grid = []
        for i in range(0, len(package_checkboxes), 2):
            row = widgets.HBox(
                package_checkboxes[i:i+2],
                layout=widgets.Layout(
                    width='100%',
                    justify_content='space-between',
                    margin='5px 0'
                )
            )
            checkbox_grid.append(row)
            
        # Create collapsible button
        collapse_btn = widgets.ToggleButton(
            value=False,
            description=f"{category.get('icon', '📦')} {category_name}",
            icon='caret-down',
            layout=widgets.Layout(
                width='auto',
                margin='0 0 5px 0',
                padding='5px 10px',
                font_weight='bold',
                border_radius='4px 4px 0 0',
                border='1px solid #dee2e6',
                background='#f8f9fa'
            )
        )
        
        # Create collapsible content
        content = widgets.VBox(
            checkbox_grid,
            layout=widgets.Layout(
                width='100%',
                padding='10px',
                border='1px solid #dee2e6',
                border_top='none',
                border_radius='0 0 4px 4px',
                display='none'  # Initially hidden
            )
        )
        
        # Add description if available
        if category.get('description'):
            desc_widget = widgets.HTML(
                f"<div style='color: #6c757d; font-size: 0.9em; margin: 0 0 10px 0;'>{category['description']}</div>"
            )
            content.children = (desc_widget,) + content.children
        
        # Toggle visibility function
        def toggle_content(change, content=content, btn=collapse_btn):
            if change['new']:  # Button is pressed
                content.layout.display = 'flex'
                btn.icon = 'caret-up'
            else:
                content.layout.display = 'none'
                btn.icon = 'caret-down'
        
        collapse_btn.observe(toggle_content, names='value')
        
        # Create category card
        category_card = widgets.VBox(
            [collapse_btn, content],
            layout=widgets.Layout(
                width='100%',
                margin='0 0 15px 0',
                padding='0',
                border_radius='4px',
                box_shadow='0 1px 3px rgba(0,0,0,0.1)'
            )
        )
        
        # Add click handler to header
        category_card.add_class('category-card')
        category_widgets.append(category_card)
    
    # Add some JavaScript for better interactivity
    display(HTML("""
    <style>
        .package-checkbox-container {
            transition: all 0.2s ease;
        }
        .package-checkbox-container:hover {
            background-color: #f0f7ff !important;
            border-color: #b8daff !important;
        }
        .package-checkbox-container.required {
            background-color: #fff8e6 !important;
            border-left: 3px solid #ffc107 !important;
        }
        .package-checkbox-container.installed {
            border-left: 3px solid #28a745 !important;
        }
        .package-checkbox-container.update-available {
            border-left: 3px solid #17a2b8 !important;
        }
        .category-card {
            transition: box-shadow 0.2s ease;
        }
        .category-card:hover {
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
    </style>
    
    <script>
        // Add hover effect to category cards
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.widget-vbox');
            cards.forEach(card => {
                card.addEventListener('mouseenter', () => {
                    card.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
                });
                card.addEventListener('mouseleave', () => {
                    card.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                });
            });
        });
    </script>
    """))
    
    return widgets.VBox(
        category_widgets,
        layout=widgets.Layout(
            width='100%',
            padding='10px',
            overflow_y='auto',
            max_height='70vh'
        )
    )

def _create_custom_package_section_with_controls() -> tuple:
    """Create custom package section with save/reset functionality"""
    # Create input field for package name
    package_input = widgets.Text(
        placeholder='Package name (e.g., numpy)',
        layout=widgets.Layout(width='40%', margin='0 10px 0 0')
    )
    
    # Create input field for version (optional)
    version_input = widgets.Text(
        placeholder='Version (optional, e.g., 1.21.0)',
        layout=widgets.Layout(width='30%', margin='0 10px 0 0')
    )
    
    # Create add button
    add_button = widgets.Button(
        description='Add',
        button_style='success',
        icon='plus',
        layout=widgets.Layout(width='100px')
    )
    
    # Create package list with cards
    package_list = widgets.VBox(
        [],
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            max_height='300px',
            overflow_y='auto',
            border='1px solid #e0e0e0',
            border_radius='4px',
            padding='5px'
        )
    )
    
    # Function to get all packages as a list of strings (name[==version])
    def get_packages():
        packages = []
        for child in package_list.children:
            if hasattr(child, 'children') and len(child.children) > 0:
                html_widget = child.children[0]
                if hasattr(html_widget, 'value'):
                    # Extract package name and version from the HTML
                    import re
                    match = re.search(r'>(.*?)<', html_widget.value)
                    if match:
                        packages.append(match.group(1))
        return packages
    
    # Function to add a package to the list
    def add_package_to_list(pkg_name: str, version: str = ''):
        if not pkg_name:
            return
            
        # Check if package already exists
        pkg_spec = f"{pkg_name}=={version}" if version else pkg_name
        existing_packages = get_packages()
        if pkg_spec in existing_packages:
            return
            
        # Create package card
        pkg_card = widgets.HBox(
            [
                widgets.HTML(
                    f"<div style='flex-grow: 1; padding: 8px;'>{pkg_name}{'==' + version if version else ''}</div>",
                    layout=widgets.Layout(overflow='hidden')
                ),
                widgets.Button(
                    icon='trash',
                    button_style='danger',
                    layout=widgets.Layout(width='40px', height='30px', margin='0 5px'),
                    tooltip='Remove package'
                )
            ],
            layout=widgets.Layout(
                width='100%',
                margin='2px 0',
                padding='5px',
                border='1px solid #e0e0e0',
                border_radius='4px',
                background='#f9f9f9'
            )
        )
        
        # Add remove button handler
        def remove_package(btn):
            package_list.children = tuple(p for p in package_list.children if p != pkg_card)
        
        pkg_card.children[1].on_click(remove_package)
        
        # Add to list
        package_list.children += (pkg_card,)
    
    # Handle add button click
    def on_add_clicked(btn):
        pkg_name = package_input.value.strip()
        version = version_input.value.strip()
        
        if pkg_name:
            add_package_to_list(pkg_name, version)
            package_input.value = ''
            version_input.value = ''
    
    add_button.on_click(on_add_clicked)
    
    # Create section container
    section = widgets.VBox([
        widgets.HTML("<h4 style='margin-top: 0;'>Custom Packages</h4>"),
        widgets.HBox([
            package_input,
            version_input,
            add_button
        ], layout=widgets.Layout(margin='0 0 10px 0')),
        package_list,
        widgets.HTML("<div style='margin: 5px 0; font-size: 0.9em; color: #666;'>"
                      "Add custom packages that are not in the predefined categories</div>")
    ], layout=widgets.Layout(width='100%'))
    
    # Add a method to get the current package list
    def get_package_list():
        return get_packages()
    
    # Add a method to clear the package list
    def clear_packages():
        package_list.children = ()
    
    # Add methods to the section object
    section.get_packages = get_package_list
    section.clear = clear_packages
    section.add_package = add_package_to_list
    
    return section, package_input, add_button, package_list

def _create_custom_package_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create custom package section with save/reset functionality"""
    # Get the section and components from the new implementation
    section, package_input, add_button, package_list = _create_custom_package_section_with_controls()
    
    # Define the add_package_to_list function
    def add_package_to_list(pkg_name: str, version: str = ''):
        if not pkg_name:
            return
            
        # Create package card
        pkg_card = widgets.HBox(
            [
                widgets.HTML(
                    f"<div style='flex-grow: 1; padding: 8px;'>{pkg_name}{'==' + version if version else ''}</div>",
                    layout=widgets.Layout(overflow='hidden')
                ),
                widgets.Button(
                    icon='trash',
                    button_style='danger',
                    layout=widgets.Layout(width='40px', height='30px', margin='0 5px'),
                    tooltip='Remove package'
                )
            ],
            layout=widgets.Layout(
                width='100%',
                margin='2px 0',
                padding='5px',
                border='1px solid #e0e0e0',
                border_radius='4px',
                background='#f9f9f9'
            )
        )
        
        # Add remove button handler
        def remove_package(btn):
            package_list.children = tuple(p for p in package_list.children if p != pkg_card)
        
        pkg_card.children[1].on_click(remove_package)
        
        # Add to list
        package_list.children += (pkg_card,)
    
    # Add any existing custom packages from config
    custom_packages = config.get('custom_packages', [])
    if isinstance(custom_packages, str):
        custom_packages = [pkg.strip() for pkg in custom_packages.split(',') if pkg.strip()]
    
    for pkg_spec in custom_packages:
        if '==' in pkg_spec:
            pkg_name, version = pkg_spec.split('==', 1)
            add_package_to_list(pkg_name, version)
        else:
            add_package_to_list(pkg_spec)
    
    return section

def _create_package_checkbox(package: Dict[str, Any], selected_packages: List[str]) -> widgets.HBox:
    """Create checkbox untuk single package with enhanced styling"""
    package_key = package.get('key', '')
    package_name = package.get('name', 'Unknown')
    package_version = package.get('version', '')
    package_desc = package.get('description', 'No description')
    package_icon = package.get('icon', '📦')
    
    # Collect status indicators
    indicators = []
    if package.get('required', False):
        indicators.append("🔒 Required")
    if package.get('default', False):
        indicators.append("⭐ Default")
    if package.get('installed', False):
        indicators.append("✅ Installed")
    if package.get('update_available', False):
        indicators.append("🆙 Update Available")

    # Create checkbox with custom styling
    checkbox = widgets.Checkbox(
        value=package_key in selected_packages,
        indent=False,
        disabled=package.get('required', False),
        layout=widgets.Layout(
            width='auto',
            margin='2px 5px 2px 0',
            padding='0 5px',
            align_self='flex-start'
        )
    )

    # Create label with package info
    label = widgets.HTML(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            font-size: 0.9em;
            margin-left: 5px;
            width: 100%;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 2px;">
                <span style="margin-right: 8px; font-size: 1.1em;">{package_icon}</span>
                <span style="font-weight: 500;">{package_name} <small style="color: #666;">{package_version}</small></span>
            </div>
            <div style="font-size: 0.85em; color: #666; margin-bottom: 4px; line-height: 1.3;">
                {package_desc}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.8em; color: #666;">
                <code style="background: #f0f0f0; padding: 1px 4px; border-radius: 3px;">
                    {package.get('pip_name', package_key)}
                </code>
                {' '.join(f'<span>{ind}</span>' for ind in indicators) if indicators else ''}
            </div>
        </div>
        """,
        layout=widgets.Layout(margin='0 0 0 5px')
    )

    # Create container for checkbox and label
    container = widgets.HBox(
        [checkbox, label],
        layout=widgets.Layout(
            width='100%',
            margin='4px 0',
            padding='8px',
            border='1px solid #e0e0e0',
            border_radius='6px',
            background='#f9f9f9',
            align_items='flex-start'
        )
    )

    # Add hover effect classes
    container.add_class('package-checkbox-container')
    checkbox.add_class('package-checkbox')

    # Store package key and checkbox reference for easy access
    container.package_key = package_key
    container.checkbox = checkbox
    container.package_name = package_name
    container.package_version = package_version

    return container

def _extract_category_components(categories_section: widgets.Widget) -> Dict[str, Any]:
    """Extract package checkboxes from categories section"""
    components = {}

    def extract_checkboxes(widget):
        if hasattr(widget, 'children'):
            for child in widget.children:
                extract_checkboxes(child)
        # Check for package checkboxes by looking for our custom attributes
        if hasattr(widget, 'package_key') and hasattr(widget, 'checkbox'):
            components[f"pkg_{widget.package_key}"] = widget.checkbox

    extract_checkboxes(categories_section)
    return components

def _extract_custom_components(custom_section: widgets.Widget) -> Dict[str, Any]:
    """Extract custom package components"""
    components = {}
    
    def extract_custom(widget):
        if hasattr(widget, 'children'):
            for child in widget.children:
                extract_custom(child)
        elif hasattr(widget, 'placeholder') and 'custom' in str(widget.placeholder).lower():
            components['custom_packages_input'] = widget
        elif hasattr(widget, 'description') and 'Add Custom' in str(widget.description):
            components['add_custom_btn'] = widget
        elif hasattr(widget, 'value') and isinstance(widget.value, str) and '<div' in widget.value:
            components['custom_packages_list'] = widget
    
    extract_custom(custom_section)
    return components

def update_package_status(ui_components: Dict[str, Any], package_key: str, status: str) -> None:
    """Update status package di UI"""
    status_messages = {
        'installing': f"⏳ Installing {package_key}...",
        'installed': f"✅ {package_key} berhasil diinstall",
        'failed': f"❌ {package_key} gagal diinstall",
        'checking': f"🔍 Checking {package_key}...",
        'updated': f"🆙 {package_key} berhasil diupdate"
    }
    
    message = status_messages.get(status, f"📦 {package_key}: {status}")
    logger_bridge = ui_components.get('logger_bridge')
    
    if logger_bridge:
        if status == 'failed':
            logger_bridge.error(message)
        elif status in ['installed', 'updated']:
            logger_bridge.success(message)
        else:
            logger_bridge.info(message)