# =============================================================================
# File: smartcash/ui/setup/dependency/components/ui_components.py - FIXED
# Deskripsi: Main UI components dengan logger_bridge pattern yang lengkap
# =============================================================================

import ipywidgets as widgets
from typing import Dict, Any, List, Optional
from smartcash.ui.components import (
    create_header, create_responsive_container, create_action_buttons,
    create_dual_progress_tracker, create_log_accordion, create_status_panel,
    create_text_input, create_checkbox, create_card, create_confirmation_area,
    create_action_section
)
from smartcash.ui.utils.logger_bridge import UILoggerBridge

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create dependency management UI dengan logger_bridge pattern"""
    config = config or {}
    
    # Header
    header = create_header("Dependency Management", "Kelola instalasi package SmartCash", "🔧")
    
    # Status panel
    status_panel = create_status_panel("🚀 Siap mengelola dependencies", "info")
    
    # Package sections
    categories_section = _create_categories_section(config)
    custom_section = _create_custom_package_section(config)
    
    # Action buttons
    action_components = create_action_buttons([
        ("install_btn", "📦 Install", "primary", False),
        ("check_updates_btn", "🔍 Check Updates", "info", False), 
        ("uninstall_btn", "🗑️ Uninstall", "danger", False)
    ])
    
    # Progress tracker
    progress_tracker = create_dual_progress_tracker("Overall Progress", "Current Operation")
    
    # Log accordion
    log_components = create_log_accordion("Operation Logs", height="300px")
    
    # Confirmation area untuk dialogs
    confirmation_area = create_confirmation_area()[0]
    
    # Action section dengan shared component
    action_section = create_action_section(
        action_buttons=action_components,  # action_components is already a container widget
        confirmation_area=confirmation_area,
        title="🚀 Dependency Operations",
        status_label="📋 Operation Status:",
        show_status=True
    )
    
    # Main container
    main_container = create_responsive_container([
        header, status_panel, categories_section, custom_section,
        action_section, progress_tracker.container, log_components['log_accordion']
    ])
    
    # Return components - logger_bridge will be initialized by CommonInitializer
    ui_components = {
        'ui': main_container, 
        'container': main_container, 
        'header': header,
        'status_panel': status_panel, 
        'categories_section': categories_section,
        'custom_section': custom_section, 
        'action_section': action_section,
        'progress_tracker': progress_tracker, 
        'log_components': log_components['log_accordion'],
        'log_output': log_components['log_accordion']  # Alias
    }
    
    # Add action buttons if action_components is a widget with children
    if hasattr(action_components, 'children') and len(action_components.children) >= 3:
        ui_components.update({
            'install_btn': action_components.children[0],
            'check_updates_btn': action_components.children[1],
            'uninstall_btn': action_components.children[2]
        })
    # If action_components is a dict, add its contents
    elif isinstance(action_components, dict):
        ui_components.update(action_components)
    
    # Add extracted components
    ui_components.update(_extract_category_components(categories_section))
    ui_components.update(_extract_custom_components(custom_section))
    
    return ui_components

def _create_categories_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create package categories section"""
    from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config
    default_config = get_default_dependency_config()
    categories = config.get('categories', default_config.get('categories', []))
    selected_packages = config.get('selected_packages', [])
    
    if not categories:
        return widgets.HTML("⚠️ Tidak ada kategori package yang tersedia")
    
    category_widgets = []
    for category in categories:
        header = widgets.HTML(f"""
        <div style='margin: 15px 0 10px 0; padding: 10px; background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px;'>
            <h4 style='margin: 0; color: #495057;'>{category.get('icon', '📦')} {category.get('name', 'Unknown')}</h4>
            <p style='margin: 5px 0 0 0; color: #6c757d; font-size: 0.9em;'>{category.get('description', 'No description')}</p>
        </div>
        """)
        
        packages = category.get('packages', [])
        package_widgets = [_create_package_checkbox(pkg, selected_packages) for pkg in packages]
        
        if package_widgets:
            category_widgets.append(widgets.VBox([
                header, 
                widgets.VBox(package_widgets, layout=widgets.Layout(padding='0 0 0 20px'))
            ]))
    
    return widgets.VBox(category_widgets) if category_widgets else widgets.HTML("⚠️ Tidak ada package yang tersedia")

def _create_custom_package_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create custom package input section"""
    custom_packages = config.get('custom_packages', '')
    
    header = widgets.HTML("""
    <div style='margin: 20px 0 10px 0; padding: 10px; background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 8px;'>
        <h4 style='margin: 0; color: #1976d2;'>⚙️ Custom Packages</h4>
        <p style='margin: 5px 0 0 0; color: #424242; font-size: 0.9em;'>Tambahkan package custom (pisahkan dengan koma)</p>
    </div>
    """)
    
    custom_input = create_text_input(
        "custom_packages_input", 
        "Custom packages (misal: scikit-learn==1.3.0, matplotlib)", 
        custom_packages,
        multiline=True
    )
    
    add_button = widgets.Button(
        description="➕ Add Custom",
        button_style='info',
        layout=widgets.Layout(width='150px')
    )
    
    custom_list = widgets.HTML(value="<div style='margin-top: 10px;'></div>")
    
    return widgets.VBox([header, custom_input, add_button, custom_list])

def _create_package_checkbox(package: Dict[str, Any], selected_packages: List[str]) -> widgets.Widget:
    """Create checkbox untuk single package"""
    is_selected = package.get('key', '') in selected_packages
    is_required = package.get('required', False)
    
    checkbox = create_checkbox(
        name=f"pkg_{package.get('key', '')}",
        value=is_selected,
        description=package.get('name', 'Unknown Package'),
        disabled=is_required
    )
    
    indicators = []
    if is_required: indicators.append("🔒 Required")
    if package.get('default', False): indicators.append("⭐ Default")
    if package.get('installed', False): indicators.append("✅ Installed")
    if package.get('update_available', False): indicators.append("🆙 Update Available")
    
    description = widgets.HTML(value=f"""
    <div style='margin-left: 25px; padding: 8px; background: #f8f9fa; border-radius: 4px;'>
        <p style='margin: 0 0 5px 0; color: #495057; font-size: 0.85em;'>{package.get('description', 'No description')}</p>
        <span style='color: #6c757d; font-family: monospace; font-size: 0.75em;'>{package.get('pip_name', package.get('key', ''))}</span>
        {' | ' + ' | '.join(indicators) if indicators else ''}
    </div>
    """)
    
    return widgets.VBox([checkbox, description])

def _extract_category_components(categories_section: widgets.Widget) -> Dict[str, Any]:
    """Extract package checkboxes dari categories section"""
    components = {}
    
    def extract_checkboxes(widget):
        if hasattr(widget, 'children'):
            for child in widget.children:
                extract_checkboxes(child)
        elif hasattr(widget, 'description') and widget.description:
            # Checkbox components
            if hasattr(widget, 'value') and isinstance(widget.value, bool):
                # Generate key dari description
                key = f"pkg_{widget.description.lower().replace(' ', '_')}"
                components[key] = widget
    
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