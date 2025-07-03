"""
File: smartcash/ui/setup/dependency/components/ui_components.py
Module: smartcash.ui.setup.dependency.components.ui_components
Description: Consolidated UI components for dependency management in SmartCash

This module provides UI components for managing Python package dependencies
in the SmartCash project, including package selection, custom package input,
and dependency resolution interfaces.

Updated to use modular container components for improved maintainability and consistency.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import ipywidgets as widgets
from enum import Enum

from smartcash.ui.components import (
    create_main_container,
    create_header_container,
    create_form_container,
    create_summary_container,
    create_action_container,
    create_footer_container
)

# Import atomic UI sections
from smartcash.ui.setup.dependency.components import (
    create_categories_section,
    create_custom_packages_section,
    DependencySummaryPanel
)

def create_package_checkbox(
    package_name: str,
    version: str = "",
    is_installed: bool = False,
    on_change: Optional[Callable[[str, bool], None]] = None,
    **style
) -> widgets.HBox:
    """Create a styled package checkbox with version and status indicator.
    
    Args:
        package_name: Name of the package
        version: Package version (optional)
        is_installed: Whether the package is installed
        on_change: Callback when checkbox state changes
        **style: Additional style parameters
        
    Returns:
        widgets.HBox containing the package checkbox with version and status
    """
    # Create the main checkbox
    checkbox = widgets.Checkbox(
        value=is_installed,
        indent=False,
        layout=widgets.Layout(width='20px', margin='0 8px 0 0')
    )
    
    # Create version label
    version_text = f" ({version})" if version else ""
    version_label = widgets.HTML(
        f"<span style='color: #666; font-size: 0.9em;'>{version_text}</span>",
        layout=widgets.Layout(margin='0 8px 0 0')
    )
    
    # Create status indicator
    status_emoji = "✅" if is_installed else "❌"
    status_tooltip = "Terinstall" if is_installed else "Belum terinstall"
    status = widgets.HTML(
        f"<span title='{status_tooltip}'>{status_emoji}</span>",
        layout=widgets.Layout(margin='0 8px 0 0', width='20px')
    )
    
    # Package name label
    name_label = widgets.HTML(
        f"<span style='font-family: monospace;'>{package_name}</span>"
    )
    
    # Create the container
    container = widgets.HBox(
        [checkbox, name_label, version_label, status],
        layout=widgets.Layout(
            margin='4px 0',
            padding='4px 8px',
            border_radius='4px',
            border='1px solid #e0e0e0',
            width='100%',
            **style.get('container', {})
        )
    )
    
    # Add hover effect
    container.add_class('package-checkbox-container')
    
    # Handle checkbox changes
    def on_checkbox_change(change):
        if on_change:
            on_change(package_name, change['new'])
    
    checkbox.observe(on_checkbox_change, names='value')
    
    return container

# ======================== MAIN UI ========================

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create main dependency management UI with consistent structure using modular container components
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing UI components and state
    """
    config = config or {}
    ui_components = {}
    
    # === HEADER CONTAINER ===
    header_container = create_header_container(
        title="📦 Manajemen Dependensi",
        subtitle="Kelola dependensi Python untuk proyek SmartCash",
        icon="settings",
        status_message="Siap untuk mengelola dependensi",
        status_type="info"
    )
    
    # === FORM CONTAINER ===
    # Create summary panel for selected packages
    summary_panel = DependencySummaryPanel()
    
    # Create categories section with package selection using the atomic component
    categories_section = create_categories_section(
        config=config,
        on_package_select=lambda package_key, is_checked: summary_panel.update_selection(package_key, is_checked)
    )
    
    # Create custom packages section for manual package entry using the atomic component
    custom_packages_section = create_custom_packages_section(
        initial_packages=config.get('custom_packages', []),
        on_packages_change=lambda packages: summary_panel.update_custom_packages(packages)
    )
    
    # Create tabs to organize categories and custom packages
    tabs = widgets.Tab()
    tabs.children = [categories_section, custom_packages_section]
    tabs.set_title(0, "📦 Paket Tersedia")
    tabs.set_title(1, "➕ Paket Kustom")
    
    # Create form container with save/reset buttons
    form_container = create_form_container(
        save_label="Simpan Konfigurasi",
        reset_label="Reset",
        alignment="right",
        show_buttons=True
    )
    
    # Add content to form container
    form_container['form_container'].children = (
        summary_panel.container,
        widgets.HTML("<hr style='margin: 20px 0; border: none; border-top: 1px solid #e0e0e0;'>"),
        tabs
    )
    
    # === ACTION CONTAINER ===
    action_container = create_action_container(
        title="🚀 Operasi Dependensi",
        buttons=[
            {
                'button_id': 'install',
                'text': '🔄 Install Dependensi',
                'style': 'primary',
                'tooltip': 'Install semua dependensi yang dipilih'
            },
            {
                'button_id': 'check',
                'text': '✓ Periksa Dependensi',
                'style': 'info',
                'tooltip': 'Periksa status instalasi dependensi'
            },
            {
                'button_id': 'uninstall',
                'text': '🗑️ Hapus Dependensi',
                'style': 'danger',
                'tooltip': 'Hapus dependensi yang dipilih'
            }
        ]
    )
    
    # === FOOTER CONTAINER ===
    footer_container = create_footer_container(
        show_progress=True,
        show_logs=True,
        show_info=True,
        show_tips=True,
        log_module_name="Dependency Manager",
        info_title="Informasi Dependensi",
        info_content="<p>Kelola dependensi Python untuk proyek SmartCash. Pilih paket yang diperlukan dan klik 'Install Dependensi' untuk menginstalnya.</p>",
        tips=[
            "Pilih paket yang diperlukan dari tab 'Paket Tersedia'",
            "Tambahkan paket kustom dengan spesifikasi versi di tab 'Paket Kustom'",
            "Gunakan tombol 'Periksa Dependensi' untuk melihat status instalasi"
        ]
    )
    
    # === SUMMARY CONTAINER ===
    # Create summary container for operation results
    summary_container = create_summary_container(
        theme="info",
        title="Status Operasi",
        icon="📊"
    )
    
    # Set initial content
    summary_container.show_message(
        "Siap Digunakan", 
        "Pilih kategori dan paket yang ingin diinstall, lalu klik tombol 'Install Dependensi'.",
        "info"
    )
    
    # === MAIN CONTAINER ===
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=form_container['container'],
        summary_container=summary_container.container,  # Add summary container above action container
        action_container=action_container['container'],
        footer_container=footer_container.container
    )
    
    # Store components in the result dictionary
    ui_components.update({
        'container': main_container.container,
        'main_container': main_container,
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'footer_container': footer_container,
        'summary_container': summary_container,
        'summary_panel': summary_panel,
        'categories_section': categories_section,
        'custom_packages_section': custom_packages_section,
        'tabs': tabs,
        'save_button': form_container['save_button'],
        'reset_button': form_container['reset_button'],
        'install_button': action_container['buttons']['install'],
        'check_button': action_container['buttons']['check'],
        'uninstall_button': action_container['buttons']['uninstall'],
        'config': config
    })
    
    # Set up default event handlers for buttons
    ui_components['save_button'].on_click(lambda b: header_container.update_status("Konfigurasi disimpan", "success"))
    ui_components['reset_button'].on_click(lambda b: header_container.update_status("Konfigurasi direset", "info"))
    
    # Operation buttons will be handled by operation_handlers.py
    # These are just temporary handlers that will be replaced when setup_all_handlers is called
    ui_components['install_button'].on_click(lambda b: None)
    ui_components['check_button'].on_click(lambda b: None)
    ui_components['uninstall_button'].on_click(lambda b: None)
    
    return ui_components


# Handler functions have been moved to the handlers directory
# See smartcash.ui.setup.dependency.handlers.operation_handlers
