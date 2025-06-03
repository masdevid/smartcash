"""
File: smartcash/ui/setup/dependency_installer/components/ui_components.py
Deskripsi: UI components dependency installer yang terintegrasi tanpa duplikasi
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from .package_selector import create_package_selector_grid

def create_dependency_installer_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create dependency installer UI dengan integrated components"""
    
    get_icon = lambda key, fallback="📦": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # Header
    header = create_header(
        f"{get_icon('download', '📦')} Dependency Installer", 
        "Setup packages yang diperlukan untuk SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel("Pilih packages yang akan diinstall dan klik tombol install", "info")
    
    # Package selector grid
    package_selector = create_package_selector_grid(config)
    
    # Custom packages input
    custom_packages = widgets.Textarea(
        placeholder='Package tambahan (satu per baris)',
        layout=widgets.Layout(width='100%', height='80px', margin='5px 0')
    )
    
    custom_section = widgets.VBox([
        widgets.HTML(f"<h4 style='color: {get_color('dark', '#333')}; margin: 8px 0;'>{get_icon('edit', '📝')} Custom Packages</h4>"),
        custom_packages
    ], layout=widgets.Layout(width='100%', margin='8px 0'))
    
    # Action buttons
    action_buttons = create_action_buttons(
        primary_label="Install Packages",
        primary_icon="download",
        secondary_buttons=[("Analyze Packages", "search", "info"), ("Check Status", "check", "")],
        cleanup_enabled=False,
        button_width='140px'
    )
    
    # Auto-analyze checkbox untuk control
    auto_analyze_checkbox = widgets.Checkbox(
        value=True,
        description="Auto-analyze setelah render",
        tooltip="Otomatis analisis packages setelah UI dimuat",
        layout=widgets.Layout(width='auto', margin='5px 0')
    )
    
    # Save & reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan", reset_label="Reset",
        save_tooltip="Simpan konfigurasi packages",
        reset_tooltip="Reset pilihan ke default"
    )
    
    # Log accordion
    log_components = create_log_accordion(module_name='dependency', height='250px')
    
    # Progress tracking
    progress_components = create_progress_tracking_container()
    
    # Help panel
    help_content = """
    <div style="padding: 8px; background: #ffffff;">
        <p style="margin: 6px 0; font-size: 13px;">Installer akan menganalisis dan menginstall packages yang dibutuhkan untuk SmartCash.</p>
        <div style="margin: 8px 0;">
            <strong style="color: #495057; font-size: 13px;">Kategori Package:</strong>
            <ul style="margin: 4px 0; padding-left: 18px; color: #495057; font-size: 12px;">
                <li><strong>Core Requirements:</strong> Package inti untuk SmartCash</li>
                <li><strong>ML/AI Libraries:</strong> PyTorch, YOLO, computer vision</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy, image processing</li>
            </ul>
        </div>
        <div style="margin-top: 8px; padding: 6px; background: #e7f3ff; border-radius: 3px; font-size: 12px;">
            <strong>💡 Tips:</strong> Package yang sudah terinstall akan dilewati secara otomatis.
        </div>
    </div>
    """
    
    help_panel = widgets.Accordion([widgets.HTML(value=help_content)])
    help_panel.set_title(0, "💡 Info Installation")
    help_panel.selected_index = None
    
    # Section headers
    packages_header = widgets.HTML(f"""
        <h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 8px 0; font-size: 16px;'>
            {get_icon('config', '⚙️')} Pilih Packages
        </h4>
    """)
    
    action_header = widgets.HTML(f"""
        <h4 style='color: {get_color('dark', '#333')}; margin: 12px 0 8px 0; font-size: 16px;'>
            {get_icon('play', '▶️')} Installation Actions
        </h4>
    """)
    
    # Main UI assembly
    ui = widgets.VBox([
        header, status_panel, packages_header, package_selector['container'],
        custom_section, auto_analyze_checkbox, save_reset_buttons['container'],
        create_divider(), action_header, action_buttons['container'],
        progress_components['container'], log_components['log_accordion'], 
        create_divider(), help_panel
    ], layout=widgets.Layout(width='100%', padding='8px', overflow='hidden'))
    
    # Compile components
    ui_components = {
        # Main UI
        'ui': ui, 'header': header, 'status_panel': status_panel,
        
        # Package selector
        'package_selector': package_selector,
        'custom_packages': custom_packages,
        'auto_analyze_checkbox': auto_analyze_checkbox,
        
        # Action buttons - mapping to expected names
        'action_buttons': action_buttons,
        'install_button': action_buttons['download_button'],
        'analyze_button': action_buttons['check_button'],
        'check_button': action_buttons.get('cleanup_button') or action_buttons['check_button'],
        
        # Save/reset buttons
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Progress components
        'progress_components': progress_components,
        'progress_container': progress_components['container'],
        'show_for_operation': progress_components.get('show_for_operation'),
        'update_progress': progress_components.get('update_progress'),
        'complete_operation': progress_components.get('complete_operation'),
        'error_operation': progress_components.get('error_operation'),
        'reset_all': progress_components.get('reset_all'),
        
        # Log components
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],
        
        # UI info
        'help_panel': help_panel,
        'module_name': 'dependency_installer'
    }
    
    # Add package checkboxes dari selector
    if 'checkboxes' in package_selector:
        ui_components.update(package_selector['checkboxes'])
    
    # Validate critical components - create fallback if missing
    critical_buttons = ['install_button', 'analyze_button', 'check_button', 'save_button', 'reset_button']
    for comp_name in critical_buttons:
        if ui_components.get(comp_name) is None:
            ui_components[comp_name] = widgets.Button(
                description=comp_name.replace('_', ' ').title(),
                button_style='primary' if 'install' in comp_name else '',
                disabled=True,
                tooltip=f"Component {comp_name} tidak tersedia",
                layout=widgets.Layout(width='auto', max_width='150px')
            )
    
    return ui_components