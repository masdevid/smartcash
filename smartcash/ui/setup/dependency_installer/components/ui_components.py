"""
File: smartcash/ui/setup/dependency_installer/components/ui_components.py
Deskripsi: UI components dependency installer dengan improved spacing dan justify alignment
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
    """Create dependency installer UI dengan improved spacing dan alignment"""
    
    get_icon = lambda key, fallback="📦": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # Header
    header = create_header(
        f"{get_icon('download', '📦')} Dependency Installer", 
        "Setup packages yang diperlukan untuk SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel("Pilih packages yang akan diinstall dan klik tombol install", "info")
    
    # Package selector grid dengan improved spacing
    package_selector = create_package_selector_grid(config)
    
    # Custom packages input dengan better styling
    custom_packages = widgets.Textarea(
        placeholder='Package tambahan (satu per baris)\ncontoh: numpy>=1.21.0\nopencv-python>=4.5.0',
        layout=widgets.Layout(
            width='100%', 
            height='90px', 
            margin='8px 0',
            border='1px solid #ddd',
            border_radius='4px'
        )
    )
    
    custom_section = widgets.VBox([
        widgets.HTML(f"""
        <h4 style='color: {get_color('dark', '#333')}; margin: 12px 0 8px 0; font-size: 15px;'>
            {get_icon('edit', '📝')} Custom Packages
        </h4>
        """),
        custom_packages
    ], layout=widgets.Layout(width='100%', margin='10px 0'))
    
    # Action buttons dengan 3 buttons yang jelas
    action_buttons = create_action_buttons(
        primary_label="Install Packages",
        primary_icon="download", 
        secondary_buttons=[("Analyze Status", "search", "info")],
        cleanup_enabled=True,
        button_width='140px'
    )
    
    # Custom button untuk System Check (repurpose cleanup button)
    if action_buttons.get('cleanup_button'):
        action_buttons['cleanup_button'].description = "System Check"
        action_buttons['cleanup_button'].tooltip = "Check system compatibility and detailed package status"
        action_buttons['cleanup_button'].icon = 'info'
        action_buttons['cleanup_button'].button_style = ''
    
    # Auto-analyze checkbox dengan better spacing
    auto_analyze_checkbox = widgets.Checkbox(
        value=True,
        description="Auto-analyze setelah render",
        tooltip="Otomatis analisis packages setelah UI dimuat",
        layout=widgets.Layout(width='auto', margin='8px 0'),
        style={'description_width': 'initial'}
    )
    
    # Save & reset buttons dengan improved layout
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan", reset_label="Reset",
        save_tooltip="Simpan konfigurasi packages",
        reset_tooltip="Reset pilihan ke default",
        button_width='100px'
    )
    
    # Log accordion dengan better height
    log_components = create_log_accordion(module_name='dependency', height='280px')
    
    # Progress tracking
    progress_components = create_progress_tracking_container()
    
    # Help panel dengan penjelasan button functions
    help_content = """
    <div style="padding: 10px; background: #ffffff; line-height: 1.5;">
        <p style="margin: 8px 0; font-size: 13px; color: #495057;">
            Installer akan menganalisis dan menginstall packages yang dibutuhkan untuk SmartCash.
        </p>
        <div style="margin: 12px 0;">
            <strong style="color: #495057; font-size: 13px; display: block; margin-bottom: 6px;">Action Buttons:</strong>
            <ul style="margin: 6px 0; padding-left: 20px; color: #495057; font-size: 12px;">
                <li style="margin: 3px 0;"><strong>Install Packages:</strong> Install packages yang dipilih (skip yang sudah terinstall)</li>
                <li style="margin: 3px 0;"><strong>Analyze Status:</strong> Cek status instalasi packages yang dipilih saja</li>
                <li style="margin: 3px 0;"><strong>System Check:</strong> Comprehensive check sistem + semua package details</li>
            </ul>
        </div>
        <div style="margin: 12px 0;">
            <strong style="color: #495057; font-size: 13px; display: block; margin-bottom: 6px;">Package Categories:</strong>
            <ul style="margin: 6px 0; padding-left: 20px; color: #495057; font-size: 12px;">
                <li style="margin: 3px 0;"><strong>Core Requirements:</strong> Package inti untuk SmartCash (IPython, widgets)</li>
                <li style="margin: 3px 0;"><strong>ML/AI Libraries:</strong> PyTorch, YOLO, computer vision frameworks</li>
                <li style="margin: 3px 0;"><strong>Data Processing:</strong> Pandas, NumPy, OpenCV untuk manipulasi data</li>
            </ul>
        </div>
        <div style="margin-top: 12px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px; border-left: 3px solid #007bff;">
            <strong>💡 Tips:</strong> Package yang sudah terinstall akan dilewati secara otomatis. 
            Gunakan "Analyze Status" untuk cek package yang dipilih, atau "System Check" untuk detail lengkap.
        </div>
    </div>
    """
    
    help_panel = widgets.Accordion([widgets.HTML(value=help_content)])
    help_panel.set_title(0, "💡 Info Installation")
    help_panel.selected_index = None
    
    # Section headers dengan improved styling
    packages_header = widgets.HTML(f"""
    <h4 style='color: {get_color('dark', '#333')}; margin: 18px 0 12px 0; font-size: 16px; 
               border-bottom: 2px solid {get_color('primary', '#007bff')}; padding-bottom: 6px;'>
        {get_icon('config', '⚙️')} Pilih Packages untuk Installation
    </h4>
    """)
    
    action_header = widgets.HTML(f"""
    <h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 10px 0; font-size: 16px; 
               border-bottom: 2px solid {get_color('success', '#28a745')}; padding-bottom: 6px;'>
        {get_icon('play', '▶️')} Installation Actions
    </h4>
    """)
    
    # Main UI assembly dengan improved spacing dan layout
    ui = widgets.VBox([
        header, 
        status_panel, 
        packages_header, 
        package_selector['container'],
        custom_section, 
        widgets.HBox([auto_analyze_checkbox], layout=widgets.Layout(justify_content='flex-start', margin='5px 0')),
        save_reset_buttons['container'],
        create_divider(margin="20px 0", color="#e0e0e0"), 
        action_header, 
        widgets.HBox([action_buttons['container']], layout=widgets.Layout(justify_content='center', margin='10px 0')),
        progress_components['container'], 
        log_components['log_accordion'], 
        create_divider(margin="15px 0", color="#f0f0f0"), 
        help_panel
    ], layout=widgets.Layout(
        width='100%', 
        max_width='100%',
        padding='10px', 
        margin='0',
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    # Compile components
    ui_components = {
        # Main UI
        'ui': ui, 
        'header': header, 
        'status_panel': status_panel,
        
        # Package selector
        'package_selector': package_selector,
        'custom_packages': custom_packages,
        'auto_analyze_checkbox': auto_analyze_checkbox,
        
        # Action buttons - mapping dengan button labels yang jelas
        'action_buttons': action_buttons,
        'install_button': action_buttons['download_button'],        # Install Packages
        'analyze_button': action_buttons['check_button'],           # Analyze Status  
        'check_button': action_buttons.get('cleanup_button'),       # System Check
        
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