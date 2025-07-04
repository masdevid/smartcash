"""
File: smartcash/ui/setup/dependency/components/ui_components.py
Deskripsi: UI components tanpa check/uncheck buttons integration
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracker.factory import create_dual_progress_tracker
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from .package_selector import create_package_selector_grid

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create dependency installer UI tanpa check/uncheck buttons"""
    
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
    
    # Progress tracker dengan dual-level untuk tracking yang lebih jelas
    progress_tracker = create_dual_progress_tracker(
        operation="Dependency Installation",
        auto_hide=True
    )
    
    # Log components
    log_components = create_log_accordion(
        module_name="dependency",
        height="200px",
        width="100%"
    )
    
    # Action buttons
    action_buttons = create_action_buttons(
        primary_label="Install Packages",
        primary_icon="download", 
        secondary_buttons=[("Analyze Status", "search", "info")],
        cleanup_enabled=True,
        button_width='140px'
    )
    
    # Force update button labels
    action_buttons['download_button'].description = "Install Packages"
    action_buttons['download_button'].tooltip = "Install packages yang dipilih"
    action_buttons['download_button'].icon = 'download'
    action_buttons['download_button'].button_style = 'primary'
    
    action_buttons['check_button'].description = "Analyze Status"  
    action_buttons['check_button'].tooltip = "Analyze status packages yang dipilih"
    action_buttons['check_button'].icon = 'search'
    action_buttons['check_button'].button_style = 'info'
    
    # Repurpose cleanup button untuk System Report
    if action_buttons.get('cleanup_button'):
        action_buttons['cleanup_button'].description = "System Report"
        action_buttons['cleanup_button'].tooltip = "Generate comprehensive system and package report"
        action_buttons['cleanup_button'].icon = 'clipboard'
        action_buttons['cleanup_button'].button_style = ''
    
    # Auto-analyze checkbox
    auto_analyze_checkbox = widgets.Checkbox(
        value=config.get('auto_analyze', True) if config else True,
        description="Auto-analyze setelah render",
        tooltip="Otomatis analisis packages setelah UI dimuat",
        layout=widgets.Layout(width='auto', margin='8px 0'),
        style={'description_width': 'initial'}
    )
    
    # Save & reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan", reset_label="Reset",
        save_tooltip="Simpan konfigurasi packages",
        reset_tooltip="Reset pilihan ke default",
        button_width='100px'
    )
    
    # Log accordion (gunakan yang sudah dibuat sebelumnya)
    # Catatan: log_components sudah dibuat sebelumnya, tidak perlu dibuat lagi
    
    # Help content
    help_content = """
    <div style="padding: 10px; background: #ffffff; line-height: 1.5;">
        <p style="margin: 8px 0; font-size: 13px; color: #495057;">
            Dependency installer untuk setup packages yang dibutuhkan SmartCash secara otomatis.
        </p>
        <div style="margin: 12px 0;">
            <strong style="color: #495057; font-size: 13px; display: block; margin-bottom: 6px;">Action Buttons:</strong>
            <ul style="margin: 6px 0; padding-left: 20px; color: #495057; font-size: 12px;">
                <li style="margin: 3px 0;"><strong>Install Packages:</strong> Install packages yang dipilih (auto-skip yang sudah terinstall)</li>
                <li style="margin: 3px 0;"><strong>Analyze Status:</strong> Analisis status packages yang dipilih</li>
                <li style="margin: 3px 0;"><strong>System Report:</strong> Generate laporan lengkap sistem dan semua packages</li>
            </ul>
        </div>
        <div style="margin: 12px 0;">
            <strong style="color: #495057; font-size: 13px; display: block; margin-bottom: 6px;">Package Categories:</strong>
            <ul style="margin: 6px 0; padding-left: 20px; color: #495057; font-size: 12px;">
                <li style="margin: 3px 0;"><strong>Core Requirements:</strong> Package inti SmartCash (IPython, widgets, YAML)</li>
                <li style="margin: 3px 0;"><strong>ML/AI Libraries:</strong> PyTorch, YOLOv5, computer vision frameworks</li>
                <li style="margin: 3px 0;"><strong>Data Processing:</strong> Pandas, NumPy, OpenCV, Matplotlib</li>
            </ul>
        </div>
        <div style="margin-top: 12px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 12px; border-left: 3px solid #007bff;">
            <strong>💡 Tips:</strong> Package yang sudah terinstall akan dilewati otomatis saat install.
        </div>
    </div>
    """
    
    help_panel = widgets.Accordion([widgets.HTML(value=help_content)])
    help_panel.set_title(0, "💡 Info Installation")
    help_panel.selected_index = None
    
    # Section headers
    packages_header = widgets.HTML(f"""
    <h4 style='color: {get_color('dark', '#333')}; margin: 18px 0 8px 0; font-size: 16px; 
               border-bottom: 2px solid {get_color('primary', '#007bff')}; padding-bottom: 6px;'>
        {get_icon('config', '⚙️')} Pilih Packages untuk Installation
    </h4>
    <p style='color: {get_color('muted', '#666')}; margin: 5px 0 10px 0; font-size: 12px; font-style: italic;'>
        ⭐ = Essential packages (direkomendasikan)
    </p>
    """)
    
    action_header = widgets.HTML(f"""
    <h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 10px 0; font-size: 16px; 
               border-bottom: 2px solid {get_color('success', '#28a745')}; padding-bottom: 6px;'>
        {get_icon('play', '▶️')} Actions
    </h4>
    """)
    
    # Main UI assembly tanpa check/uncheck buttons
    ui = widgets.VBox([
        header, 
        status_panel, 
        packages_header,
        package_selector['container'],
        custom_section, 
        widgets.HBox([auto_analyze_checkbox], layout=widgets.Layout(justify_content='flex-start', margin='5px 0')),
        save_reset_buttons['container'],
        action_header, 
        widgets.HBox([action_buttons['container']], layout=widgets.Layout(justify_content='center', margin='10px 0')),
        progress_tracker.container, 
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
    
    # Compile components tanpa check/uncheck integration
    ui_components = {
        # Main UI
        'ui': ui, 
        'header': header, 
        'status_panel': status_panel,
        
        # Package selector
        'package_selector': package_selector,
        'custom_packages': custom_packages,
        'auto_analyze_checkbox': auto_analyze_checkbox,
        
        # Action buttons
        'action_buttons': action_buttons,
        'install_button': action_buttons['download_button'],        
        'analyze_button': action_buttons['check_button'],           
        'check_button': action_buttons.get('cleanup_button'),       
        
        # Save/reset buttons
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Progress components dengan API yang benar
        'progress_tracker': progress_tracker,
        'progress_container': progress_tracker.container,
        'show_for_operation': progress_tracker.show,
        'update_overall': progress_tracker.update_overall,  # API yang benar untuk level 1
        'update_current': progress_tracker.update_current,  # API yang benar untuk level 2
        'update_step_progress': progress_tracker.update_step_progress if hasattr(progress_tracker, 'update_step_progress') else None,  # API untuk level 3 dengan fallback
        'complete_operation': progress_tracker.complete,
        'error_operation': progress_tracker.error,
        'reset_all': progress_tracker.reset,
        
        # Backward compatibility untuk kode lama
        'update_progress': lambda type='overall', progress=0, message='', color=None: (
            progress_tracker.update_overall(progress, message, color) if type == 'overall' or type == 'level1' else
            progress_tracker.update_current(progress, message, color) if type == 'step' or type == 'level2' else
            progress_tracker.update_step_progress(progress, message, color) if hasattr(progress_tracker, 'update_step_progress') else
            progress_tracker.update_current(progress, message, color)
        ),
        
        # Log components
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],
        
        # UI info
        'help_panel': help_panel,
        'module_name': 'dependency'
    }
    
    # Add package checkboxes dari selector
    if 'checkboxes' in package_selector:
        ui_components.update(package_selector['checkboxes'])
    
    # Validate critical components
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