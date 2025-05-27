"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: UI components preprocessing yang terintegrasi tanpa duplikasi utils
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
from .input_options import create_preprocessing_input_options

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create preprocessing UI dengan integrated components"""
    
    get_icon = lambda key, fallback="⚙️": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # Header
    header = create_header(
        f"{get_icon('processing', '🔧')} Dataset Preprocessing", 
        "Preprocessing dataset untuk training model SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel("Siap memulai preprocessing dataset", "info")
    
    # Input options
    input_options = create_preprocessing_input_options(config)
    
    # Action buttons
    action_buttons = create_action_buttons(
        primary_label="Mulai Preprocessing",
        primary_icon="play",
        secondary_buttons=[("Check Dataset", "search", "info")],
        cleanup_enabled=True,
        button_width='130px'
    )
    
    # Confirmation area
    confirmation_area = widgets.Output(
        layout=widgets.Layout(width='100%', height='auto', margin='8px 0', overflow='visible')
    )
    
    # Save & reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan", reset_label="Reset",
        save_tooltip="Simpan konfigurasi preprocessing",
        reset_tooltip="Reset konfigurasi ke default"
    )
    
    # Log accordion
    log_components = create_log_accordion(module_name='preprocessing', height='220px')
    
    # Progress tracking
    progress_components = create_progress_tracking_container()
    
    # Help panel
    help_content = """
    <div style="padding: 8px; background: #ffffff;">
        <p style="margin: 6px 0; font-size: 13px;">Preprocessing mengubah ukuran gambar dan normalisasi pixel untuk training optimal.</p>
        <div style="margin: 8px 0;">
            <strong style="color: #495057; font-size: 13px;">Parameter Utama:</strong>
            <ul style="margin: 4px 0; padding-left: 18px; color: #495057; font-size: 12px;">
                <li><strong>Resolusi:</strong> Ukuran output (640x640 direkomendasikan)</li>
                <li><strong>Normalisasi:</strong> Metode normalisasi pixel (minmax)</li>
                <li><strong>Workers:</strong> Thread paralel (4-8 optimal)</li>
                <li><strong>Split:</strong> Bagian dataset (all untuk semua)</li>
            </ul>
        </div>
        <div style="margin-top: 8px; padding: 6px; background: #e7f3ff; border-radius: 3px; font-size: 12px;">
            <strong>💡 Tips:</strong> Gunakan "Check Dataset" untuk validasi sebelum preprocessing.
        </div>
    </div>
    """
    
    help_panel = widgets.Accordion([widgets.HTML(value=help_content)])
    help_panel.set_title(0, "💡 Info Preprocessing")
    help_panel.selected_index = None
    
    # Section headers
    config_header = widgets.HTML(f"""
        <h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 8px 0; font-size: 16px;'>
            {get_icon('settings', '⚙️')} Konfigurasi Preprocessing
        </h4>
    """)
    
    action_header = widgets.HTML(f"""
        <h4 style='color: {get_color('dark', '#333')}; margin: 12px 0 8px 0; font-size: 16px;'>
            {get_icon('play', '▶️')} Aksi Preprocessing
        </h4>
    """)
    
    # Main UI assembly
    ui = widgets.VBox([
        header, status_panel, config_header, input_options, save_reset_buttons['container'],
        create_divider(), action_header, action_buttons['container'], confirmation_area,
        progress_components['container'], log_components['log_accordion'], 
        create_divider(), help_panel
    ], layout=widgets.Layout(width='100%', padding='8px', overflow='hidden'))
    
    # Compile components
    ui_components = {
        # Main UI
        'ui': ui, 'header': header, 'status_panel': status_panel,
        
        # Input components
        'input_options': input_options,
        'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
        'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None), 
        'worker_slider': getattr(input_options, 'worker_slider', None),
        'split_dropdown': getattr(input_options, 'split_dropdown', None),
        
        # Action buttons
        'action_buttons': action_buttons,
        'preprocess_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Confirmation area
        'confirmation_area': confirmation_area,
        
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
        'module_name': 'preprocessing'
    }
    
    # Validate critical components - create fallback buttons if missing
    critical_components = ['preprocess_button', 'check_button', 'save_button', 'reset_button']
    for comp_name in critical_components:
        if ui_components.get(comp_name) is None:
            ui_components[comp_name] = widgets.Button(
                description=comp_name.replace('_', ' ').title(),
                button_style='primary' if 'preprocess' in comp_name else '',
                disabled=True,
                tooltip=f"Component {comp_name} tidak tersedia",
                layout=widgets.Layout(width='auto', max_width='150px')
            )
    
    return ui_components