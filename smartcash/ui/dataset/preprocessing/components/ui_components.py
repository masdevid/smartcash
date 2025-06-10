"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Enhanced UI assembly dengan progress tracker kompatibel dan optimized layout
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracker import create_triple_progress_tracker
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from .input_options import create_preprocessing_input_options

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Enhanced preprocessing UI dengan kompatible progress tracker dan optimized components"""
    
    get_icon = lambda key, fallback="⚙️": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    get_color = lambda key, fallback="#333": COLORS.get(key, fallback) if 'COLORS' in globals() else fallback
    
    # === HEADER & STATUS ===
    header = create_header(
        f"{get_icon('processing', '🔧')} Dataset Preprocessing", 
        "Enhanced preprocessing dengan validasi, multi-split, dan aspect ratio support"
    )
    
    status_panel = create_status_panel("🚀 Siap memulai preprocessing dataset dengan konfigurasi enhanced", "info")
    
    # === ENHANCED INPUT OPTIONS ===
    input_options = create_preprocessing_input_options(config)
    
    # === ACTION BUTTONS ===
    action_buttons = create_action_buttons(
        primary_label="🚀 Mulai Preprocessing",
        primary_icon="play",
        secondary_buttons=[("🔍 Check Dataset", "search", "info")],
        cleanup_enabled=True,
        button_width='140px'
    )
    
    # === CONFIRMATION AREA (FIXED PATTERN) ===
    confirmation_area = widgets.Output(layout=widgets.Layout(
        width='100%', 
        min_height='50px',
        max_height='200px', 
        margin='10px 0',
        padding='5px',
        border='1px solid #e0e0e0',
        border_radius='4px',
        overflow='auto',
        background_color='#fafafa'
    ))
    
    # === SAVE & RESET BUTTONS ===
    save_reset_buttons = create_save_reset_buttons(
        save_label="💾 Simpan Config", 
        reset_label="🔄 Reset",
        save_tooltip="Simpan konfigurasi preprocessing enhanced",
        reset_tooltip="Reset ke konfigurasi default enhanced",
        button_width='110px'
    )
    
    # === ENHANCED PROGRESS TRACKER (COMPATIBLE API) ===
    progress_components = create_triple_progress_tracker(auto_hide=False)
    progress_tracker = progress_components['tracker']
    
    # === LOG ACCORDION ===
    log_components = create_log_accordion(module_name='preprocessing', height='240px')
    
    # === ENHANCED HELP PANEL ===
    help_content = """
    <div style="padding:10px;background:#ffffff;border-radius:4px;">
        <div style="margin-bottom:8px;">
            <strong style="color:#495057;font-size:14px;">🚀 Enhanced Preprocessing Features:</strong>
        </div>
        
        <div style="margin:8px 0;">
            <strong style="color:#28a745;">🆕 Fitur Baru:</strong>
            <ul style="margin:4px 0;padding-left:18px;color:#495057;font-size:12px;">
                <li><strong>Multi-Split Support:</strong> Pilih kombinasi train/valid/test</li>
                <li><strong>Aspect Ratio:</strong> Pertahankan proporsi gambar asli</li>
                <li><strong>Validasi Lengkap:</strong> Move invalid files dengan custom lokasi</li>
                <li><strong>Batch Processing:</strong> Optimized dengan batch size configuration</li>
            </ul>
        </div>
        
        <div style="margin:8px 0;">
            <strong style="color:#007bff;">⚙️ Parameter Utama:</strong>
            <ul style="margin:4px 0;padding-left:18px;color:#495057;font-size:12px;">
                <li><strong>Resolusi:</strong> 320x320 hingga 832x832 (640x640 optimal)</li>
                <li><strong>Normalisasi:</strong> Min-Max, Standard, atau None</li>
                <li><strong>Target Splits:</strong> Multi-select untuk flexibility</li>
                <li><strong>Batch Size:</strong> 1-128 (32 default untuk memory optimal)</li>
            </ul>
        </div>
        
        <div style="margin-top:10px;padding:8px;background:#e7f3ff;border-radius:4px;">
            <strong style="color:#0056b3;">💡 Tips Enhanced:</strong>
            <div style="font-size:12px;color:#495057;margin-top:4px;">
                • Gunakan aspect ratio untuk preserve bentuk uang<br>
                • Multi-split memungkinkan preprocessing selective<br>
                • Validasi membantu identify dan isolate problematic files
            </div>
        </div>
    </div>
    """
    
    help_panel = widgets.Accordion([widgets.HTML(value=help_content)])
    help_panel.set_title(0, "💡 Enhanced Preprocessing Guide")
    help_panel.selected_index = None
    
    # === SECTION HEADERS ===
    config_header = widgets.HTML(f"""
        <h4 style='color:{get_color('dark', '#333')};margin:15px 0 8px 0;font-size:16px;
                   border-bottom:2px solid {get_color('primary', '#007bff')};padding-bottom:6px;'>
            {get_icon('settings', '⚙️')} Enhanced Configuration
        </h4>
    """)
    
    action_header = widgets.HTML(f"""
        <h4 style='color:{get_color('dark', '#333')};margin:15px 0 10px 0;font-size:16px;
                   border-bottom:2px solid {get_color('success', '#28a745')};padding-bottom:6px;'>
            {get_icon('play', '▶️')} Preprocessing Actions
        </h4>
    """)
    
    # === ACTION SECTION (FIXED PATTERN) ===
    action_section = widgets.VBox([
        action_header,
        action_buttons['container'],
        # Status & Konfirmasi section
        widgets.HTML("<div style='margin: 8px 0 5px 0; font-weight: bold; color: #495057;'>📋 Status & Konfirmasi:</div>"),
        confirmation_area
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        padding='12px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#f9f9f9'
    ))
    
    # === MAIN UI ASSEMBLY (FIXED ORDER) ===
    ui = widgets.VBox([
        header,
        status_panel,
        config_header,
        input_options,
        save_reset_buttons['container'],
        action_section,  # Action section with confirmation area
        progress_components['progress_container'],  # Progress after confirmation
        log_components['log_accordion'],  # Log accordion last
        create_divider(),
        help_panel
    ], layout=widgets.Layout(width='100%', padding='8px', overflow='hidden'))
    
    # === COMPILE ENHANCED COMPONENTS ===
    ui_components = {
        # Main UI
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        
        # Enhanced Input Components
        'input_options': input_options,
        'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
        'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None),
        'preserve_aspect_checkbox': getattr(input_options, 'preserve_aspect_checkbox', None),
        'target_splits_select': getattr(input_options, 'target_splits_select', None),
        'batch_size_input': getattr(input_options, 'batch_size_input', None),
        'validation_checkbox': getattr(input_options, 'validation_checkbox', None),
        'move_invalid_checkbox': getattr(input_options, 'move_invalid_checkbox', None),
        'invalid_dir_input': getattr(input_options, 'invalid_dir_input', None),
        
        # Action Buttons
        'action_buttons': action_buttons,
        'preprocess_button': action_buttons['download_button'],  # Alias untuk compatibility
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Confirmation Area
        'confirmation_area': confirmation_area,
        
        # Save/Reset Buttons
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Enhanced Progress Tracker (Compatible API)
        'progress_tracker': progress_tracker,
        'progress_container': progress_components['progress_container'],
        'status_widget': progress_components['status_widget'],
        
        # Progress Tracker Compatible Methods
        'show_for_operation': progress_components.get('show_for_operation', progress_tracker.show),
        'update_overall': progress_components.get('update_overall', progress_tracker.update_overall),
        'update_step': progress_components.get('update_step', progress_tracker.update_step),
        'update_current': progress_components.get('update_current', progress_tracker.update_current),
        'update_progress': progress_components.get('update_progress', progress_tracker.update),
        'complete_operation': progress_components.get('complete_operation', progress_tracker.complete),
        'error_operation': progress_components.get('error_operation', progress_tracker.error),
        'reset_all': progress_components.get('reset_all', progress_tracker.reset),
        
        # Log Components
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],  # Alias untuk compatibility
        
        # Additional Components
        'help_panel': help_panel,
        'module_name': 'preprocessing'
    }
    
    # === VALIDATE CRITICAL COMPONENTS ===
    critical_components = [
        'preprocess_button', 'check_button', 'save_button', 'reset_button',
        'resolution_dropdown', 'target_splits_select', 'batch_size_input'
    ]
    
    for comp_name in critical_components:
        if ui_components.get(comp_name) is None:
            # Create fallback component
            if 'button' in comp_name:
                ui_components[comp_name] = widgets.Button(
                    description=comp_name.replace('_', ' ').title(),
                    button_style='primary' if 'preprocess' in comp_name else '',
                    disabled=True,
                    tooltip=f"Component {comp_name} tidak tersedia",
                    layout=widgets.Layout(width='auto', max_width='150px')
                )
            elif 'dropdown' in comp_name:
                ui_components[comp_name] = widgets.Dropdown(
                    options=['default'],
                    value='default',
                    disabled=True,
                    layout=widgets.Layout(width='100%')
                )
            elif 'select' in comp_name:
                ui_components[comp_name] = widgets.SelectMultiple(
                    options=[('Default', 'default')],
                    value=('default',),
                    disabled=True,
                    layout=widgets.Layout(width='100%')
                )
            elif 'input' in comp_name:
                ui_components[comp_name] = widgets.BoundedIntText(
                    value=32,
                    disabled=True,
                    layout=widgets.Layout(width='100%')
                )
    
    return ui_components