"""
File: smartcash/ui/dataset/downloader/components/ui_layout.py
Deskripsi: Fixed UI layout dengan form components yang muncul dan tanpa info section berlebihan
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import try_operation_safe
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color
from smartcash.ui.components.progress_tracking import create_progress_tracking_container

def create_downloader_ui(config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """Create downloader UI dengan form yang benar-benar muncul"""
    config = config or {}
    
    ui_components = {}
    
    # 1. Header dengan namespace color
    ui_components['header'] = _create_themed_header()
    
    # 2. Status panel
    ui_components['status_panel'] = _create_status_panel(env)
    
    # 3. Form fields dan form container (FIXED - pastikan muncul)
    form_components = _create_form_components_fixed(config)
    ui_components.update(form_components)
    
    # 4. Save/Reset buttons
    save_reset_components = _create_save_reset_components()
    ui_components.update(save_reset_components)
    
    # 5. Confirmation area
    ui_components['confirmation_area'] = _create_confirmation_area()
    
    # 6. Action buttons
    action_components = _create_action_components()
    ui_components.update(action_components)
    
    # 7. Progress tracker
    progress_components = _create_progress_components()
    ui_components.update(progress_components)
    
    # 8. Log output
    log_components = _create_log_components()
    ui_components.update(log_components)
    
    # Main layout dengan fixed order
    ui_components['ui'] = _create_fixed_main_layout(ui_components)
    ui_components['main_container'] = ui_components['ui']
    
    return ui_components

def _create_themed_header() -> widgets.HTML:
    """Create header dengan namespace color"""
    namespace_color = try_operation_safe(lambda: get_namespace_color('DOWNLOAD'), '#96CEB4')
    return widgets.HTML(f"""
    <div style="background: linear-gradient(135deg, {namespace_color}, {namespace_color}CC); 
               padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px;
               box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="margin: 0; color: white; font-weight: 600;">📥 Dataset Downloader</h3>
        <p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">
            Download dan organize dataset untuk SmartCash training
        </p>
    </div>
    """)

def _create_status_panel(env=None) -> widgets.HTML:
    """Create status panel dengan environment info"""
    if env and getattr(env, 'is_drive_mounted', False):
        status_html = """<div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 4px; margin-bottom: 15px;">
           <span style="color: #2e7d32;">✅ Google Drive terhubung - Dataset akan tersimpan permanen</span></div>"""
    elif env and getattr(env, 'is_colab', False):
        status_html = """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px;">
           <span style="color: #856404;">⚠️ Drive tidak terhubung - Dataset akan hilang saat restart</span></div>"""
    else:
        status_html = """<div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin-bottom: 15px;">
           <span style="color: #1976d2;">📊 Status: Ready untuk download dataset</span></div>"""
    
    return widgets.HTML(status_html)

def _create_form_components_fixed(config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create form components yang PASTI muncul - no fallback complexity"""
    roboflow = config.get('roboflow', {})
    
    # FIXED: Langsung buat form fields tanpa try-except berlebihan
    form_fields = {
        'workspace_input': widgets.Text(
            value=roboflow.get('workspace', 'smartcash-wo2us'), 
            description='Workspace:', 
            placeholder='Nama workspace Roboflow',
            layout=widgets.Layout(width='100%'), 
            style={'description_width': '90px'}
        ),
        'project_input': widgets.Text(
            value=roboflow.get('project', 'rupiah-emisi-2022'),
            description='Project:', 
            placeholder='Nama project Roboflow', 
            layout=widgets.Layout(width='100%'), 
            style={'description_width': '90px'}
        ),
        'version_input': widgets.Text(
            value=str(roboflow.get('version', '3')), 
            description='Version:', 
            placeholder='Versi dataset',
            layout=widgets.Layout(width='100%'), 
            style={'description_width': '90px'}
        ),
        'api_key_input': widgets.Password(
            value=roboflow.get('api_key', ''), 
            description='API Key:', 
            placeholder='Masukkan API Key Roboflow',
            layout=widgets.Layout(width='100%'), 
            style={'description_width': '90px'}
        ),
        'validate_checkbox': widgets.Checkbox(
            value=True, 
            description='Validasi download',
            layout=widgets.Layout(width='100%')
        ),
        'organize_checkbox': widgets.Checkbox(
            value=True, 
            description='Organisir dataset',
            layout=widgets.Layout(width='100%')
        ),
        'backup_checkbox': widgets.Checkbox(
            value=False, 
            description='Backup existing',
            layout=widgets.Layout(width='100%')
        )
    }
    
    # Create form container
    form_fields['form_container'] = _create_two_column_form_layout(form_fields)
    return form_fields

def _create_two_column_form_layout(form_fields: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create two column form layout yang sederhana"""
    # Left column - Dataset info
    left_widgets = [
        form_fields['workspace_input'],
        form_fields['project_input'], 
        form_fields['version_input'],
        form_fields['api_key_input']
    ]
    left_column = widgets.VBox(left_widgets, layout=widgets.Layout(width='48%', margin='0', padding='8px'))
    
    # Right column - Options dengan format info
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px;">
        <small style="color: #1976d2;"><strong>Format:</strong> YOLOv5 PyTorch (hardcoded)</small>
    </div>
    """)
    
    right_widgets = [
        format_info,
        form_fields['validate_checkbox'],
        form_fields['organize_checkbox'], 
        form_fields['backup_checkbox']
    ]
    right_column = widgets.VBox(right_widgets, layout=widgets.Layout(width='48%', margin='0', padding='8px'))
    
    # Two column container
    two_column_container = widgets.HBox(
        [left_column, right_column], 
        layout=widgets.Layout(
            width='100%', 
            justify_content='space-between',
            border='1px solid #ddd', 
            border_radius='5px', 
            padding='15px', 
            margin='0 0 15px 0'
        )
    )
    
    return widgets.VBox([two_column_container], layout=widgets.Layout(width='100%', margin='0', padding='0'))

def _create_save_reset_components() -> Dict[str, widgets.Widget]:
    """Create save/reset components"""
    save_button = widgets.Button(
        description='Simpan', 
        button_style='secondary', 
        tooltip='Simpan konfigurasi',
        layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px')
    )
    reset_button = widgets.Button(
        description='Reset', 
        button_style='', 
        tooltip='Reset ke default',
        layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px')
    )
    
    save_reset_container = widgets.HBox(
        [save_button, reset_button], 
        layout=widgets.Layout(width='100%', justify_content='flex-end', margin='10px 0')
    )
    
    return {
        'save_button': save_button, 
        'reset_button': reset_button, 
        'save_reset_container': save_reset_container
    }

def _create_confirmation_area() -> widgets.Output:
    """Create confirmation area"""
    return widgets.Output(
        layout=widgets.Layout(
            width='100%', 
            max_height='400px', 
            overflow='auto', 
            display='none', 
            margin='10px 0'
        )
    )

def _create_action_components() -> Dict[str, widgets.Widget]:
    """Create action components"""
    button_layout = widgets.Layout(width='auto', min_width='150px', height='35px', margin='5px')
    
    buttons = {
        'download_button': widgets.Button(
            description='📥 Download', 
            button_style='primary', 
            tooltip='Download dataset dari Roboflow', 
            layout=button_layout
        ),
        'check_button': widgets.Button(
            description='🔍 Check', 
            button_style='info',
            tooltip='Check dataset status', 
            layout=button_layout
        ),
        'cleanup_button': widgets.Button(
            description='🧹 Cleanup', 
            button_style='danger',
            tooltip='Hapus dataset files', 
            layout=button_layout
        )
    }
    
    action_container = widgets.HBox(
        list(buttons.values()), 
        layout=widgets.Layout(
            width='100%', 
            justify_content='flex-start',
            margin='15px 0', 
            flex_wrap='wrap'
        )
    )
    
    buttons['action_container'] = action_container
    return buttons

def _create_progress_components() -> Dict[str, widgets.Widget]:
    """Create progress components"""
    progress_tracker_components = try_operation_safe(
        lambda: create_progress_tracking_container(),
        fallback_value={'tracker': widgets.HTML("📊 Progress tracker akan muncul saat operasi")}
    )
    
    # Progress container
    progress_container = widgets.VBox(
        [progress_tracker_components.get('container', widgets.HTML(""))],
        layout=widgets.Layout(width='100%', margin='15px 0')
    )
    
    progress_tracker_components['progress_container'] = progress_container
    return progress_tracker_components

def _create_log_components() -> Dict[str, widgets.Widget]:
    """Create log components"""
    log_output = widgets.Output(
        layout=widgets.Layout(
            width='100%', 
            max_height='300px', 
            border='1px solid #ddd',
            border_radius='4px', 
            padding='8px', 
            overflow='auto'
        )
    )
    
    log_accordion = widgets.Accordion([log_output], layout=widgets.Layout(width='100%', margin='10px 0'))
    log_accordion.set_title(0, '📋 Download Logs')
    log_accordion.selected_index = None
    
    return {'log_output': log_output, 'log_accordion': log_accordion}

def _create_fixed_main_layout(ui_components: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create main layout dengan FIXED ORDER"""
    # Fixed order components
    main_components = [
        ui_components.get('header'),                    # Header
        ui_components.get('status_panel'),              # Status panel  
        ui_components.get('form_container'),            # 1. Form 2 kolom
        ui_components.get('save_reset_container'),      # 2. Save Reset Button
        ui_components.get('confirmation_area'),         # 3. Area Konfirmasi
        ui_components.get('action_container'),          # 4. Action Buttons
        ui_components.get('progress_container'),        # 5. Progress Tracker
        ui_components.get('log_accordion')              # 6. Log Output
    ]
    
    # Filter None components
    valid_components = [comp for comp in main_components if comp is not None]
    
    return widgets.VBox(
        valid_components, 
        layout=widgets.Layout(
            width='100%', 
            max_width='100%', 
            padding='0', 
            margin='0', 
            overflow='hidden'
        )
    )