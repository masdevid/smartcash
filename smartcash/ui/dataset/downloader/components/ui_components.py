"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: Enhanced UI components dengan form backup_dir/preprocessed_dir dan fixed persistence
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color
from smartcash.ui.dataset.downloader.utils.colab_secrets import get_api_key_from_secrets

def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI dengan enhanced form fields dan fixed persistence"""
    config = config or {}
    roboflow = config.get('roboflow', {})
    
    # Auto-detect API key saat UI creation
    detected_api_key = get_api_key_from_secrets()
    
    ui_components = _create_enhanced_downloader_ui(config, roboflow, detected_api_key)
    ui_components['layout_order_fixed'] = True
    
    return ui_components

def _create_enhanced_downloader_ui(config: Dict[str, Any], roboflow: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Create enhanced downloader UI dengan backup/preprocessed form fields - no overflow"""
    
    # 1. Header dengan responsive design
    header = widgets.HTML(f"""
    <div style="background: linear-gradient(135deg, {get_namespace_color('DOWNLOAD')}, {get_namespace_color('DOWNLOAD')}CC); 
                padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 100%; box-sizing: border-box;">
        <h3 style="margin: 0; color: white; font-weight: 600;">📥 Dataset Downloader</h3>
        <p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">Download dan organize dataset untuk SmartCash training</p>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # 2. Status panel dengan auto environment detection
    status_panel = widgets.HTML(_get_dynamic_status_html(), layout=widgets.Layout(width='100%', margin='0 0 15px 0'))
    
    # 3. Enhanced form fields dengan backup/preprocessed dirs
    form_fields = _create_enhanced_form_fields(roboflow, api_key, config)
    
    # 4. Enhanced form container dengan path configuration
    form_container = _create_enhanced_grid_form_container(form_fields)
    
    # 5. Save/Reset buttons
    save_reset_components = _create_fixed_save_reset_buttons()
    
    # 6. Confirmation area (hidden by default)
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', max_height='400px', overflow='auto', display='none', margin='10px 0'))
    
    # 7. Action buttons dengan state management
    action_components = _create_state_managed_action_buttons()
    
    # 8. Enhanced progress tracking container
    progress_components = create_progress_tracking_container()
    progress_components['container'].layout.display = 'none'
    
    # 9. Log output - accordion terbuka by default
    log_components = _create_open_log_accordion()
    
    # 10. Main container dengan CSS Flexbox - no overflow
    main_ui = widgets.VBox([
        header,
        status_panel,
        form_container,
        save_reset_components['container'],
        confirmation_area,
        action_components['container'],
        progress_components['container'],
        log_components['accordion']
    ], layout=widgets.Layout(
        width='100%', 
        max_width='100%', 
        padding='0', 
        margin='0',
        display='flex',
        flex_flow='column nowrap',
        align_items='stretch',
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    return {**{
        'ui': main_ui, 'main_container': main_ui, 'header': header, 'status_panel': status_panel,
        'form_container': form_container, 'confirmation_area': confirmation_area
    }, **form_fields, **save_reset_components, **action_components, **progress_components, **log_components}

def _get_dynamic_status_html() -> str:
    """Get dynamic status HTML dengan environment detection - one-liner conditionals"""
    try:
        import google.colab
        from pathlib import Path
        is_drive_mounted = Path('/content/drive/MyDrive').exists()
        api_key = get_api_key_from_secrets()
        
        # One-liner status generation dengan nested conditionals
        return ("""<div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #2e7d32;">✅ Drive terhubung + API Key terdeteksi - Siap download!</span></div>""" if (is_drive_mounted and api_key) else
                """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #856404;">⚠️ Drive terhubung - Masukkan API Key untuk mulai</span></div>""" if is_drive_mounted else
                """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #856404;">⚠️ API Key tersedia - Mount Drive untuk penyimpanan permanen</span></div>""" if api_key else
                """<div style="padding: 12px; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #721c24;">❌ Perlu mount Drive dan setup API Key</span></div>""")
    except ImportError:
        return """<div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #1976d2;">📊 Status: Local environment - Ready</span></div>"""

def _create_enhanced_form_fields(roboflow: Dict[str, Any], api_key: str, config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create enhanced form fields dengan backup/preprocessed directories dan one-liner layout"""
    common_layout = widgets.Layout(width='100%', margin='2px 0')
    common_style = {'description_width': '100px'}
    
    # Get paths dari config atau defaults dengan one-liner fallback
    backup_dir = config.get('backup_dir', config.get('paths', {}).get('backup', 'data/backup'))
    preprocessed_dir = config.get('preprocessed_dir', config.get('paths', {}).get('preprocessed', 'data/preprocessed'))
    
    return {
        # Dataset configuration
        'workspace_input': widgets.Text(
            value=roboflow.get('workspace', 'smartcash-wo2us'), 
            description='Workspace:', 
            placeholder='Nama workspace Roboflow', 
            layout=common_layout, 
            style=common_style
        ),
        'project_input': widgets.Text(
            value=roboflow.get('project', 'rupiah-emisi-2022'), 
            description='Project:', 
            placeholder='Nama project Roboflow', 
            layout=common_layout, 
            style=common_style
        ),
        'version_input': widgets.Text(
            value=str(roboflow.get('version', '3')), 
            description='Version:', 
            placeholder='Versi dataset', 
            layout=common_layout, 
            style=common_style
        ),
        'api_key_input': widgets.Password(
            value=api_key or roboflow.get('api_key', ''), 
            description='API Key:', 
            placeholder='🔑 Auto-detect dari Colab secrets' if api_key else 'Masukkan API Key Roboflow', 
            layout=common_layout, 
            style=common_style
        ),
        
        # Path configuration - NEW FIELDS
        'backup_dir_input': widgets.Text(
            value=backup_dir,
            description='Backup Dir:', 
            placeholder='Direktori untuk backup dataset', 
            layout=common_layout, 
            style=common_style
        ),
        'preprocessed_dir_input': widgets.Text(
            value=preprocessed_dir,
            description='Preproc Dir:', 
            placeholder='Direktori untuk preprocessed data', 
            layout=common_layout, 
            style=common_style
        ),
        
        # Options
        'validate_checkbox': widgets.Checkbox(
            value=config.get('validate_download', True), 
            description='Validasi download', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        'backup_checkbox': widgets.Checkbox(
            value=config.get('backup_existing', False), 
            description='Backup existing', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        )
    }

def _create_enhanced_grid_form_container(form_fields: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create enhanced form container dengan 3-column layout untuk path config"""
    
    # Format info yang selalu ditampilkan
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px; width: 100%; box-sizing: border-box;">
        <small style="color: #1976d2;"><strong>📦 Format:</strong> YOLOv5 PyTorch (hardcoded)</small>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # Left column - dataset config dengan one-liner VBox
    left_column = widgets.VBox([
        form_fields['workspace_input'],
        form_fields['project_input'], 
        form_fields['version_input'],
        form_fields['api_key_input']
    ], layout=widgets.Layout(width='32%', padding='8px', box_sizing='border-box', flex='1 1 32%'))
    
    # Middle column - path config dengan one-liner VBox  
    middle_column = widgets.VBox([
        widgets.HTML("""<div style="padding: 8px; background: #fff3cd; border-radius: 4px; margin-bottom: 8px; width: 100%; box-sizing: border-box;"><small style="color: #856404;"><strong>📂 Path Config:</strong> Direktori penyimpanan</small></div>""", layout=widgets.Layout(width='100%', margin='0')),
        form_fields['backup_dir_input'],
        form_fields['preprocessed_dir_input'],
        widgets.HTML("""<div style="height: 20px;"></div>""")  # Spacer
    ], layout=widgets.Layout(width='32%', padding='8px', box_sizing='border-box', flex='1 1 32%'))
    
    # Right column - options dengan one-liner VBox
    right_column = widgets.VBox([
        format_info,
        form_fields['validate_checkbox'],
        form_fields['backup_checkbox'],
        widgets.HTML("""<div style="height: 40px;"></div>""")  # Spacer untuk alignment
    ], layout=widgets.Layout(width='32%', padding='8px', box_sizing='border-box', flex='1 1 32%'))
    
    # Form row dengan responsive 3-column flexbox
    form_row = widgets.HBox([left_column, middle_column, right_column], layout=widgets.Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='flex-start',
        border='1px solid #ddd',
        border_radius='5px',
        padding='15px',
        margin='0 0 15px 0',
        box_sizing='border-box',
        overflow='hidden'
    ))
    
    return widgets.VBox([form_row], layout=widgets.Layout(width='100%', margin='0'))

def _create_fixed_save_reset_buttons() -> Dict[str, widgets.Widget]:
    """Create save/reset buttons - save secondary, tanpa icon"""
    save_button = widgets.Button(
        description='Simpan',  # Tanpa icon
        button_style='secondary',  # Secondary/default style (grey)
        layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px')
    )
    
    reset_button = widgets.Button(
        description='Reset',  # Tanpa icon
        button_style='',  # Default style
        layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px')
    )
    
    container = widgets.HBox([save_button, reset_button], layout=widgets.Layout(
        width='100%',
        justify_content='flex-end',
        margin='10px 0',
        display='flex',
        flex_flow='row nowrap',
        align_items='center'
    ))
    
    return {'save_button': save_button, 'reset_button': reset_button, 'container': container}

def _create_state_managed_action_buttons() -> Dict[str, widgets.Widget]:
    """Create action buttons dengan state management untuk mutual exclusion"""
    button_layout = widgets.Layout(
        width='auto', 
        min_width='140px', 
        height='35px', 
        margin='5px',
        flex='0 1 auto'
    )
    
    download_button = widgets.Button(description='📥 Download', button_style='primary', layout=button_layout)
    check_button = widgets.Button(description='🔍 Check', button_style='info', layout=button_layout)
    cleanup_button = widgets.Button(description='🧹 Cleanup', button_style='danger', layout=button_layout)
    
    # Add state management attributes dengan one-liner setattr
    [setattr(btn, '_all_buttons', [download_button, check_button, cleanup_button]) 
     for btn in [download_button, check_button, cleanup_button]]
    
    container = widgets.HBox([download_button, check_button, cleanup_button], layout=widgets.Layout(
        width='100%',
        justify_content='flex-start',
        margin='15px 0',
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        overflow='hidden'
    ))
    
    return {'download_button': download_button, 'check_button': check_button, 'cleanup_button': cleanup_button, 'container': container}

def _create_open_log_accordion() -> Dict[str, widgets.Widget]:
    """Create log accordion yang terbuka by default"""
    log_output = widgets.Output(layout=widgets.Layout(
        width='100%', 
        max_height='300px', 
        border='1px solid #ddd', 
        border_radius='4px', 
        padding='8px', 
        overflow='auto',
        box_sizing='border-box'
    ))
    
    log_accordion = widgets.Accordion([log_output], layout=widgets.Layout(
        width='100%', 
        margin='10px 0',
        box_sizing='border-box'
    ))
    
    # Set accordion terbuka dan title dengan one-liner chaining
    log_accordion.set_title(0, '📋 Download Logs') and setattr(log_accordion, 'selected_index', 0)
    
    return {'log_output': log_output, 'log_accordion': log_accordion, 'accordion': log_accordion}

# Enhanced utilities dengan path support
def detect_api_key() -> str:
    """Detect API key dengan one-liner fallback"""
    return get_api_key_from_secrets() or ''

def validate_ui_layout(ui: Dict[str, Any]) -> bool:
    """Validate UI layout dengan enhanced field checking"""
    required_fields = ['ui', 'form_container', 'save_button', 'reset_button', 'download_button', 'log_output', 
                      'backup_dir_input', 'preprocessed_dir_input']
    return all(key in ui for key in required_fields)

def get_enhanced_ui_status(ui: Dict[str, Any]) -> str:
    """Get enhanced UI status dengan path config info"""
    component_count = len([k for k in ui.keys() if not k.startswith('_')])
    api_status = '✅' if detect_api_key() else '❌'
    path_fields = '✅' if all(field in ui for field in ['backup_dir_input', 'preprocessed_dir_input']) else '❌'
    return f"✅ Enhanced UI Ready: {component_count} components | API Key: {api_status} | Path Config: {path_fields}"

def disable_other_buttons(active_button: widgets.Button) -> None:
    """Disable other buttons saat satu action berjalan - one-liner state management"""
    hasattr(active_button, '_all_buttons') and [setattr(btn, 'disabled', True) for btn in active_button._all_buttons if btn != active_button]

def enable_all_buttons(button_list: list) -> None:
    """Enable all buttons setelah operation selesai - one-liner restore"""
    [setattr(btn, 'disabled', False) for btn in button_list if hasattr(btn, 'disabled')]