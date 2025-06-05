"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: Fixed UI components dengan proper responsive layout dan API key auto-detection
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color
from smartcash.ui.dataset.downloader.utils.colab_secrets import get_api_key_from_secrets

def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI dengan responsive layout dan auto API key detection"""
    config = config or {}
    roboflow = config.get('roboflow', {})
    
    # Auto-detect API key saat UI creation
    detected_api_key = get_api_key_from_secrets()
    
    ui_components = _create_responsive_downloader_ui(config, roboflow, detected_api_key)
    ui_components['layout_order_fixed'] = True
    
    return ui_components

def _create_responsive_downloader_ui(config: Dict[str, Any], roboflow: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Create responsive downloader UI dengan flexbox layout - no overflow"""
    
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
    
    # 3. Form fields dengan auto API key
    form_fields = _create_responsive_form_fields(roboflow, api_key)
    
    # 4. Form container dengan CSS Grid layout
    form_container = _create_grid_form_container(form_fields)
    
    # 5. Save/Reset buttons tanpa icon, save secondary style
    save_reset_components = _create_fixed_save_reset_buttons()
    
    # 6. Confirmation area (hidden by default)
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', max_height='400px', overflow='auto', display='none', margin='10px 0'))
    
    # 7. Action buttons dengan responsive layout
    action_components = _create_responsive_action_buttons()
    
    # 8. Progress tracking (hidden by default)
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
    """Get dynamic status HTML dengan environment detection"""
    try:
        import google.colab
        from pathlib import Path
        is_drive_mounted = Path('/content/drive/MyDrive').exists()
        api_key = get_api_key_from_secrets()
        
        if is_drive_mounted and api_key:
            return """<div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #2e7d32;">✅ Drive terhubung + API Key terdeteksi - Siap download!</span></div>"""
        elif is_drive_mounted:
            return """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #856404;">⚠️ Drive terhubung - Masukkan API Key untuk mulai</span></div>"""
        elif api_key:
            return """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #856404;">⚠️ API Key tersedia - Mount Drive untuk penyimpanan permanen</span></div>"""
        else:
            return """<div style="padding: 12px; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #721c24;">❌ Perlu mount Drive dan setup API Key</span></div>"""
    except ImportError:
        return """<div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin-bottom: 15px; width: 100%; box-sizing: border-box;"><span style="color: #1976d2;">📊 Status: Local environment - Ready</span></div>"""

def _create_responsive_form_fields(roboflow: Dict[str, Any], api_key: str) -> Dict[str, widgets.Widget]:
    """Create responsive form fields dengan proper sizing"""
    common_layout = widgets.Layout(width='100%', margin='2px 0')
    common_style = {'description_width': '90px'}
    
    return {
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
        'validate_checkbox': widgets.Checkbox(
            value=True, 
            description='Validasi download', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        'backup_checkbox': widgets.Checkbox(
            value=False, 
            description='Backup existing', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        )
    }

def _create_grid_form_container(form_fields: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create form container dengan CSS Grid layout - responsive tanpa overflow"""
    
    # Format info yang selalu ditampilkan
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px; width: 100%; box-sizing: border-box;">
        <small style="color: #1976d2;"><strong>📦 Format:</strong> YOLOv5 PyTorch (hardcoded)</small>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # Left column - dataset config
    left_column = widgets.VBox([
        form_fields['workspace_input'],
        form_fields['project_input'], 
        form_fields['version_input'],
        form_fields['api_key_input']
    ], layout=widgets.Layout(
        width='48%', 
        padding='8px', 
        box_sizing='border-box',
        flex='1 1 48%'
    ))
    
    # Right column - options
    right_column = widgets.VBox([
        format_info,
        form_fields['validate_checkbox'],
        form_fields['backup_checkbox'],
        widgets.HTML("""<div style="height: 20px;"></div>""")  # Spacer untuk alignment
    ], layout=widgets.Layout(
        width='48%', 
        padding='8px',
        box_sizing='border-box',
        flex='1 1 48%'
    ))
    
    # Form row dengan responsive flexbox
    form_row = widgets.HBox([left_column, right_column], layout=widgets.Layout(
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
        button_style='',  # Secondary/default style (grey)
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

def _create_responsive_action_buttons() -> Dict[str, widgets.Widget]:
    """Create action buttons dengan responsive flexbox layout"""
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
    
    # Set accordion terbuka dan title
    log_accordion.set_title(0, '📋 Download Logs')
    log_accordion.selected_index = 0  # Terbuka by default
    
    return {'log_output': log_output, 'log_accordion': log_accordion, 'accordion': log_accordion}

# One-liner utilities
detect_api_key = lambda: get_api_key_from_secrets() or ''
validate_ui_layout = lambda ui: all(key in ui for key in ['ui', 'form_container', 'save_button', 'reset_button', 'download_button', 'log_output'])
get_ui_status = lambda ui: f"✅ UI Ready: {len([k for k in ui.keys() if not k.startswith('_')])} components | API Key: {'✅' if detect_api_key() else '❌'}"