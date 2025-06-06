"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: Fixed UI components dengan API progress_tracker yang benar dan optimasi progress tracker
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color
from smartcash.ui.dataset.downloader.utils.colab_secrets import get_api_key_from_secrets

def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI dengan API progress tracker yang benar"""
    config = config or {}
    roboflow = config.get('roboflow', {})
    detected_api_key = get_api_key_from_secrets()
    
    ui_components = _create_streamlined_downloader_ui(config, roboflow, detected_api_key)
    ui_components['layout_optimized'] = True
    return ui_components

def _create_streamlined_downloader_ui(config: Dict[str, Any], roboflow: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Create streamlined downloader UI dengan API progress tracker yang benar"""
    
    # 1. Header dengan gradient design
    header = widgets.HTML(f"""
    <div style="background: linear-gradient(135deg, {get_namespace_color('DOWNLOAD')}, {get_namespace_color('DOWNLOAD')}CC); 
                padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 100%; box-sizing: border-box;">
        <h3 style="margin: 0; color: white; font-weight: 600;">📥 Dataset Downloader</h3>
        <p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">Download dataset Roboflow untuk SmartCash training (format YOLOv5)</p>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # 2. Status panel dinamis
    status_panel = widgets.HTML(_get_dynamic_status_html(), layout=widgets.Layout(width='100%', margin='0 0 15px 0'))
    
    # 3. Streamlined form fields tanpa path config
    form_fields = _create_streamlined_form_fields(roboflow, api_key, config)
    
    # 4. 2-column form container (dataset config + options)
    form_container = _create_two_column_form_container(form_fields)
    
    # 5. Save/Reset buttons
    save_reset_components = _create_save_reset_buttons()
    
    # 6. Confirmation area
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', max_height='400px', overflow='auto', display='none', margin='10px 0'))
    
    # 7. Action buttons dengan state management
    action_components = _create_action_buttons()
    
    # 8. Progress tracker dengan dual level - API yang benar
    progress_tracker = create_dual_progress_tracker(operation="Dataset Download")
    progress_container = progress_tracker.container
    progress_container.layout.display = 'none'
    
    # 9. Log accordion terbuka by default
    log_components = _create_log_accordion()
    
    # 10. Action header
    action_header = widgets.HTML("""
    <h4 style='color: #333; margin: 15px 0 10px 0; font-size: 16px; 
               border-bottom: 2px solid #28a745; padding-bottom: 6px;'>
        ▶️ Actions
    </h4>""")
    
    # 11. Main container streamlined
    main_ui = widgets.VBox([
        header,
        status_panel,
        form_container,
        save_reset_components['container'],
        action_header,
        confirmation_area,
        action_components['container'],
        progress_container,
        log_components['accordion']
    ], layout=widgets.Layout(
        width='100%', max_width='100%', padding='0', margin='0',
        display='flex', flex_flow='column nowrap', align_items='stretch',
        overflow='hidden', box_sizing='border-box'
    ))
    
    return {
        # Main UI
        'ui': main_ui, 'main_container': main_ui, 'header': header, 'status_panel': status_panel,
        'form_container': form_container, 'confirmation_area': confirmation_area,
        
        # Form fields tanpa path fields
        **form_fields,
        
        # Buttons
        **save_reset_components, **action_components,
        
        # Progress tracker dengan API yang benar
        'progress_tracker': progress_tracker, 'progress_container': progress_container,
        
        # Log components
        **log_components
    }

def _create_streamlined_form_fields(roboflow: Dict[str, Any], api_key: str, config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create streamlined form fields tanpa backup/preprocessing directories"""
    common_layout = widgets.Layout(width='100%', margin='2px 0')
    common_style = {'description_width': '100px'}
    
    return {
        # Dataset configuration only
        'workspace_input': widgets.Text(
            value=roboflow.get('workspace', 'smartcash-wo2us'), 
            description='Workspace:', placeholder='Nama workspace Roboflow', 
            layout=common_layout, style=common_style
        ),
        'project_input': widgets.Text(
            value=roboflow.get('project', 'rupiah-emisi-2022'), 
            description='Project:', placeholder='Nama project Roboflow', 
            layout=common_layout, style=common_style
        ),
        'version_input': widgets.Text(
            value=str(roboflow.get('version', '3')), 
            description='Version:', placeholder='Versi dataset', 
            layout=common_layout, style=common_style
        ),
        'api_key_input': widgets.Password(
            value=api_key or roboflow.get('api_key', ''), 
            description='API Key:', 
            placeholder='🔑 Auto-detect dari Colab secrets' if api_key else 'Masukkan API Key Roboflow', 
            layout=common_layout, style=common_style
        ),
        
        # Options only (no path configuration)
        'validate_checkbox': widgets.Checkbox(
            value=config.get('validate_download', True), 
            description='Validasi download', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        'backup_checkbox': widgets.Checkbox(
            value=config.get('backup_existing', False), 
            description='Backup existing data', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        )
    }

def _create_two_column_form_container(form_fields: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create 2-column form container tanpa path configuration"""
    
    # Format info hardcoded
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px; width: 100%; box-sizing: border-box;">
        <small style="color: #1976d2;"><strong>📦 Format:</strong> YOLOv5 PyTorch (hardcoded)</small>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # Left column - dataset config
    left_column = widgets.VBox([
        form_fields['workspace_input'], form_fields['project_input'], 
        form_fields['version_input'], form_fields['api_key_input']
    ], layout=widgets.Layout(width='48%', padding='8px', box_sizing='border-box', flex='1 1 48%'))
    
    # Right column - options
    right_column = widgets.VBox([
        format_info, form_fields['validate_checkbox'], form_fields['backup_checkbox'],
        widgets.HTML("""<div style="height: 60px;"></div>""")  # Spacer
    ], layout=widgets.Layout(width='48%', padding='8px', box_sizing='border-box', flex='1 1 48%'))
    
    # Form row responsive
    form_row = widgets.HBox([left_column, right_column], layout=widgets.Layout(
        width='100%', display='flex', flex_flow='row wrap', justify_content='space-between',
        align_items='flex-start', border='1px solid #ddd', border_radius='5px',
        padding='15px', margin='0 0 15px 0', box_sizing='border-box', overflow='hidden'
    ))
    
    return widgets.VBox([form_row], layout=widgets.Layout(width='100%', margin='0'))

def _create_save_reset_buttons() -> Dict[str, widgets.Widget]:
    """Create save/reset buttons tanpa icon"""
    save_button = widgets.Button(description='Simpan', button_style='primary', 
                                layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px'))
    reset_button = widgets.Button(description='Reset', button_style='', 
                                 layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px'))
    
    container = widgets.HBox([save_button, reset_button], layout=widgets.Layout(
        width='100%', justify_content='flex-end', margin='10px 0',
        display='flex', flex_flow='row nowrap', align_items='center'
    ))
    
    return {'save_button': save_button, 'reset_button': reset_button, 'container': container}

def _create_action_buttons() -> Dict[str, widgets.Widget]:
    """Create action buttons dengan state management"""
    button_layout = widgets.Layout(width='auto', min_width='140px', height='35px', margin='5px', flex='0 1 auto')
    
    download_button = widgets.Button(description='📥 Download', button_style='primary', layout=button_layout)
    check_button = widgets.Button(description='🔍 Check', button_style='info', layout=button_layout)
    cleanup_button = widgets.Button(description='🧹 Cleanup', button_style='danger', layout=button_layout)
    
    # State management attributes - one-liner setup
    all_buttons = [download_button, check_button, cleanup_button]
    [setattr(btn, '_all_buttons', all_buttons) for btn in all_buttons]
    
    container = widgets.HBox(all_buttons, layout=widgets.Layout(
        width='100%', justify_content='flex-start', margin='15px 0',
        display='flex', flex_flow='row wrap', align_items='center', overflow='hidden'
    ))
    
    return {'download_button': download_button, 'check_button': check_button, 'cleanup_button': cleanup_button, 'container': container}

def _create_log_accordion() -> Dict[str, widgets.Widget]:
    """Create log accordion terbuka by default"""
    log_output = widgets.Output(layout=widgets.Layout(
        width='100%', max_height='300px', border='1px solid #ddd', 
        border_radius='4px', padding='8px', overflow='auto', box_sizing='border-box'
    ))
    
    log_accordion = widgets.Accordion([log_output], layout=widgets.Layout(width='100%', margin='10px 0', box_sizing='border-box'))
    log_accordion.set_title(0, '📋 Download Logs')
    log_accordion.selected_index = 0  # Terbuka by default
    
    return {'log_output': log_output, 'log_accordion': log_accordion, 'accordion': log_accordion}

def _get_dynamic_status_html() -> str:
    """Get status HTML dengan environment detection - one-liner nested conditionals"""
    try:
        import google.colab
        from pathlib import Path
        is_drive_mounted = Path('/content/drive/MyDrive').exists()
        api_key = get_api_key_from_secrets()
        
        # Status generation dengan nested conditionals - one-liner
        return ("""<div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 4px; margin-bottom: 15px;"><span style="color: #2e7d32;">✅ Drive terhubung + API Key terdeteksi - Siap download!</span></div>""" if (is_drive_mounted and api_key) else
                """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px;"><span style="color: #856404;">⚠️ Drive terhubung - Masukkan API Key untuk mulai</span></div>""" if is_drive_mounted else
                """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px;"><span style="color: #856404;">⚠️ API Key tersedia - Mount Drive untuk penyimpanan permanen</span></div>""" if api_key else
                """<div style="padding: 12px; background: #f8d7da; border-left: 4px solid #dc3545; border-radius: 4px; margin-bottom: 15px;"><span style="color: #721c24;">❌ Perlu mount Drive dan setup API Key</span></div>""")
    except ImportError:
        return """<div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin-bottom: 15px;"><span style="color: #1976d2;">📊 Status: Local environment - Ready</span></div>"""

# Utilities dengan one-liner style
detect_api_key = lambda: get_api_key_from_secrets() or ''
validate_ui_layout = lambda ui: all(key in ui for key in ['ui', 'form_container', 'save_button', 'download_button', 'log_output', 'progress_tracker'])
get_ui_status = lambda ui: f"✅ Streamlined UI Ready: {len([k for k in ui.keys() if not k.startswith('_')])} components | API: {'✅' if detect_api_key() else '❌'} | Progress: {'✅' if 'progress_tracker' in ui else '❌'}"
disable_other_buttons = lambda btn: hasattr(btn, '_all_buttons') and [setattr(b, 'disabled', True) for b in btn._all_buttons if b != btn]
enable_all_buttons = lambda btns: [setattr(btn, 'disabled', False) for btn in btns if hasattr(btn, 'disabled')]