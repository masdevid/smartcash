"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: Fixed UI components dengan one-liner style tanpa fallbacks berlebihan
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color

def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI dengan one-liner direct creation - no fallbacks"""
    config, roboflow = config or {}, config.get('roboflow', {})
    
    # Direct UI creation tanpa try-catch berlebihan
    ui_components = _create_fixed_downloader_ui(config)
    ui_components['layout_order_fixed'] = True
    
    return ui_components

def _create_fixed_downloader_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create downloader UI dengan fixed layout order - one-liner style"""
    roboflow = config.get('roboflow', {})
    
    # 1. Header dengan gradient background
    header = widgets.HTML(f"""<div style="background: linear-gradient(135deg, {get_namespace_color('DOWNLOAD')}, {get_namespace_color('DOWNLOAD')}CC); padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><h3 style="margin: 0; color: white; font-weight: 600;">📥 Dataset Downloader</h3><p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">Download dan organize dataset untuk SmartCash training</p></div>""")
    
    # 2. Status panel dengan environment detection
    status_panel = widgets.HTML(_get_status_panel_html())
    
    # 3. Form fields dengan one-liner creation
    form_fields = _create_form_fields_oneliners(roboflow)
    
    # 4. Form container dengan two-column layout
    form_container = _create_form_container_oneliner(form_fields)
    
    # 5. Save/Reset buttons
    save_reset_components = _create_save_reset_oneliners()
    
    # 6. Confirmation area
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', max_height='400px', overflow='auto', display='none', margin='10px 0'))
    
    # 7. Action buttons
    action_components = _create_action_buttons_oneliners()
    
    # 8. Progress tracking
    progress_components = create_progress_tracking_container()
    
    # 9. Log output
    log_components = _create_log_components_oneliners()
    
    # 10. Main UI container dengan fixed order
    main_ui = widgets.VBox([
        header,                                    # 1. Header
        status_panel,                             # 2. Status  
        form_container,                           # 3. Form
        save_reset_components['container'],       # 4. Save/Reset
        confirmation_area,                        # 5. Confirmation
        action_components['container'],           # 6. Actions
        progress_components['container'],         # 7. Progress
        log_components['accordion']               # 8. Logs
    ], layout=widgets.Layout(width='100%', max_width='100%', padding='0', margin='0'))
    
    # Combine all components dengan one-liner dict merge
    return {**{
        'ui': main_ui, 'main_container': main_ui, 'header': header, 'status_panel': status_panel,
        'form_container': form_container, 'confirmation_area': confirmation_area
    }, **form_fields, **save_reset_components, **action_components, **progress_components, **log_components}

def _get_status_panel_html() -> str:
    """Get status panel HTML dengan environment detection - one-liner"""
    try:
        import google.colab
        from pathlib import Path
        is_drive_mounted = Path('/content/drive/MyDrive').exists()
        return """<div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 4px; margin-bottom: 15px;"><span style="color: #2e7d32;">✅ Google Drive terhubung - Dataset akan tersimpan permanen</span></div>""" if is_drive_mounted else """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px;"><span style="color: #856404;">⚠️ Drive tidak terhubung - Dataset akan hilang saat restart</span></div>"""
    except ImportError:
        return """<div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin-bottom: 15px;"><span style="color: #1976d2;">📊 Status: Ready untuk download dataset</span></div>"""

def _create_form_fields_oneliners(roboflow: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create form fields dengan one-liner style"""
    api_key = _detect_api_key_oneliner()
    
    return {
        'workspace_input': widgets.Text(value=roboflow.get('workspace', 'smartcash-wo2us'), description='Workspace:', placeholder='Nama workspace Roboflow', layout=widgets.Layout(width='100%'), style={'description_width': '90px'}),
        'project_input': widgets.Text(value=roboflow.get('project', 'rupiah-emisi-2022'), description='Project:', placeholder='Nama project Roboflow', layout=widgets.Layout(width='100%'), style={'description_width': '90px'}),
        'version_input': widgets.Text(value=str(roboflow.get('version', '3')), description='Version:', placeholder='Versi dataset', layout=widgets.Layout(width='100%'), style={'description_width': '90px'}),
        'api_key_input': widgets.Password(value=api_key or roboflow.get('api_key', ''), description='API Key:', placeholder='🔑 Auto-detect dari Colab secrets' if api_key else 'Masukkan API Key Roboflow', layout=widgets.Layout(width='100%'), style={'description_width': '90px'}),
        'validate_checkbox': widgets.Checkbox(value=True, description='Validasi download', layout=widgets.Layout(width='100%')),
        'organize_checkbox': widgets.Checkbox(value=True, description='Organisir dataset', layout=widgets.Layout(width='100%')),
        'backup_checkbox': widgets.Checkbox(value=False, description='Backup existing', layout=widgets.Layout(width='100%'))
    }

def _create_form_container_oneliner(form_fields: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create form container dengan two-column layout - one-liner style"""
    format_info = widgets.HTML("""<div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px;"><small style="color: #1976d2;"><strong>Format:</strong> YOLOv5 PyTorch (hardcoded)</small></div>""")
    left_column = widgets.VBox([form_fields['workspace_input'], form_fields['project_input'], form_fields['version_input'], form_fields['api_key_input']], layout=widgets.Layout(width='48%', padding='8px'))
    right_column = widgets.VBox([format_info, form_fields['validate_checkbox'], form_fields['organize_checkbox'], form_fields['backup_checkbox']], layout=widgets.Layout(width='48%', padding='8px'))
    
    return widgets.VBox([widgets.HBox([left_column, right_column], layout=widgets.Layout(width='100%', justify_content='space-between', border='1px solid #ddd', border_radius='5px', padding='15px', margin='0 0 15px 0'))], layout=widgets.Layout(width='100%'))

def _create_save_reset_oneliners() -> Dict[str, widgets.Widget]:
    """Create save/reset buttons dengan one-liner style"""
    save_button = widgets.Button(description='💾 Simpan', button_style='success', layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px'))
    reset_button = widgets.Button(description='🔄 Reset', button_style='', layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px'))
    container = widgets.HBox([save_button, reset_button], layout=widgets.Layout(width='100%', justify_content='flex-end', margin='10px 0'))
    
    return {'save_button': save_button, 'reset_button': reset_button, 'container': container}

def _create_action_buttons_oneliners() -> Dict[str, widgets.Widget]:
    """Create action buttons dengan one-liner style"""
    button_layout = widgets.Layout(width='auto', min_width='150px', height='35px', margin='5px')
    download_button = widgets.Button(description='📥 Download', button_style='primary', layout=button_layout)
    check_button = widgets.Button(description='🔍 Check', button_style='info', layout=button_layout)
    cleanup_button = widgets.Button(description='🧹 Cleanup', button_style='danger', layout=button_layout)
    container = widgets.HBox([download_button, check_button, cleanup_button], layout=widgets.Layout(width='100%', justify_content='flex-start', margin='15px 0'))
    
    return {'download_button': download_button, 'check_button': check_button, 'cleanup_button': cleanup_button, 'container': container}

def _create_log_components_oneliners() -> Dict[str, widgets.Widget]:
    """Create log components dengan one-liner style"""
    log_output = widgets.Output(layout=widgets.Layout(width='100%', max_height='300px', border='1px solid #ddd', border_radius='4px', padding='8px', overflow='auto'))
    log_accordion = widgets.Accordion([log_output], layout=widgets.Layout(width='100%', margin='10px 0'))
    setattr(log_accordion, 'selected_index', None), log_accordion.set_title(0, '📋 Download Logs')
    
    return {'log_output': log_output, 'log_accordion': log_accordion, 'accordion': log_accordion}

def _detect_api_key_oneliner() -> str:
    """Detect API key dari Colab secrets dengan one-liner style"""
    try:
        from google.colab import userdata
        return next((userdata.get(key, '').strip() for key in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'API_KEY'] if userdata.get(key, '').strip() and len(userdata.get(key, '').strip()) > 10), '')
    except (ImportError, Exception):
        return ''

# One-liner utilities untuk UI management
create_minimal_form = lambda config: _create_form_fields_oneliners(config.get('roboflow', {}))
validate_ui_components = lambda ui: all(key in ui for key in ['ui', 'form_container', 'save_button', 'reset_button', 'download_button', 'check_button', 'cleanup_button', 'log_output'])
get_component_summary = lambda ui: f"UI Components: {len([k for k in ui.keys() if not k.startswith('_')])} | Fixed Layout: {ui.get('layout_order_fixed', False)}"
is_ui_ready = lambda ui: validate_ui_components(ui) and ui.get('layout_order_fixed', False)