"""
File: smartcash/ui/dataset/downloader/components/ui_layout.py
Deskripsi: One-liner UI layout tanpa fallbacks berlebihan - direct component creation
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color
from smartcash.ui.components.progress_tracking import create_progress_tracking_container

def create_downloader_ui(config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """Create downloader UI dengan one-liner direct creation tanpa fallbacks"""
    config, roboflow = config or {}, config.get('roboflow', {})
    
    # One-liner component creation
    header = widgets.HTML(f"""<div style="background: linear-gradient(135deg, {get_namespace_color('DOWNLOAD')}, {get_namespace_color('DOWNLOAD')}CC); padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"><h3 style="margin: 0; color: white; font-weight: 600;">📥 Dataset Downloader</h3><p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">Download dan organize dataset untuk SmartCash training</p></div>""")
    
    status_panel = widgets.HTML("""<div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; border-radius: 4px; margin-bottom: 15px;"><span style="color: #2e7d32;">✅ Google Drive terhubung - Dataset akan tersimpan permanen</span></div>""" if env and getattr(env, 'is_drive_mounted', False) else """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin-bottom: 15px;"><span style="color: #856404;">⚠️ Drive tidak terhubung - Dataset akan hilang saat restart</span></div>""" if env and getattr(env, 'is_colab', False) else """<div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px; margin-bottom: 15px;"><span style="color: #1976d2;">📊 Status: Ready untuk download dataset</span></div>""")
    
    # Form fields dengan one-liner creation
    workspace_input = widgets.Text(value=roboflow.get('workspace', 'smartcash-wo2us'), description='Workspace:', placeholder='Nama workspace Roboflow', layout=widgets.Layout(width='100%'), style={'description_width': '90px'})
    project_input = widgets.Text(value=roboflow.get('project', 'rupiah-emisi-2022'), description='Project:', placeholder='Nama project Roboflow', layout=widgets.Layout(width='100%'), style={'description_width': '90px'})
    version_input = widgets.Text(value=str(roboflow.get('version', '3')), description='Version:', placeholder='Versi dataset', layout=widgets.Layout(width='100%'), style={'description_width': '90px'})
    api_key_input = widgets.Password(value=roboflow.get('api_key', ''), description='API Key:', placeholder='Masukkan API Key Roboflow', layout=widgets.Layout(width='100%'), style={'description_width': '90px'})
    validate_checkbox = widgets.Checkbox(value=True, description='Validasi download', layout=widgets.Layout(width='100%'))
    organize_checkbox = widgets.Checkbox(value=True, description='Organisir dataset', layout=widgets.Layout(width='100%'))
    backup_checkbox = widgets.Checkbox(value=False, description='Backup existing', layout=widgets.Layout(width='100%'))
    
    # Form container dengan one-liner two-column layout
    format_info = widgets.HTML("""<div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px;"><small style="color: #1976d2;"><strong>Format:</strong> YOLOv5 PyTorch (hardcoded)</small></div>""")
    left_column = widgets.VBox([workspace_input, project_input, version_input, api_key_input], layout=widgets.Layout(width='48%', padding='8px'))
    right_column = widgets.VBox([format_info, validate_checkbox, organize_checkbox, backup_checkbox], layout=widgets.Layout(width='48%', padding='8px'))
    form_container = widgets.VBox([widgets.HBox([left_column, right_column], layout=widgets.Layout(width='100%', justify_content='space-between', border='1px solid #ddd', border_radius='5px', padding='15px', margin='0 0 15px 0'))], layout=widgets.Layout(width='100%'))
    
    # Save/Reset buttons dengan one-liner
    save_button = widgets.Button(description='💾 Simpan', button_style='success', layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px'))
    reset_button = widgets.Button(description='🔄 Reset', button_style='', layout=widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px'))
    save_reset_container = widgets.HBox([save_button, reset_button], layout=widgets.Layout(width='100%', justify_content='flex-end', margin='10px 0'))
    
    # Confirmation area dengan one-liner
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', max_height='400px', overflow='auto', display='none', margin='10px 0'))
    
    # Action buttons dengan one-liner
    button_layout = widgets.Layout(width='auto', min_width='150px', height='35px', margin='5px')
    download_button = widgets.Button(description='📥 Download', button_style='primary', layout=button_layout)
    check_button = widgets.Button(description='🔍 Check', button_style='info', layout=button_layout)
    cleanup_button = widgets.Button(description='🧹 Cleanup', button_style='danger', layout=button_layout)
    action_container = widgets.HBox([download_button, check_button, cleanup_button], layout=widgets.Layout(width='100%', justify_content='flex-start', margin='15px 0'))
    
    # Progress components dengan one-liner
    progress_components = create_progress_tracking_container()
    progress_container = widgets.VBox([progress_components.get('container', widgets.HTML("📊 Progress akan muncul saat operasi"))], layout=widgets.Layout(width='100%', margin='15px 0'))
    
    # Log components dengan one-liner
    log_output = widgets.Output(layout=widgets.Layout(width='100%', max_height='300px', border='1px solid #ddd', border_radius='4px', padding='8px', overflow='auto'))
    log_accordion = widgets.Accordion([log_output], layout=widgets.Layout(width='100%', margin='10px 0'))
    setattr(log_accordion, 'selected_index', None), log_accordion.set_title(0, '📋 Download Logs')
    
    # Main UI dengan one-liner fixed order
    ui = widgets.VBox([header, status_panel, form_container, save_reset_container, confirmation_area, action_container, progress_container, log_accordion], layout=widgets.Layout(width='100%', max_width='100%', padding='0', margin='0', overflow='hidden'))
    
    # Return components dengan one-liner dict update
    return dict(**{
        'ui': ui, 'main_container': ui, 'header': header, 'status_panel': status_panel,
        'form_container': form_container, 'save_reset_container': save_reset_container,
        'confirmation_area': confirmation_area, 'action_container': action_container,
        'progress_container': progress_container, 'log_output': log_output, 'log_accordion': log_accordion,
        'save_button': save_button, 'reset_button': reset_button,
        'download_button': download_button, 'check_button': check_button, 'cleanup_button': cleanup_button,
        'workspace_input': workspace_input, 'project_input': project_input, 
        'version_input': version_input, 'api_key_input': api_key_input,
        'validate_checkbox': validate_checkbox, 'organize_checkbox': organize_checkbox, 
        'backup_checkbox': backup_checkbox
    }, **progress_components)