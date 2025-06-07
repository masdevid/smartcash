"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: FIXED UI components dengan proper button creation dan component mapping
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components.progress_tracker import create_triple_progress_tracker
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.header import create_header
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column
from smartcash.ui.dataset.downloader.utils.colab_secrets import get_api_key_from_secrets

def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """FIXED: Create main downloader UI dengan proper component creation dan mapping"""
    config = config or {}
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    detected_api_key = get_api_key_from_secrets()
    
    # 1. Header
    header = create_header(
        title="Dataset Downloader",
        description="Download dataset Roboflow untuk SmartCash training (format YOLOv5)",
        icon="📥"
    )
    
    # 2. Status panel
    status_panel = widgets.HTML(_get_dynamic_status_html(), 
                               layout=widgets.Layout(width='100%', margin='0 0 15px 0'))
    
    # 3. Form fields - FIXED: Proper creation
    form_fields = _create_form_fields(roboflow, detected_api_key, download)
    form_container = _create_form_container_with_shared_layout(form_fields)
    
    # 4. Save/Reset buttons - FIXED: Proper extraction
    save_reset_components = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset", 
        button_width="100px",
        with_sync_info=True,
        sync_message="Konfigurasi tersimpan ke dataset_config.yaml"
    )
    
    # 5. Action buttons - FIXED: Proper creation
    action_components = _create_action_buttons()
    
    # 6. Progress tracker - FIXED: Proper creation
    progress_components = create_triple_progress_tracker(auto_hide=True)
    progress_components['container'].layout.display = 'none'
    
    # 7. Log accordion - FIXED: Proper creation
    log_components = create_log_accordion(
        module_name='downloader',
        height='250px',
        width='100%'
    )
    
    # 8. Main container
    main_ui = create_responsive_container([
        header,
        status_panel, 
        form_container,
        save_reset_components['container'],
        _create_action_header(),
        action_components['container'],
        progress_components['container'],
        log_components['log_accordion']
    ], container_type='vbox')
    
    # FIXED: Proper component mapping dengan explicit keys
    result = {
        # Main UI
        'ui': main_ui,
        'main_container': main_ui,
        'header': header,
        'status_panel': status_panel,
        'form_container': form_container,
        
        # Form fields - explicit mapping
        'workspace_input': form_fields['workspace_input'],
        'project_input': form_fields['project_input'],
        'version_input': form_fields['version_input'],
        'api_key_input': form_fields['api_key_input'],
        'validate_checkbox': form_fields['validate_checkbox'],
        'backup_checkbox': form_fields['backup_checkbox'],
        
        # Save/Reset buttons - FIXED: Explicit mapping
        'save_button': save_reset_components['save_button'],
        'reset_button': save_reset_components['reset_button'],
        'save_reset_container': save_reset_components['container'],
        
        # Action buttons - FIXED: Explicit mapping dengan fallback protection
        'download_button': action_components.get('download_button'),
        'check_button': action_components.get('check_button'), 
        'cleanup_button': action_components.get('cleanup_button'),
        'action_container': action_components['container'],
        
        # Progress tracker - FIXED: Explicit mapping
        'progress_tracker': progress_components['tracker'],
        'progress_container': progress_components['container'],
        'progress_status_widget': progress_components.get('status_widget'),
        
        # Log components - FIXED: Explicit mapping
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
    }
    
    # FIXED: Validation dan debugging
    critical_components = ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button', 'log_output', 'progress_tracker']
    missing_components = [comp for comp in critical_components if result.get(comp) is None]
    
    if missing_components:
        print(f"⚠️ MISSING COMPONENTS: {missing_components}")
    
    return result

def _create_form_fields(roboflow: Dict[str, Any], api_key: str, download: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create form fields dengan consistent layout - unchanged"""
    common_layout = widgets.Layout(width='100%', margin='2px 0')
    common_style = {'description_width': '100px'}
    
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
            value=download.get('validate_download', True), 
            description='Validasi download', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        ),
        'backup_checkbox': widgets.Checkbox(
            value=download.get('backup_existing', False), 
            description='Backup existing data', 
            layout=widgets.Layout(width='100%', margin='2px 0')
        )
    }

def _create_form_container_with_shared_layout(form_fields: Dict[str, widgets.Widget]) -> widgets.Widget:
    """Create form container - unchanged"""
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px; 
                width: 100%; box-sizing: border-box; word-wrap: break-word; overflow-wrap: break-word;">
        <small style="color: #1976d2;"><strong>📦 Format:</strong> YOLOv5 PyTorch (hardcoded)</small>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    left_content = create_responsive_container([
        form_fields['workspace_input'], 
        form_fields['project_input'], 
        form_fields['version_input'], 
        form_fields['api_key_input']
    ], container_type='vbox')
    
    right_content = create_responsive_container([
        format_info, 
        form_fields['validate_checkbox'], 
        form_fields['backup_checkbox']
    ], container_type='vbox')
    
    form_container = create_responsive_two_column(
        left_content, 
        right_content,
        left_width='48%', 
        right_width='48%'
    )
    
    return widgets.VBox([form_container], layout=widgets.Layout(
        width='100%', 
        border='1px solid #ddd', 
        border_radius='5px',
        padding='15px', 
        margin='0 0 15px 0', 
        box_sizing='border-box',
        overflow='hidden'
    ))

def _create_action_buttons() -> Dict[str, widgets.Widget]:
    """FIXED: Create action buttons dengan proper creation dan guaranteed components"""
    button_layout = widgets.Layout(
        width='auto', 
        min_width='140px', 
        max_width='200px',
        height='35px', 
        margin='5px',
        overflow='hidden'
    )
    
    # FIXED: Guaranteed button creation
    download_button = widgets.Button(
        description='📥 Download', 
        button_style='primary', 
        layout=button_layout,
        tooltip='Download dataset dari Roboflow'
    )
    
    check_button = widgets.Button(
        description='🔍 Check', 
        button_style='info', 
        layout=button_layout,
        tooltip='Check status dataset existing'
    )
    
    cleanup_button = widgets.Button(
        description='🧹 Cleanup', 
        button_style='danger', 
        layout=button_layout,
        tooltip='Hapus dataset existing'
    )
    
    # FIXED: Container dengan proper children
    container = widgets.HBox([download_button, check_button, cleanup_button], layout=widgets.Layout(
        width='100%', 
        justify_content='flex-start', 
        margin='15px 0',
        display='flex', 
        flex_flow='row wrap', 
        align_items='center', 
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    # FIXED: Return dengan guaranteed components
    return {
        'download_button': download_button, 
        'check_button': check_button, 
        'cleanup_button': cleanup_button, 
        'container': container,
        'buttons': [download_button, check_button, cleanup_button]  # Explicit list
    }


def _create_action_header() -> widgets.HTML:
    """Create action section header - unchanged"""
    return widgets.HTML("""
    <h4 style='color: #333; margin: 15px 0 10px 0; font-size: 16px; font-weight: 600;
               border-bottom: 2px solid #28a745; padding-bottom: 6px; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap;'>
        ▶️ Actions
    </h4>""", layout=widgets.Layout(width='100%', margin='0', overflow='hidden'))

def _get_dynamic_status_html() -> str:
    """Get status HTML dengan environment detection - unchanged"""
    try:
        import google.colab
        from pathlib import Path
        is_drive_mounted = Path('/content/drive/MyDrive/SmartCash').exists()
        api_key = get_api_key_from_secrets()
        
        if is_drive_mounted and api_key:
            return """<div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; 
                      border-radius: 4px; margin-bottom: 15px; word-wrap: break-word; overflow-wrap: break-word;">
                      <span style="color: #2e7d32;">✅ Drive terhubung + API Key terdeteksi - Siap download!</span></div>"""
        elif is_drive_mounted:
            return """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; 
                      border-radius: 4px; margin-bottom: 15px; word-wrap: break-word; overflow-wrap: break-word;">
                      <span style="color: #856404;">⚠️ Drive terhubung - Masukkan API Key untuk mulai</span></div>"""
        elif api_key:
            return """<div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; 
                      border-radius: 4px; margin-bottom: 15px; word-wrap: break-word; overflow-wrap: break-word;">
                      <span style="color: #856404;">⚠️ API Key tersedia - Mount Drive untuk penyimpanan permanen</span></div>"""
        else:
            return """<div style="padding: 12px; background: #f8d7da; border-left: 4px solid #dc3545; 
                      border-radius: 4px; margin-bottom: 15px; word-wrap: break-word; overflow-wrap: break-word;">
                      <span style="color: #721c24;">❌ Perlu mount Drive dan setup API Key</span></div>"""
    except ImportError:
        return """<div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; 
                  border-radius: 4px; margin-bottom: 15px; word-wrap: break-word; overflow-wrap: break-word;">
                  <span style="color: #1976d2;">📊 Status: Local environment - Ready</span></div>"""