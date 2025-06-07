"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: FIXED UI components menggunakan progress_tracker yang benar
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.components.progress_tracker import create_triple_progress_tracker  # ✅ FIXED
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.header import create_header
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column
from smartcash.ui.dataset.downloader.utils.colab_secrets import get_api_key_from_secrets

def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI menggunakan shared components"""
    config = config or {}
    roboflow = config.get('data', {}).get('roboflow', {})
    download = config.get('download', {})
    detected_api_key = get_api_key_from_secrets()
    
    ui_components = _create_downloader_ui_with_shared_components(config, roboflow, download, detected_api_key)
    ui_components['layout_optimized'] = True
    return ui_components

def _create_downloader_ui_with_shared_components(config: Dict[str, Any], roboflow: Dict[str, Any], 
                                               download: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Create downloader UI menggunakan shared components dari ui/components"""
    
    # 1. Header menggunakan shared component
    header = create_header(
        title="Dataset Downloader",
        description="Download dataset Roboflow untuk SmartCash training (format YOLOv5)",
        icon="📥"
    )
    
    # 2. Status panel dengan environment detection
    status_panel = widgets.HTML(_get_dynamic_status_html(), 
                               layout=widgets.Layout(width='100%', margin='0 0 15px 0'))
    
    # 3. Form fields dengan proper layout
    form_fields = _create_form_fields(roboflow, api_key, download)
    
    # 4. Two-column form menggunakan shared layout
    form_container = _create_form_container_with_shared_layout(form_fields)
    
    # 5. Save/Reset buttons menggunakan shared component
    save_reset_components = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        button_width="100px",
        with_sync_info=True,
        sync_message="Konfigurasi tersimpan ke dataset_config.yaml"
    )
    
    # 6. Action buttons dengan proper state management
    action_components = _create_action_buttons_fixed()
    
    # 7. Progress tracker - ✅ FIXED: Menggunakan triple untuk Overall + Step + Current
    progress_components = create_triple_progress_tracker(
        operation="Dataset Download",
        steps=["Inisialisasi", "Download", "Organisasi", "UUID Rename", "Validasi", "Cleanup"],
        auto_hide=True
    )
    
    # Hide progress container initially
    progress_components['container'].layout.display = 'none'
    
    # 8. Log accordion menggunakan shared component dengan FIXED overflow
    log_components = create_log_accordion(
        module_name='downloader',
        height='250px',
        width='100%'
    )
    
    # 9. Main container menggunakan responsive layout
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
    
    return {
        # Main UI
        'ui': main_ui, 'main_container': main_ui, 'header': header, 
        'status_panel': status_panel, 'form_container': form_container,
        
        # Form fields
        **form_fields,
        
        # Buttons menggunakan shared components
        **save_reset_components, **action_components,
        
        # Progress tracker - ✅ FIXED: Menggunakan komponen yang benar
        'progress_tracker': progress_components['tracker'],
        'progress_container': progress_components['container'],
        
        # Log components - ✅ FIXED: Key names yang benar
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        
        # Additional progress components
        **{k: v for k, v in progress_components.items() if k not in ['tracker', 'container']}
    }

def _create_form_fields(roboflow: Dict[str, Any], api_key: str, download: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create form fields dengan consistent layout"""
    common_layout = widgets.Layout(width='100%', margin='2px 0')
    common_style = {'description_width': '100px'}
    
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
        
        # Options
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
    """Create form container menggunakan shared layout utilities"""
    
    # Format info
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px; 
                width: 100%; box-sizing: border-box; word-wrap: break-word; overflow-wrap: break-word;">
        <small style="color: #1976d2;"><strong>📦 Format:</strong> YOLOv5 PyTorch (hardcoded)</small>
    </div>""", layout=widgets.Layout(width='100%', margin='0'))
    
    # Left column - dataset config
    left_content = create_responsive_container([
        form_fields['workspace_input'], 
        form_fields['project_input'], 
        form_fields['version_input'], 
        form_fields['api_key_input']
    ], container_type='vbox')
    
    # Right column - options
    right_content = create_responsive_container([
        format_info, 
        form_fields['validate_checkbox'], 
        form_fields['backup_checkbox']
    ], container_type='vbox')
    
    # Two-column layout menggunakan shared utility
    form_container = create_responsive_two_column(
        left_content, 
        right_content,
        left_width='48%', 
        right_width='48%'
    )
    
    # Wrapper dengan border
    wrapper = widgets.VBox([form_container], layout=widgets.Layout(
        width='100%', 
        border='1px solid #ddd', 
        border_radius='5px',
        padding='15px', 
        margin='0 0 15px 0', 
        box_sizing='border-box',
        overflow='hidden'
    ))
    
    return wrapper

def _create_action_buttons_fixed() -> Dict[str, widgets.Widget]:
    """Create action buttons dengan proper state management dan FIXED overflow"""
    button_layout = widgets.Layout(
        width='auto', 
        min_width='140px', 
        max_width='200px',  # Prevent overflow
        height='35px', 
        margin='5px',
        overflow='hidden'  # FIXED: Prevent text overflow
    )
    
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
    
    # State management attributes
    all_buttons = [download_button, check_button, cleanup_button]
    [setattr(btn, '_all_buttons', all_buttons) for btn in all_buttons]
    
    container = widgets.HBox(all_buttons, layout=widgets.Layout(
        width='100%', 
        justify_content='flex-start', 
        margin='15px 0',
        display='flex', 
        flex_flow='row wrap', 
        align_items='center', 
        overflow='hidden',  # FIXED: Prevent container overflow
        box_sizing='border-box'
    ))
    
    return {
        'download_button': download_button, 
        'check_button': check_button, 
        'cleanup_button': cleanup_button, 
        'container': container
    }

def _create_action_header() -> widgets.HTML:
    """Create action section header"""
    return widgets.HTML("""
    <h4 style='color: #333; margin: 15px 0 10px 0; font-size: 16px; font-weight: 600;
               border-bottom: 2px solid #28a745; padding-bottom: 6px; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap;'>
        ▶️ Actions
    </h4>""", layout=widgets.Layout(width='100%', margin='0', overflow='hidden'))

def _get_dynamic_status_html() -> str:
    """Get status HTML dengan environment detection"""
    try:
        import google.colab
        from pathlib import Path
        is_drive_mounted = Path('/content/drive/MyDrive').exists()
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

# Utilities dengan overflow fixes
validate_ui_layout = lambda ui: all(key in ui for key in ['ui', 'form_container', 'save_button', 'download_button', 'log_output', 'progress_tracker'])
get_ui_status = lambda ui: f"✅ UI Ready: {len([k for k in ui.keys() if not k.startswith('_')])} components"