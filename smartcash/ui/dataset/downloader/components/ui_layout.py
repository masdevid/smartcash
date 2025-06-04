"""
File: smartcash/ui/dataset/downloader/components/ui_layout.py
Deskripsi: Improved UI layout dengan flex design, two columns, dan namespace colors
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.fallback_utils import try_operation_safe, create_fallback_ui
from smartcash.ui.utils.ui_logger_namespace import get_namespace_color, KNOWN_NAMESPACES

def create_downloader_ui(config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """Create downloader UI dengan improved flex layout dan two columns."""
    config = config or {}
    
    try:
        ui_components = {}
        
        # Header dengan namespace color
        ui_components['header'] = _create_themed_header()
        
        # Status panel
        ui_components['status_panel'] = _create_status_panel(env)
        
        # Form fields
        form_fields = try_operation_safe(
            lambda: _create_form_fields(config),
            fallback_value=_create_basic_form_fields(config)
        )
        ui_components.update(form_fields)
        
        # Two column form layout
        ui_components['form_container'] = _create_two_column_form(ui_components)
        
        # Action buttons
        action_buttons = _create_action_buttons()
        ui_components.update(action_buttons)
        
        # Save/Reset buttons
        save_reset = _create_save_reset_buttons()
        ui_components.update(save_reset)
        
        # Log components
        log_components = _create_log_components()
        ui_components.update(log_components)
        
        # Main layout dengan flex
        ui_components['ui'] = _create_flex_main_layout(ui_components)
        ui_components['main_container'] = ui_components['ui']
        
        return ui_components
        
    except Exception as e:
        return create_fallback_ui(f"Error creating downloader UI: {str(e)}", 'downloader')

def _create_themed_header() -> widgets.HTML:
    """Create header dengan namespace color theme."""
    try:
        # Get color dari ui_logger_namespace untuk downloader
        namespace_color = get_namespace_color('DOWNLOAD')
        
        header_html = f"""
        <div style="background: linear-gradient(135deg, {namespace_color}, {namespace_color}CC); 
                   padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin: 0; color: white; font-weight: 600;">📥 Dataset Downloader</h3>
            <p style="margin: 8px 0 0; opacity: 0.95; font-size: 14px;">
                Download dan organize dataset untuk SmartCash training
            </p>
        </div>
        """
        return widgets.HTML(header_html)
    except Exception:
        # Fallback dengan default color
        return widgets.HTML("""
        <div style="background: linear-gradient(135deg, #96CEB4, #96CEB4CC); 
                   padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px;">
            <h3 style="margin: 0; color: white;">📥 Dataset Downloader</h3>
            <p style="margin: 8px 0 0; opacity: 0.95;">Download dataset untuk SmartCash training</p>
        </div>
        """)

def _create_status_panel(env=None) -> widgets.HTML:
    """Create status panel dengan environment info."""
    try:
        if env and hasattr(env, 'is_drive_mounted'):
            if env.is_drive_mounted:
                status_html = """
                <div style="padding: 12px; background: #e8f5e8; border-left: 4px solid #4caf50; 
                           border-radius: 4px; margin-bottom: 15px;">
                    <span style="color: #2e7d32;">✅ Google Drive terhubung - Dataset akan tersimpan permanen</span>
                </div>
                """
            else:
                status_html = """
                <div style="padding: 12px; background: #fff3cd; border-left: 4px solid #ffc107; 
                           border-radius: 4px; margin-bottom: 15px;">
                    <span style="color: #856404;">⚠️ Drive tidak terhubung - Dataset akan hilang saat restart</span>
                </div>
                """
        else:
            status_html = """
            <div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; 
                       border-radius: 4px; margin-bottom: 15px;">
                <span style="color: #1976d2;">📊 Status: Ready untuk download dataset</span>
            </div>
            """
        return widgets.HTML(status_html)
    except Exception:
        return widgets.HTML("<div style='padding:12px; background:#f8f9fa;'>📊 Status: Ready</div>")

def _create_form_fields(config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create form fields dengan improved imports."""
    try:
        from smartcash.ui.dataset.downloader.components.ui_form import create_form_fields
        return create_form_fields(config)
    except ImportError:
        return _create_basic_form_fields(config)

def _create_basic_form_fields(config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create basic form fields dengan responsive layout."""
    roboflow = config.get('roboflow', {})
    local = config.get('local', {})
    
    # Auto-detect API key
    from smartcash.ui.dataset.downloader.handlers.defaults import get_default_api_key
    api_key = roboflow.get('api_key') or get_default_api_key()
    
    return {
        # Dataset Information fields
        'workspace_field': widgets.Text(
            value=roboflow.get('workspace', 'smartcash-wo2us'),
            description='Workspace:',
            placeholder='Nama workspace Roboflow',
            layout=widgets.Layout(width='100%', max_width='100%'),
            style={'description_width': '90px'}
        ),
        'project_field': widgets.Text(
            value=roboflow.get('project', 'rupiah-emisi-2022'),
            description='Project:',
            placeholder='Nama project Roboflow',
            layout=widgets.Layout(width='100%', max_width='100%'),
            style={'description_width': '90px'}
        ),
        'version_field': widgets.Text(
            value=str(roboflow.get('version', '3')),
            description='Version:',
            placeholder='Versi dataset',
            layout=widgets.Layout(width='100%', max_width='100%'),
            style={'description_width': '90px'}
        ),
        'api_key_field': widgets.Password(
            value=api_key,
            description='API Key:',
            placeholder='Terdeteksi otomatis dari Colab secrets' if api_key else 'Masukkan API Key Roboflow',
            layout=widgets.Layout(width='100%', max_width='100%'),
            style={'description_width': '90px'}
        ),
        
        # Storage Settings fields
        'output_dir_field': widgets.Text(
            value=local.get('output_dir', '/content/data'),
            description='Output:',
            placeholder='Direktori output dataset',
            layout=widgets.Layout(width='100%', max_width='100%'),
            style={'description_width': '70px'}
        ),
        'backup_dir_field': widgets.Text(
            value=local.get('backup_dir', '/content/data/backup'),
            description='Backup:',
            placeholder='Direktori backup',
            layout=widgets.Layout(width='100%', max_width='100%'),
            style={'description_width': '70px'}
        ),
        
        # Checkbox options
        'backup_checkbox': widgets.Checkbox(
            value=local.get('backup_enabled', False),
            description='Enable backup dataset existing',
            layout=widgets.Layout(width='100%'),
            style={'description_width': 'initial'}
        )
    }

def _create_two_column_form(ui_components: Dict[str, widgets.Widget]) -> widgets.Widget:
    """Create two column form layout dengan flex design."""
    
    # Left Column: Dataset Information
    left_column = widgets.VBox([
        widgets.HTML("""
        <h4 style="margin: 0 0 15px 0; color: #495057; 
                   border-bottom: 2px solid #96CEB4; padding-bottom: 8px;
                   font-size: 16px; font-weight: 600;">
            📊 Dataset Information
        </h4>
        """),
        ui_components.get('workspace_field'),
        ui_components.get('project_field'),
        ui_components.get('version_field'),
        ui_components.get('api_key_field'),
        widgets.HTML("""
        <div style="margin-top: 15px; padding: 12px; background: #e8f5e8; 
                   border-radius: 6px; border-left: 4px solid #4caf50;">
            <small style="color: #2e7d32; line-height: 1.4;">
                <strong>🤖 Auto-Organization:</strong> Dataset akan otomatis diorganisir ke struktur 
                <code>/data/train</code>, <code>/data/valid</code>, <code>/data/test</code> 
                setelah download selesai.
            </small>
        </div>
        """),
        widgets.HTML("""
        <div style="margin-top: 10px; padding: 12px; background: #fff3cd; 
                   border-radius: 6px; border-left: 4px solid #ffc107;">
            <small style="color: #856404; line-height: 1.4;">
                <strong>🔑 API Key:</strong> Akan terdeteksi otomatis dari Colab secrets 
                (<code>ROBOFLOW_API_KEY</code>) atau environment variables.
            </small>
        </div>
        """)
    ], layout=widgets.Layout(
        width='calc(50% - 10px)',
        padding='15px',
        margin='0 5px 0 0',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#fafafa',
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    # Right Column: Storage Settings & Options
    right_column = widgets.VBox([
        widgets.HTML("""
        <h4 style="margin: 0 0 15px 0; color: #495057; 
                   border-bottom: 2px solid #96CEB4; padding-bottom: 8px;
                   font-size: 16px; font-weight: 600;">
            📁 Storage Settings
        </h4>
        """),
        ui_components.get('output_dir_field'),
        ui_components.get('backup_dir_field'),
        widgets.HTML("""
        <h4 style="margin: 20px 0 15px 0; color: #495057; 
                   border-bottom: 2px solid #96CEB4; padding-bottom: 8px;
                   font-size: 16px; font-weight: 600;">
            ⚙️ Options
        </h4>
        """),
        ui_components.get('backup_checkbox'),
        widgets.HTML("""
        <div style="margin-top: 15px; padding: 12px; background: #e3f2fd; 
                   border-radius: 6px; border-left: 4px solid #2196f3;">
            <small style="color: #1976d2; line-height: 1.4;">
                <strong>📋 Format:</strong> Dataset akan didownload dalam format YOLOv5 PyTorch 
                yang siap untuk training SmartCash model.
            </small>
        </div>
        """),
        widgets.HTML("""
        <div style="margin-top: 10px; padding: 12px; background: #f3e5f5; 
                   border-radius: 6px; border-left: 4px solid #9c27b0;">
            <small style="color: #7b1fa2; line-height: 1.4;">
                <strong>💾 Storage:</strong> Backup akan menyimpan dataset existing sebelum 
                download yang baru untuk mencegah kehilangan data.
            </small>
        </div>
        """)
    ], layout=widgets.Layout(
        width='calc(50% - 10px)',
        padding='15px',
        margin='0 0 0 5px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#fafafa',
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    # Two column container dengan flex layout
    form_container = widgets.HBox([left_column, right_column], layout=widgets.Layout(
        width='100%',
        max_width='100%',
        margin='0 0 20px 0',
        overflow='hidden',
        display='flex',
        flex_flow='row nowrap',
        justify_content='space-between',
        align_items='stretch',
        box_sizing='border-box'
    ))
    
    return form_container

def _create_action_buttons() -> Dict[str, widgets.Widget]:
    """Create action buttons dengan consistent styling."""
    button_style = widgets.Layout(width='auto', min_width='150px', height='35px', margin='5px')
    
    return {
        'download_button': widgets.Button(
            description='📥 Download Dataset',
            button_style='primary',
            tooltip='Download dataset dari Roboflow dan organize struktur',
            layout=button_style
        ),
        'check_button': widgets.Button(
            description='🔍 Check Dataset',
            button_style='info',
            tooltip='Validasi dataset yang sudah ada',
            layout=button_style
        ),
        'cleanup_button': widgets.Button(
            description='🗑️ Hapus Hasil',
            button_style='danger',
            tooltip='Hapus semua file dataset yang sudah didownload',
            layout=button_style
        )
    }

def _create_save_reset_buttons() -> Dict[str, widgets.Widget]:
    """Create save/reset buttons dengan consistent styling."""
    button_style = widgets.Layout(width='auto', min_width='100px', height='32px', margin='3px')
    
    return {
        'save_button': widgets.Button(
            description='💾 Simpan',
            button_style='success',
            tooltip='Simpan konfigurasi downloader',
            layout=button_style
        ),
        'reset_button': widgets.Button(
            description='🔄 Reset',
            button_style='',
            tooltip='Reset konfigurasi ke default',
            layout=button_style
        )
    }

def _create_log_components() -> Dict[str, widgets.Widget]:
    """Create log dan status components."""
    return {
        'log_output': widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_height='300px',
                border='1px solid #ddd',
                border_radius='4px',
                padding='8px',
                overflow='auto'
            )
        ),
        'confirmation_area': widgets.Output(
            layout=widgets.Layout(
                width='100%',
                max_height='400px',
                overflow='auto',
                display='none'
            )
        ),
        'log_accordion': widgets.Accordion(
            children=[],
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
    }

def _create_flex_main_layout(ui_components: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create main layout dengan flex design yang responsive."""
    
    # Action buttons container
    action_buttons_container = widgets.HBox([
        ui_components.get('download_button'),
        ui_components.get('check_button'),
        ui_components.get('cleanup_button')
    ], layout=widgets.Layout(
        width='100%',
        justify_content='flex-start',
        align_items='center',
        margin='15px 0',
        flex_wrap='wrap'
    ))
    
    # Save/Reset buttons container
    save_reset_container = widgets.HBox([
        ui_components.get('save_button'),
        ui_components.get('reset_button')
    ], layout=widgets.Layout(
        width='100%',
        justify_content='flex-end',
        align_items='center',
        margin='10px 0',
        flex_wrap='wrap'
    ))
    
    # Setup log accordion
    log_accordion = ui_components.get('log_accordion')
    if log_accordion:
        log_accordion.children = [ui_components.get('log_output')]
        log_accordion.set_title(0, '📋 Download Logs')
        log_accordion.selected_index = None  # Collapsed by default
    
    # Main container dengan responsive flex layout
    main_container = widgets.VBox([
        ui_components.get('header'),
        ui_components.get('status_panel'),
        ui_components.get('form_container'),
        action_buttons_container,
        save_reset_container,
        log_accordion,
        ui_components.get('confirmation_area')
    ], layout=widgets.Layout(
        width='100%',
        max_width='100%',
        padding='0',
        margin='0',
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    return main_container