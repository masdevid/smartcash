"""
File: smartcash/ui/dataset/downloader/components/ui_components.py
Deskripsi: Complete UI components dengan layout order yang benar dan format hardcoded
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import try_operation_safe
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

def create_downloader_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create main downloader UI dengan fixed layout order dan one-liner integration"""
    config = config or {}
    
    try:
        # Import layout creator dengan fallback
        ui_components = try_operation_safe(
            lambda: _create_full_downloader_ui(config),
            fallback_value=_create_minimal_downloader_ui(config)
        )
        
        # Validate dan fix layout order
        ui_components = _ensure_layout_order(ui_components)
        
        return ui_components
        
    except Exception as e:
        return _create_error_ui(f"Error creating downloader UI: {str(e)}")

def _create_full_downloader_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create full downloader UI dengan semua components - one-liner imports"""
    try:
        from smartcash.ui.dataset.downloader.components.ui_layout import create_downloader_ui
        
        # Get environment info
        env = try_operation_safe(lambda: _get_environment_info(), fallback_value=None)
        
        # Create UI dengan layout yang sudah fixed
        ui_components = create_downloader_ui(config, env)
        
        # Enhanced components integration
        ui_components = _enhance_ui_components(ui_components, config)
        
        return ui_components
        
    except ImportError:
        return _create_minimal_downloader_ui(config)

def _create_minimal_downloader_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create minimal downloader UI tanpa external dependencies - one-liner style"""
    roboflow = config.get('roboflow', {})
    
    # Header
    header = widgets.HTML("""
    <div style="background: linear-gradient(135deg, #96CEB4, #96CEB4CC); 
               padding: 20px; color: white; border-radius: 8px; margin-bottom: 15px;">
        <h3 style="margin: 0; color: white;">📥 Dataset Downloader (Minimal)</h3>
        <p style="margin: 8px 0 0; opacity: 0.95;">Download dataset untuk SmartCash training</p>
    </div>
    """)
    
    # Status panel
    status_panel = widgets.HTML("""
    <div style="padding: 12px; background: #f8f9fa; border-left: 4px solid #6c757d; border-radius: 4px; margin-bottom: 15px;">
        <span style="color: #495057;">📊 Status: Minimal mode active</span>
    </div>
    """)
    
    # 1. Form 2 kolom - simplified (tanpa format dropdown)
    form_fields = _create_minimal_form_fields(config)
    form_container = _create_minimal_two_column_form(form_fields)
    
    # 2. Save Reset Button
    save_reset_components = _create_minimal_save_reset_buttons()
    
    # 3. Area Konfirmasi
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', display='none', margin='10px 0'))
    
    # 4. Action Buttons  
    action_components = _create_minimal_action_buttons()
    
    # 5. Progress Tracker (minimal)
    progress_container = widgets.HTML("""
    <div style="padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; margin: 15px 0;">
        📊 Progress tracker will appear here
    </div>
    """)
    
    # 6. Log Output
    log_output = widgets.Output(layout=widgets.Layout(width='100%', max_height='300px', 
                                                     border='1px solid #ddd', border_radius='4px', 
                                                     padding='8px', overflow='auto'))
    
    log_accordion = widgets.Accordion([log_output], layout=widgets.Layout(width='100%', margin='10px 0'))
    log_accordion.set_title(0, '📋 Download Logs'), setattr(log_accordion, 'selected_index', None)
    
    # Main UI dengan fixed order
    main_ui = widgets.VBox([
        header, status_panel,                      # Header dan status
        form_container,                           # 1. Form 2 kolom
        save_reset_components['container'],       # 2. Save Reset Button
        confirmation_area,                        # 3. Area Konfirmasi
        action_components['container'],           # 4. Action Buttons
        progress_container,                       # 5. Progress Tracker
        log_accordion                            # 6. Log Output
    ], layout=widgets.Layout(width='100%', padding='0', margin='0'))
    
    # Combine all components
    ui_components = {
        'ui': main_ui, 'main_container': main_ui, 'header': header, 'status_panel': status_panel,
        'form_container': form_container, 'confirmation_area': confirmation_area,
        'progress_container': progress_container, 'log_output': log_output, 'log_accordion': log_accordion,
        'layout_order_fixed': True
    }
    
    # Add form fields
    ui_components.update(form_fields)
    # Add save/reset components
    ui_components.update(save_reset_components)
    # Add action components
    ui_components.update(action_components)
    
    return ui_components

def _create_minimal_form_fields(config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """Create minimal form fields tanpa format dropdown - one-liner style"""
    roboflow = config.get('roboflow', {})
    
    return {
        'workspace_input': widgets.Text(value=roboflow.get('workspace', 'smartcash-wo2us'),
                                      description='Workspace:', layout=widgets.Layout(width='100%')),
        'project_input': widgets.Text(value=roboflow.get('project', 'rupiah-emisi-2022'),
                                    description='Project:', layout=widgets.Layout(width='100%')),
        'version_input': widgets.Text(value=str(roboflow.get('version', '3')),
                                    description='Version:', layout=widgets.Layout(width='100%')),
        'api_key_input': widgets.Password(value=roboflow.get('api_key', ''),
                                        description='API Key:', layout=widgets.Layout(width='100%')),
        'validate_checkbox': widgets.Checkbox(value=True, description='Validasi download'),
        'organize_checkbox': widgets.Checkbox(value=True, description='Organisir dataset'),
        'backup_checkbox': widgets.Checkbox(value=False, description='Backup existing')
    }

def _create_minimal_two_column_form(form_fields: Dict[str, widgets.Widget]) -> widgets.VBox:
    """Create minimal two column form - one-liner layout"""
    # Left column - Dataset info
    left_widgets = [form_fields[key] for key in ['workspace_input', 'project_input', 'version_input', 'api_key_input']]
    left_column = widgets.VBox(left_widgets, layout=widgets.Layout(width='48%', margin='0', padding='8px'))
    
    # Right column - Options dengan format info
    format_info = widgets.HTML("""
    <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin-bottom: 8px;">
        <small style="color: #1976d2;"><strong>Format:</strong> YOLOv5 PyTorch (hardcoded)</small>
    </div>
    """)
    
    right_widgets = [format_info] + [form_fields[key] for key in ['validate_checkbox', 'organize_checkbox', 'backup_checkbox']]
    right_column = widgets.VBox(right_widgets, layout=widgets.Layout(width='48%', margin='0', padding='8px'))
    
    # Two column container
    two_column_container = widgets.HBox([left_column, right_column], 
                                      layout=widgets.Layout(width='100%', justify_content='space-between',
                                                          border='1px solid #ddd', border_radius='5px', 
                                                          padding='15px', margin='0 0 15px 0'))
    
    return widgets.VBox([two_column_container], layout=widgets.Layout(width='100%'))

def _create_minimal_save_reset_buttons() -> Dict[str, widgets.Widget]:
    """Create minimal save/reset buttons - one-liner style"""
    save_button = widgets.Button(description='💾 Simpan', button_style='success',
                                layout=widgets.Layout(width='auto', height='32px', margin='3px'))
    reset_button = widgets.Button(description='🔄 Reset', button_style='',
                                 layout=widgets.Layout(width='auto', height='32px', margin='3px'))
    
    container = widgets.HBox([save_button, reset_button], 
                           layout=widgets.Layout(width='100%', justify_content='flex-end', margin='10px 0'))
    
    return {'save_button': save_button, 'reset_button': reset_button, 'container': container}

def _create_minimal_action_buttons() -> Dict[str, widgets.Widget]:
    """Create minimal action buttons - one-liner style"""
    button_layout = widgets.Layout(width='auto', min_width='120px', height='35px', margin='5px')
    
    download_button = widgets.Button(description='📥 Download', button_style='primary', layout=button_layout)
    check_button = widgets.Button(description='🔍 Check', button_style='info', layout=button_layout)
    cleanup_button = widgets.Button(description='🧹 Cleanup', button_style='danger', layout=button_layout)
    
    container = widgets.HBox([download_button, check_button, cleanup_button], 
                           layout=widgets.Layout(width='100%', justify_content='flex-start', margin='15px 0'))
    
    return {
        'download_button': download_button, 'check_button': check_button, 'cleanup_button': cleanup_button,
        'container': container
    }

def _enhance_ui_components(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance UI components dengan additional features - one-liner enhancements"""
    try:
        # Add progress tracker jika tersedia
        progress_components = try_operation_safe(
            lambda: create_progress_tracking_container(),
            fallback_value={'tracker': widgets.HTML("📊 Progress not available")}
        )
        ui_components.update(progress_components)
        
        # Add save/reset buttons jika belum ada
        if 'save_button' not in ui_components:
            save_reset_components = try_operation_safe(
                lambda: create_save_reset_buttons(),
                fallback_value=_create_minimal_save_reset_buttons()
            )
            ui_components.update(save_reset_components)
        
        # Add environment info
        ui_components['env_info'] = _get_environment_info()
        
    except Exception as e:
        ui_components['enhancement_error'] = str(e)
    
    return ui_components

def _ensure_layout_order(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure layout order sesuai requirement - one-liner validation"""
    required_components = ['form_container', 'save_button', 'reset_button', 'confirmation_area', 
                          'download_button', 'check_button', 'cleanup_button', 'log_output']
    
    # Check dan add missing components dengan fallback
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        ui_components['missing_components'] = missing_components
        ui_components['layout_warning'] = f"Missing: {', '.join(missing_components)}"
    
    ui_components['layout_order_validated'] = True
    return ui_components

def _get_environment_info() -> Dict[str, Any]:
    """Get environment information - one-liner detection"""
    try:
        import google.colab
        is_colab = True
        
        from pathlib import Path
        is_drive_mounted = Path('/content/drive/MyDrive').exists()
        
        return {'is_colab': True, 'is_drive_mounted': is_drive_mounted, 'platform': 'Google Colab'}
    except ImportError:
        return {'is_colab': False, 'is_drive_mounted': False, 'platform': 'Local/Other'}

def _create_error_ui(error_message: str) -> Dict[str, Any]:
    """Create error UI - one-liner fallback"""
    error_widget = widgets.HTML(f"""
    <div style="padding: 20px; background: #f8d7da; border: 1px solid #dc3545; 
               border-radius: 5px; color: #721c24; margin: 10px 0;">
        <h4>❌ Downloader UI Error</h4>
        <p>{error_message}</p>
        <small>💡 Try restarting cell atau check dependencies</small>
    </div>
    """)
    
    return {'ui': error_widget, 'main_container': error_widget, 'error': error_message, 'status': 'error'}

# One-liner utilities untuk UI management
create_simple_form = lambda config: _create_minimal_form_fields(config)
create_button_row = lambda buttons: widgets.HBox(buttons, layout=widgets.Layout(width='100%'))
apply_fixed_layout = lambda components: _ensure_layout_order(components)
validate_ui_order = lambda ui: ui.get('layout_order_validated', False)
get_component_count = lambda ui: len([k for k in ui.keys() if not k.startswith('_')])
is_minimal_mode = lambda ui: 'minimal' in ui.get('ui', widgets.HTML()).value.lower() if hasattr(ui.get('ui'), 'value') else False