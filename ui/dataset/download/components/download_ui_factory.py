"""
File: smartcash/ui/dataset/download/components/download_ui_factory.py
Deskripsi: Main UI factory untuk download module dengan responsive layout dan consolidated components
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.download.components.download_layout import create_download_layout
from smartcash.ui.dataset.download.components.download_forms import create_download_forms
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.common.environment import get_environment_manager

def create_download_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create main UI untuk download module dengan factory pattern yang clean."""
    config = config or {}
    env_manager = get_environment_manager()
    
    # Forms dan input components
    forms_components = create_download_forms(config, env_manager)
    
    # Action buttons standardized
    action_buttons = create_action_buttons(
        primary_label="Download Dataset", primary_icon="download", primary_style="success",
        secondary_buttons=[("Check Dataset", "search", "info")], cleanup_enabled=True,
        button_width="140px"
    )
    
    # Progress tracking standardized
    progress_components = create_progress_tracking_container()
    
    # Log accordion
    log_components = create_log_accordion(module_name='download', height='200px', width='100%')
    
    # Status panel dengan environment info
    status_message = f"{'🔗 Drive terhubung' if env_manager.is_drive_mounted else '⚠️ Drive tidak terhubung'} - Siap untuk download dataset"
    status_panel = create_status_panel(status_message, "info")
    
    # Confirmation area
    import ipywidgets as widgets
    confirmation_area = widgets.Output(layout=widgets.Layout(margin='5px 0', width='100%', min_height='30px'))
    
    # Layout assembly dengan responsiveness
    layout_components = create_download_layout({
        'forms': forms_components, 'action_buttons': action_buttons,
        'progress': progress_components, 'log_components': log_components,
        'status_panel': status_panel, 'confirmation_area': confirmation_area
    }, config, env_manager)
    
    # Assembly final UI components
    ui_components = {
        'ui': layout_components['main_container'],
        'main_container': layout_components['main_container'],
        'module_name': 'download',
        'parent_module': 'dataset',
        'env_manager': env_manager,
        
        # Forms mappings
        **{k: v for k, v in forms_components.items() if not k.startswith('_')},
        
        # Button mappings standardized
        'download_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'], 
        'cleanup_button': action_buttons.get('cleanup_button'),
        'save_button': forms_components.get('save_button'),
        'reset_button': forms_components.get('reset_button'),
        
        # Progress mappings dengan latest integration
        'progress_container': progress_components['container'],
        'update_progress': progress_components.get('update_progress'),
        'complete_operation': progress_components.get('complete_operation'),
        'error_operation': progress_components.get('error_operation'),
        'show_for_operation': progress_components.get('show_for_operation'),
        'tracker': progress_components.get('tracker'),
        
        # Log dan status mappings
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        'status_panel': status_panel,
        'confirmation_area': confirmation_area,
        
        # Layout components
        **{k: v for k, v in layout_components.items() if k != 'main_container'}
    }
    
    return ui_components

def get_download_critical_components() -> list:
    """Get critical components yang harus ada untuk download UI."""
    return ['ui', 'main_container', 'download_button', 'check_button', 'save_button', 'reset_button', 'log_output']

def validate_download_ui_structure(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate download UI structure dengan comprehensive checks."""
    critical = get_download_critical_components()
    missing = [comp for comp in critical if comp not in ui_components]
    
    # Check button functionality
    functional_buttons = []
    for btn_key in ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']:
        if btn_key in ui_components and hasattr(ui_components[btn_key], 'on_click'):
            functional_buttons.append(btn_key)
    
    # Check progress integration
    progress_methods = ['update_progress', 'complete_operation', 'error_operation', 'show_for_operation']
    available_progress = [method for method in progress_methods if method in ui_components]
    
    return {
        'valid': len(missing) == 0,
        'missing_components': missing,
        'functional_buttons': functional_buttons,
        'progress_integration': len(available_progress) == len(progress_methods),
        'available_progress_methods': available_progress,
        'total_components': len(ui_components),
        'structure_score': max(0, 100 - (len(missing) * 20))
    }

def get_download_component_info(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get informasi lengkap tentang download components untuk debugging."""
    validation = validate_download_ui_structure(ui_components)
    
    component_categories = {
        'forms': [k for k in ui_components.keys() if k.endswith(('_input', '_slider', '_checkbox', '_dropdown'))],
        'buttons': [k for k in ui_components.keys() if k.endswith('_button')],
        'containers': [k for k in ui_components.keys() if k.endswith(('_container', '_panel', '_accordion'))],
        'progress': [k for k in ui_components.keys() if 'progress' in k or k in ['tracker', 'update_progress']],
        'outputs': [k for k in ui_components.keys() if k.endswith(('_output', '_area'))]
    }
    
    return {
        'validation': validation,
        'categories': component_categories,
        'env_info': {
            'is_colab': ui_components.get('env_manager', {}).get('is_colab', False),
            'drive_mounted': ui_components.get('env_manager', {}).get('is_drive_mounted', False)
        },
        'module_info': {
            'module_name': ui_components.get('module_name'),
            'parent_module': ui_components.get('parent_module')
        }
    }