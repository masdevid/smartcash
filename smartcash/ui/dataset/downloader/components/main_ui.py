"""
File: smartcash/ui/dataset/downloader/components/main_ui.py
Deskripsi: Fixed main UI dengan semua komponen yang diperlukan dikembalikan
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.dataset.downloader.components.form_fields import create_form_fields
from smartcash.ui.dataset.downloader.components.action_buttons import create_action_buttons

def create_downloader_ui(config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Create main downloader UI dengan semua komponen yang diperlukan."""
    env_info = _get_environment_info(env)
    header = create_header("📥 Dataset Downloader", f"Download dataset dari Roboflow dengan organized structure | {env_info}")
    
    # Get all components
    form_components = create_form_fields(config)
    action_components = create_action_buttons()
    progress_components = create_progress_tracking_container()
    status_log_components = _create_status_log_container()
    
    # Create layout sections
    form_section = create_responsive_two_column(form_components['left_panel'], form_components['right_panel'], left_width="48%", right_width="48%", gap="4%")
    action_section = create_responsive_container([action_components['container']], container_type="hbox", justify_content="center", padding="15px 0")
    progress_section = create_responsive_container([progress_components['container']], padding="10px 0")
    status_section = create_responsive_container([status_log_components['container']], padding="5px 0")
    
    # Main container
    main_container = create_responsive_container([header, form_section, action_section, progress_section, status_section], container_type="vbox", max_width="100%", padding="10px")
    
    # Extract advanced options components dari accordion
    advanced_components = _extract_advanced_components(form_components.get('advanced_accordion'))
    
    # Pastikan semua komponen kritis tersedia
    critical_components = {
        # Form fields
        'workspace_field': form_components.get('workspace_field'),
        'project_field': form_components.get('project_field'),
        'version_field': form_components.get('version_field'),
        'api_key_field': form_components.get('api_key_field'),
        # Action buttons
        'download_button': action_components.get('download_button'),
        'validate_button': action_components.get('validate_button'),
    }
    
    # Periksa komponen kritis
    missing_components = [k for k, v in critical_components.items() if v is None]
    if missing_components:
        print(f"⚠️ Komponen kritis tidak tersedia: {', '.join(missing_components)}")
    
    # Return all components dengan merge yang benar
    return {
        'ui': main_container,
        'main_container': main_container,
        'header': header,
        # Form fields - critical components
        'workspace_field': form_components.get('workspace_field'),
        'project_field': form_components.get('project_field'),
        'version_field': form_components.get('version_field'),
        'api_key_field': form_components.get('api_key_field'),
        'format_dropdown': form_components.get('format_dropdown'),
        # Checkbox options
        'validate_checkbox': form_components.get('validate_checkbox'),
        'organize_checkbox': form_components.get('organize_checkbox'),
        'backup_checkbox': form_components.get('backup_checkbox'),
        'progress_checkbox': form_components.get('progress_checkbox'),
        'detailed_progress_checkbox': form_components.get('detailed_progress_checkbox'),
        # Action buttons - critical components
        'download_button': action_components.get('download_button'),
        'validate_button': action_components.get('validate_button'),
        'quick_validate_button': action_components.get('quick_validate_button'),
        'save_button': action_components.get('save_button'),
        'reset_button': action_components.get('reset_button'),
        # Progress components
        **progress_components,
        # Status components
        **status_log_components,
        # Advanced options
        **advanced_components,
        # Panels
        'left_panel': form_components.get('left_panel'),
        'right_panel': form_components.get('right_panel'),
        'info_panel': form_components.get('info_panel'),
        'env_status': form_components.get('env_status'),
        'advanced_accordion': form_components.get('advanced_accordion'),
        # Additional components from action buttons
        'api_status': action_components.get('api_status') if 'api_status' in action_components else None,
        'dataset_status': action_components.get('dataset_status') if 'dataset_status' in action_components else None,
        'connection_status': action_components.get('connection_status') if 'connection_status' in action_components else None,
        'refresh_status_button': action_components.get('refresh_status_button') if 'refresh_status_button' in action_components else None,
        'clear_logs_button': action_components.get('clear_logs_button') if 'clear_logs_button' in action_components else None
    }

def _get_environment_info(env) -> str:
    """Get environment information string."""
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        return "🔗 Colab + Drive" if env_manager.is_colab and env_manager.is_drive_mounted else "⚠️ Colab (Local)" if env_manager.is_colab else "💻 Local Environment"
    except Exception:
        return "🌐 Environment"

def _create_status_log_container() -> Dict[str, Any]:
    """Create status dan log container."""
    status_panel = widgets.HTML(value="<div style='padding:8px; background:#f8f9fa; border-radius:4px; color:#666;'>📋 Ready untuk download dataset</div>", layout=widgets.Layout(width='100%', margin='5px 0'))
    log_output = widgets.Output(layout=widgets.Layout(width='100%', max_height='200px', border='1px solid #ddd', border_radius='4px', overflow='auto', margin='5px 0', padding='5px'))
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%', margin='5px 0', min_height='30px'))
    
    container = create_responsive_container([status_panel, log_output, confirmation_area], container_type="vbox", padding="0")
    
    return {
        'container': container,
        'status_panel': status_panel,
        'log_output': log_output,
        'confirmation_area': confirmation_area,
        'status': log_output  # Alias untuk compatibility
    }

def _extract_advanced_components(accordion: widgets.Accordion) -> Dict[str, Any]:
    """Extract components dari advanced accordion."""
    if not accordion or not hasattr(accordion, 'children') or len(accordion.children) == 0:
        return {
            'retry_field': None,
            'timeout_field': None,
            'chunk_size_field': None
        }
    
    # Get container dari accordion
    container = accordion.children[0] if accordion.children else None
    if not container or not hasattr(container, 'children'):
        return {
            'retry_field': None,
            'timeout_field': None,
            'chunk_size_field': None
        }
    
    # Extract fields berdasarkan urutan
    components = {}
    field_names = ['retry_field', 'timeout_field', 'chunk_size_field']
    
    for i, field_name in enumerate(field_names):
        if i < len(container.children):
            components[field_name] = container.children[i]
        else:
            components[field_name] = None
    
    return components