"""
File: smartcash/ui/dataset/downloader/components/main_ui.py
Deskripsi: Fixed main UI dengan proper component integration dan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.layout_utils import create_responsive_container, create_responsive_two_column
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.dataset.downloader.components.form_fields import create_form_fields
from smartcash.ui.dataset.downloader.components.action_buttons import create_action_buttons

def create_downloader_ui(config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Create main downloader UI dengan proper component naming."""
    env_info = _get_environment_info(env)
    header = create_header("📥 Dataset Downloader", f"Download dataset dari Roboflow dengan organized structure | {env_info}")
    
    form_components = create_form_fields(config)
    action_components = create_action_buttons()
    progress_components = create_progress_tracking_container()
    status_log_components = _create_status_log_container()
    
    form_section = create_responsive_two_column(form_components['left_panel'], form_components['right_panel'], left_width="48%", right_width="48%", gap="4%")
    action_section = create_responsive_container([action_components['container']], container_type="hbox", justify_content="center", padding="15px 0")
    progress_section = create_responsive_container([progress_components['container']], padding="10px 0")
    status_section = create_responsive_container([status_log_components['container']], padding="5px 0")
    
    main_container = create_responsive_container([header, form_section, action_section, progress_section, status_section], container_type="vbox", max_width="100%", padding="10px")
    
    return {'ui': main_container, 'main_container': main_container, 'header': header, **form_components, **action_components, **progress_components, **status_log_components}

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
    
    return {'container': container, 'status_panel': status_panel, 'log_output': log_output, 'confirmation_area': confirmation_area}