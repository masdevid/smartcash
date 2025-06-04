"""
File: smartcash/ui/dataset/download/components/download_layout.py
Deskripsi: Responsive layout assembly untuk download module dengan flex dan grid support
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.layout_utils import create_responsive_container, create_divider
from smartcash.ui.utils.header_utils import create_header

def create_download_layout(components: Dict[str, Any], config: Optional[Dict[str, Any]] = None, env_manager=None) -> Dict[str, Any]:
    """Create responsive layout dengan flex dan grid untuk download module."""
    
    # Header section dengan environment info
    storage_info = f" | Storage: {'Drive' if env_manager and env_manager.is_drive_mounted else 'Local'}"
    header = create_header("📥 Dataset Download", f"Download dataset untuk SmartCash{storage_info}")
    
    # Environment info widget
    env_info_widget = _create_environment_info_widget(env_manager)
    
    # Settings section header
    settings_header = widgets.HTML(f"<h4 style='color: #333; margin: 15px 0 10px;'>⚙️ Pengaturan Download</h4>")
    
    # Main content sections dengan responsive layout
    content_section = create_responsive_container([
        components['forms']['forms_main_container']
    ], container_type="vbox", width="100%")
    
    # Action section dengan centered layout
    action_section = create_responsive_container([
        create_divider(),
        components['action_buttons']['container'],
        components['confirmation_area']
    ], container_type="vbox", width="100%", justify_content="center")
    
    # Progress section dengan flexible height
    progress_section = create_responsive_container([
        components['progress']['container']
    ], container_type="vbox", width="100%")
    
    # Output section dengan log accordion
    output_section = create_responsive_container([
        components['log_components']['log_accordion']
    ], container_type="vbox", width="100%")
    
    # Main container assembly dengan flex layout
    main_container = widgets.VBox([
        header,
        components['status_panel'],
        env_info_widget,
        settings_header,
        content_section,
        action_section,
        progress_section,
        output_section
    ], layout=widgets.Layout(
        width='100%', max_width='100%', margin='0', padding='10px',
        display='flex', flex_direction='column', align_items='stretch',
        overflow='hidden', box_sizing='border-box', border='1px solid #ddd',
        border_radius='5px', background_color='#fff'
    ))
    
    return {
        'main_container': main_container,
        'header': header,
        'env_info_widget': env_info_widget,
        'settings_header': settings_header,
        'content_section': content_section,
        'action_section': action_section,
        'progress_section': progress_section,
        'output_section': output_section,
        'layout_type': 'responsive_flex'
    }

def _create_environment_info_widget(env_manager) -> widgets.HTML:
    """Create widget untuk environment information."""
    if not env_manager:
        return widgets.HTML("")
    
    if env_manager.is_colab and env_manager.is_drive_mounted:
        info_html = f"""
        <div style="background: #e8f5e8; border: 1px solid #4caf50; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #2e7d32;">✅ Dataset akan disimpan di Google Drive: {env_manager.drive_path}</span>
        </div>
        """
    elif env_manager.is_colab:
        info_html = """
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #856404;">⚠️ Drive tidak terhubung - dataset akan disimpan lokal (hilang saat restart)</span>
        </div>
        """
    else:
        info_html = """
        <div style="background: #e3f2fd; border: 1px solid #2196f3; border-radius: 4px; padding: 8px; margin: 5px 0;">
            <span style="color: #1565c0;">ℹ️ Environment lokal - dataset akan disimpan lokal</span>
        </div>
        """
    
    return widgets.HTML(info_html)

def create_download_mobile_layout(components: Dict[str, Any]) -> Dict[str, Any]:
    """Mobile-optimized single column layout untuk download."""
    return create_responsive_container([
        components['forms']['forms_main_container'],
        components['action_buttons']['container'],
        components['progress']['container'],
        components['log_components']['log_accordion']
    ], container_type="vbox", width="100%")

def apply_download_responsive_styling(widget: widgets.Widget, mobile_breakpoint: str = "768px") -> widgets.Widget:
    """Apply responsive CSS styling untuk download components."""
    widget.add_class('download-responsive-widget')
    return widget