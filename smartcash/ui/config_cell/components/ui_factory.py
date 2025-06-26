"""
UI Component Factory for Config Cell

This module provides factory functions for creating standardized UI components
used throughout the config cell interface.
"""
from typing import Dict, Any, Optional, List
import ipywidgets as widgets

from smartcash.common.logger import get_logger
from smartcash.ui.components import (
    create_header,
    create_status_panel,
    create_info_accordion,
    create_log_accordion,
)
from smartcash.ui.config_cell.constants import StatusType
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler

__all__ = [
    'create_config_summary_panel',
    'create_log_components',
    'create_info_components',
    'create_config_cell_ui',
    'create_container'
]

logger = get_logger(__name__)

def create_container(title: str = None, container_id: str = None) -> Dict[str, Any]:
    """Create a styled container for grouping related UI components.
    
    Args:
        title: Optional title displayed at the top of the container
        container_id: Optional unique identifier for the container
        
    Returns:
        Dictionary containing:
        - 'container': The VBox container widget
        - 'content_area': The VBox where child components should be added
    """
    container = widgets.VBox(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #e0e0e0',
            border_radius='4px',
            padding='10px',
            margin='5px 0',
            overflow='hidden'
        )
    )
    
    # Add ID for easier debugging and testing
    if container_id:
        container.add_class(f'container-{container_id}')
    
    content_area = widgets.VBox(
        layout={
            'margin': '5px 0 0 0',
            'width': '100%'
        }
    )
    
    if title:
        header = widgets.HTML(
            f"<h3 style='margin: 0 0 10px 0; color: #333;'>{title}</h3>"
        )
        container.children = [header, content_area]
    else:
        container.children = [content_area]
    
    return {
        'container': container,
        'content_area': content_area
    }


def create_config_summary_panel() -> widgets.VBox:
    """Create a config summary panel.
    
    Returns:
        widgets.VBox: The configured summary panel
    """
    return widgets.VBox(
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            padding='10px',
            border='1px dashed #e0e0e0',
            border_radius='4px',
            display='none'  # Hidden by default
        )
    )

def create_log_components(module_name: str) -> Dict[str, Any]:
    """Create log components for the UI.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Dict[str, Any]: Dictionary containing log components
    """
    return create_log_accordion(
        module_name=module_name,
        height='200px',
        width='100%',
        show_timestamps=True,
        auto_scroll=True,
        enable_deduplication=True
    )


def _get_default_info_content(module_name: str) -> str:
    """Generate default info content for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        str: HTML content for the info box
    """
    return (
        f"<div class='info-content'>"
        f"<h4>{module_name.replace('_', ' ').title()}</h4>"
        f"<p>This panel displays configuration options and information about the {module_name} module.</p>"
        "<p>Use the controls above to customize your settings.</p>"
        "</div>"
    )

def create_info_components(module_name: str) -> Dict[str, Any]:
    """Create info components for the UI.
    
    This is the main entry point for creating info components. It handles both
    custom module info and fallback to default content.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Dict[str, Any]: Dictionary containing info components
    """
    try:
        # Try to import module-specific info
        module = __import__(f'smartcash.ui.info_boxes.{module_name.lower()}_info', fromlist=[''])
        if info_func := getattr(module, f'get_{module_name.lower()}_info', None):
            info_content = info_func()
        else:
            info_content = _get_default_info_content(module_name)
    except (ImportError, AttributeError):
        info_content = _get_default_info_content(module_name)
    except Exception as e:
        logger.warning(f"Error loading module info for {module_name}: {e}")
        info_content = _get_default_info_content(module_name)
    
    return create_info_accordion(
        title=f"{module_name.replace('_', ' ').title()} Information",
        content=info_content,
        icon='info',
        open_by_default=False
    )


def create_config_cell_ui(
    module_name: str,
    handler: ConfigCellHandler,
    parent_module: Optional[str] = None
) -> Dict[str, Any]:
    """Create and configure all UI components for the config cell.
    
    Args:
        module_name: Name of the module
        handler: Config handler instance
        parent_module: Optional parent module name
        
    Returns:
        Dict[str, Any]: Dictionary containing all UI components
    """
    ui_components: Dict[str, Any] = {}
    
    try:
        # Create child UI components
        child_components = handler.create_ui_components(handler.config)
        
        # Get child content container with fallback
        child_content = child_components.get('container', widgets.VBox())
        
        # Setup header with overridable defaults
        header_title = child_components.get(
            'header_title',
            module_name.replace('_', ' ').title()
        )
        header_description = child_components.get(
            'header_description',
            f"Configuration for {module_name}"
        )
        header_icon = child_components.get('header_icon', "⚙️")
        
        # Create header and status panel
        ui_components.update({
            'header': create_header(header_title, header_description, header_icon),
            'status_panel': create_status_panel("Ready", StatusType.INFO),
            'child_components': child_components,
            'child_content': child_content,
            'config_summary_panel': create_config_summary_panel()
        })
        
        # Create log components
        log_components = create_log_components(module_name)
        ui_components.update({
            'log_output': log_components['log_output'],
            'log_accordion': log_components['log_accordion'],
            'log_entries_container': log_components.get('entries_container')
        })
        
        # Create info components
        info_components = create_info_components(module_name)
        ui_components.update({
            'info_box': info_components['content'],
            'info_accordion': info_components['accordion']
        })
        
        # Create main container
        ui_components['container'] = widgets.VBox(
            [
                ui_components['header'],
                ui_components['status_panel'],
                ui_components['config_summary_panel'],
                child_content,
                ui_components['log_accordion'],
                ui_components['info_accordion']
            ],
            layout=widgets.Layout(
                width='100%',
                padding='15px',
                border='1px solid #e0e0e0',
                border_radius='8px',
                margin='10px 0',
                display='flex',
                flex_flow='column',
                align_items='stretch'
            )
        )
        
        return ui_components
        
    except Exception as e:
        logger.error(f"Failed to create UI components for {module_name}: {str(e)}")
        raise
