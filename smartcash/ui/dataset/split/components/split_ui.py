"""
Dataset Split UI Components Module.

This module provides UI components for the dataset split configuration interface,
built using shared container components from the SmartCash UI library.
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container

# Import local components
from .ratio_section import create_ratio_section
from .path_section import create_path_section
from .advanced_section import create_advanced_section


def create_split_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create split UI components following the container-based pattern.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components and their references
    """
    if config is None:
        config = {}
    
    # Initialize components dictionary
    components = {}
    
    # Create form sections
    ratio_components = create_ratio_section(config)
    path_components = create_path_section(config)
    advanced_components = create_advanced_section(config)
    
    # Combine all components
    components.update(ratio_components)
    components.update(path_components)
    components.update(advanced_components)
    
    # Create form rows for the container
    form_rows = [
        ratio_components["ratio_section"],
        path_components["path_section"],
        advanced_components["advanced_section"]
    ]
    
    # Create log accordion for status messages
    log_output = widgets.Output()
    log_accordion = widgets.Accordion(children=[log_output], selected_index=None)
    log_accordion.set_title(0, 'Log Messages')
    
    # Add log components
    components.update({
        'log_output': log_output,
        'log_accordion': log_accordion
    })
    
    # Add log accordion to form rows
    form_rows.append(log_accordion)
    
    # Create header container
    header_container = create_header_container(
        title="Dataset Split Configuration",
        description="Configure how to split your dataset into train/validation/test sets"
    )
    
    # Create form container
    form_container = create_form_container(
        form_rows=form_rows,
        layout_type=LayoutType.COLUMN,  # Using COLUMN instead of SINGLE_COLUMN
        **kwargs
    )
    
    # Create action container with default save/reset buttons
    action_buttons = create_action_container(
        buttons=[],  # No additional buttons, use defaults
        show_save_reset=True,  # Use default save/reset buttons
        **kwargs
    )
    
    # Add action buttons to components (access through action_container)
    components.update({
        'save_button': action_buttons['action_container'].save_button,
        'reset_button': action_buttons['action_container'].reset_button
    })
    
    # Create footer with log accordion
    footer_container = create_footer_container(
        log_accordion=log_accordion,
        show_progress=False  # No progress tracking for config-only module
    )
    
    # Create main layout
    main_container = create_main_container(
        header=header_container,
        body=widgets.VBox([form_container['container'], action_buttons['container']]),
        footer=footer_container,
        **kwargs
    )
    
    # Update components with container references
    components.update({
        'main_container': main_container,
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_buttons,
        'footer_container': footer_container,
        'form_rows': form_rows,
        'form_components': components
    })
    
    return components
