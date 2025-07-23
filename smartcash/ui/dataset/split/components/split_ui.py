"""
File: smartcash/ui/dataset/split/components/split_ui.py

Dataset Split UI Components Module.

This module provides UI components for the dataset split configuration interface,
built using shared container components from the SmartCash UI library.
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
import ipywidgets as widgets
from IPython.display import display

# Standard UI components
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.decorators import handle_ui_errors
from smartcash.ui.core.errors.enums import ErrorLevel

# Local components
from .ratio_section import create_ratio_section
from .path_section import create_path_section
from .advanced_section import create_advanced_section

# Import module constants
from smartcash.ui.dataset.split.constants import (
    UI_CONFIG,
    BUTTON_CONFIG,
    DEFAULT_SPLIT_RATIOS,
    VALIDATION_RULES
)

# Re-export constants to maintain backward compatibility
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG

def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets for the split module using two-column layout like backbone.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing form widgets and container
    """
    import ipywidgets as widgets
    
    # Create form sections
    ratio_components = create_ratio_section(config)
    path_components = create_path_section(config)
    advanced_components = create_advanced_section(config)
    
    # Create two-column layout like backbone for space efficiency
    left_column = widgets.VBox([
        widgets.HTML("<h4>📊 Split Configuration</h4>"),
        ratio_components['ratio_section'],
        widgets.HTML("<h4>⚙️ Advanced Settings</h4>"),
        advanced_components['advanced_section']
    ], layout=widgets.Layout(
        width='48%',
        margin='0 1% 0 0'
    ))
    
    right_column = widgets.VBox([
        widgets.HTML("<h4>📁 Output Paths</h4>"),
        path_components['path_section'],
        widgets.HTML("<h4>ℹ️ Split Information</h4>"),
        widgets.HTML("""
        <div style='padding: 10px; background-color: #f8f9fa; border-radius: 4px; border-left: 4px solid #007bff;'>
            <p><strong>Split Ratios:</strong> Define how your dataset will be divided</p>
            <p><strong>Output Paths:</strong> Specify where split datasets will be saved</p>
            <p><strong>Advanced:</strong> Configure splitting behavior and file operations</p>
        </div>
        """)
    ], layout=widgets.Layout(
        width='48%', 
        margin='0 0 0 1%'
    ))
    
    # Create horizontal layout container
    form_ui = widgets.HBox(
        [left_column, right_column],
        layout=widgets.Layout(
            width='100%',
            justify_content='space-between',
            margin='0',
            padding='16px'
        )
    )
    
    # Combine all components
    components = {
        **ratio_components,
        **path_components,
        **advanced_components,
        'form_container': form_ui,
        'left_column': left_column,
        'right_column': right_column
    }
    
    return {
        'components': components,
        'form_container': {'container': form_ui},
        'form_rows': [form_ui]
    }


def _create_module_summary_content(components: Dict[str, Any]) -> widgets.Widget:
    """Create summary content for the module.
    
    Args:
        components: Dictionary of UI components
        
    Returns:
        Widget containing the summary content
    """
    summary = widgets.HTML(
        value="<h4>Dataset Split Summary</h4><p>Configured dataset split parameters.</p>"
    )
    return summary

def _create_module_info_box() -> widgets.Widget:
    """Create an info box with module documentation.
    
    Returns:
        Info box widget
    """
    info_text = """
    <h4>Dataset Split Help</h4>
    <p>Use this module to split your dataset into training, validation, and test sets.</p>
    <ul>
        <li><b>Ratios:</b> Define the split ratios between sets</li>
        <li><b>Paths:</b> Specify input and output directories</li>
        <li><b>Advanced:</b> Configure additional split options</li>
    </ul>
    """
    return widgets.HTML(info_text)

@handle_ui_errors(
    error_component_title=f"{UI_CONFIG['module_name']} Error",
    level=ErrorLevel.ERROR,
    show_dialog=True
)
def create_split_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create the dataset split configuration UI.
    
    This function creates a complete UI for configuring dataset splits with the following sections:
    - Ratio configuration
    - Path settings
    - Advanced options
    
    Args:
        config: Optional configuration dictionary with initial values
        **kwargs: Additional keyword arguments passed to container components
        
    Returns:
        Dictionary containing all UI components and containers
    """
    # Initialize config if not provided
    config = config or {}
    
    # Create UI components dictionary with error handling
    ui_components = {}
    
    # Create form widgets using the form container
    form_widgets = _create_module_form_widgets(config)
    components = form_widgets['components']
    
    # Create header with title and subtitle
    header = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='✂️',  # Scissors icon for split operation
        show_environment=True,
        environment='local',
        config_path='split_config.yaml'
    )
    components['header_container'] = header
    
    # Get the form container from the form widgets
    form_container = form_widgets.get('form_container')
    if form_container is None:
        # Fallback to creating a new form container if not provided
        form_container = create_form_container(
            layout_type=LayoutType.COLUMN,
            gap='16px',
            container_padding='16px',
            width='100%'
        )
        
        # Add form rows to container if available
        for row in form_widgets.get('form_rows', []):
            form_container['add_item'](row)
    
    
    # Create action container with only save/reset buttons
    action_container = create_action_container(
        buttons=[],  # No custom buttons, just using save/reset
        show_save_reset=True,
    )
    action_buttons = action_container['buttons']
    
    # Create operation container with consistent logging and status panel
    operation_container = create_operation_container(
        show_progress=False,  # No progress tracking needed for config only
        show_logs=True,
        show_dialog=True,
        log_module_name=UI_CONFIG['module_name'],
        log_height="150px",
        collapsible=True,
        collapsed=True,  # Start with logs collapsed
        hide_progress=True,  # Hide progress tracker as requested
        **kwargs
    )
    
    # Create footer with log accordion
    footer_container = create_footer_container(
        **kwargs
    )
    
    # Create main container using the new container API
    container_components = [
        # Header is a HeaderContainer object with .container attribute
        {'type': 'header', 'component': header.container, 'order': 0},
        # Form container is a dict with 'container' key
        {'type': 'form', 'component': form_container['container'], 'order': 1},
        # Action container is a dict with 'container' key
        {'type': 'action', 'component': action_container['container'], 'order': 2},
        # Operation container is a dict with 'container' key
        {'type': 'operation', 'component': operation_container['container'], 'order': 3},
        # Footer is a FooterContainer object with .container attribute
        {'type': 'footer', 'component': footer_container.container, 'order': 4}
    ]
    
    main_container = create_main_container(
        components=container_components,
        **kwargs
    )
    
    # Create summary and info box
    summary_content = _create_module_summary_content(components)
    info_box = _create_module_info_box()
    # Update components dictionary with all UI elements
    ui_components.update({
        # Containers
        'form_components': components,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        'header_container': header,
        'main_container': main_container.container,  # Use the actual widget, not the MainContainer object
        
        # UI Components
        'summary_content': summary_content,
        'info_box': info_box,
        
        # Buttons (only save/reset as requested) with safe access
        'save': action_buttons.get('save') if action_buttons else None,
        'reset': action_buttons.get('reset') if action_buttons else None,
        
        # Form Controls
        'train_ratio': components.get('train_ratio'),
        'val_ratio': components.get('val_ratio'),
        'test_ratio': components.get('test_ratio'),
        'train_dir': components.get('train_dir'),
        'val_dir': components.get('val_dir'),
        'test_dir': components.get('test_dir'),
        'create_subdirs': components.get('create_subdirs'),
        'overwrite': components.get('overwrite'),
        'seed': components.get('seed'),
        'shuffle': components.get('shuffle'),
        'stratify': components.get('stratify'),
        'use_relative_paths': components.get('use_relative_paths'),
        'preserve_structure': components.get('preserve_structure'),
        'symlink': components.get('symlink'),
        'backup': components.get('backup'),
        'backup_dir': components.get('backup_dir')
    })
    
    return ui_components
    
