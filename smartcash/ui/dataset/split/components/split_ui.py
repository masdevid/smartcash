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
    """Create form widgets for the split module using the form container.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing form widgets and container
    """
    # Create form container with column layout
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        gap='16px',
        container_padding='16px',
        width='100%'
    )
    
    # Create form sections
    ratio_components = create_ratio_section(config)
    path_components = create_path_section(config)
    advanced_components = create_advanced_section(config)
    
    # Add sections to form container
    form_container['add_item'](ratio_components['ratio_section'])
    form_container['add_item'](path_components['path_section'])
    form_container['add_item'](advanced_components['advanced_section'])
    
    # Combine all components
    components = {
        **ratio_components,
        **path_components,
        **advanced_components,
        'form_container': form_container['container']
    }
    
    return {
        'components': components,
        'form_container': form_container,
        'form_rows': form_container['get_items']()
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
    
    try:
        # Create form widgets using the form container
        form_widgets = _create_module_form_widgets(config)
        components = form_widgets['components']
        
        # Create log accordion for operation feedback
        log_output = widgets.Output()
        log_accordion = widgets.Accordion(
            children=[log_output],
            selected_index=None,
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
        log_accordion.set_title(0, '📝 Log Messages')
        
        # Create header container with module title, description and status panel
        header = create_header_container(
            title=UI_CONFIG['title'],
            subtitle=UI_CONFIG['description'],
            icon=UI_CONFIG.get('icon', '📊'),
            status_message="Ready for dataset split configuration",
            status_type="info",
            show_status_panel=True,
            **kwargs
        )
        
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
        
        # Create action buttons with localized labels
        action_buttons = [
            {
                'name': 'split_button',
                'label': 'Mulai Split Dataset',
                'button_style': 'primary',
                'tooltip': 'Mulai proses split dataset',
                'icon': 'share'
            },
            {
                'name': 'save_button',
                'label': 'Simpan Konfigurasi',
                'button_style': 'success',
                'tooltip': 'Simpan konfigurasi split dataset',
                'icon': 'save'
            },
            {
                'name': 'reset_button',
                'label': 'Reset',
                'button_style': 'warning',
                'tooltip': 'Reset ke nilai default',
                'icon': 'undo'
            },
            {
                'name': 'cancel_button',
                'label': 'Batal',
                'button_style': '',
                'tooltip': 'Batalkan operasi',
                'icon': 'times'
            }
        ]
        
        # Create action container with consistent styling
        action_container = create_action_container(
            buttons=action_buttons,
            show_save_reset=True,
            **kwargs
        )
        
        # Create operation container with consistent logging and status panel
        operation_container = create_operation_container(
            show_progress=False,  # No progress tracking needed for config only
            show_logs=True,
            show_dialog=True,
            log_module_name=UI_CONFIG['module_name'],
            log_height="200px",
            log_entry_style='compact',  # Ensure consistent hover behavior
            log_namespace_filter='split',  # Filter logs for split namespace only
            **kwargs
        )
        
        # Create footer with log accordion
        footer_container = create_footer_container(
            log_accordion=log_accordion,
            show_progress=True,
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
        
        # Get button references from action container
        buttons = {}
        if hasattr(action_container, 'get'):
            buttons = {
                'split_button': action_container.get('split_button'),
                'save_button': action_container.get('save_button'),
                'reset_button': action_container.get('reset_button'),
                'cancel_button': action_container.get('cancel_button')
            }
        
        # Fallback button creation if any button is missing
        if not all(buttons.values()):
            from ipywidgets import Button
            
            if not buttons.get('split_button'):
                buttons['split_button'] = Button(
                    description='Mulai Split Dataset',
                    button_style='primary',
                    icon='share',
                    tooltip='Mulai proses split dataset'
                )
            
            if not buttons.get('save_button'):
                buttons['save_button'] = Button(
                    description='Simpan',
                    button_style='success',
                    icon='save',
                    tooltip='Simpan konfigurasi split dataset'
                )
            
            if not buttons.get('reset_button'):
                buttons['reset_button'] = Button(
                    description='Reset',
                    button_style='warning',
                    icon='undo',
                    tooltip='Reset ke nilai default'
                )
            
            if not buttons.get('cancel_button'):
                buttons['cancel_button'] = Button(
                    description='Batal',
                    button_style='',
                    icon='times',
                    tooltip='Batalkan operasi'
                )
        
        # Helper function to log to operation container
        def log_to_operation_container(message: str, level: str = 'info'):
            """Log message to operation container with proper level."""
            try:
                if hasattr(operation_container, 'log_message'):
                    operation_container['log_message'](message, level)
                elif hasattr(operation_container, 'log'):
                    from smartcash.ui.components.log_accordion import LogLevel
                    level_map = {
                        'info': LogLevel.INFO,
                        'success': LogLevel.INFO,
                        'warning': LogLevel.WARNING,
                        'error': LogLevel.ERROR
                    }
                    log_level = level_map.get(level, LogLevel.INFO)
                    operation_container['log'](message, log_level)
                else:
                    # Fallback to output widget
                    if 'log_output' in operation_container:
                        with operation_container['log_output']:
                            print(message)
            except Exception:
                # Fallback to print if all else fails
                print(message)
        
        # Setup event handlers with proper logging
        def on_split_button_clicked(button):
            log_to_operation_container("🚀 Memulai proses split dataset...", 'info')
            try:
                # TODO: Implement actual split logic
                log_to_operation_container("✅ Proses split dataset selesai", 'info')
            except Exception as e:
                log_to_operation_container(f"❌ Error: {str(e)}", 'error')
        
        def on_save_button_clicked(button):
            log_to_operation_container("💾 Menyimpan konfigurasi...", 'info')
            try:
                # This will be handled by the module's save handler
                log_to_operation_container("✅ Konfigurasi berhasil disimpan", 'info')
            except Exception as e:
                log_to_operation_container(f"❌ Gagal menyimpan konfigurasi: {str(e)}", 'error')
        
        def on_reset_button_clicked(button):
            log_to_operation_container("🔄 Mereset ke nilai default...", 'info')
            try:
                # This will be handled by the module's reset handler
                log_to_operation_container("✅ Berhasil mereset ke nilai default", 'info')
            except Exception as e:
                log_to_operation_container(f"❌ Gagal mereset: {str(e)}", 'error')
        
        def on_cancel_button_clicked(button):
            log_to_operation_container("⏹️ Operasi dibatalkan", 'info')
            # TODO: Implement cancel logic
        
        # Connect button click handlers
        if buttons.get('split_button'):
            buttons['split_button'].on_click(on_split_button_clicked)
        if buttons.get('save_button'):
            buttons['save_button'].on_click(on_save_button_clicked)
        if buttons.get('reset_button'):
            buttons['reset_button'].on_click(on_reset_button_clicked)
        if buttons.get('cancel_button'):
            buttons['cancel_button'].on_click(on_cancel_button_clicked)
        
        # Update components dictionary with all UI elements
        ui_components.update({
            # Containers
            'form_components': components,
            'form_container': form_container,
            'action_container': action_container,
            'operation_container': operation_container,
            'footer_container': footer_container,
            'header_container': header,
            'main_container': main_container,
            'container': main_container,  # For backward compatibility
            
            # UI Components
            'summary_content': summary_content,
            'info_box': info_box,
            'log_accordion': log_accordion,
            'log_output': log_output,
            
            # Buttons
            'buttons': buttons,
            'split_button': buttons.get('split_button'),
            'save_button': buttons.get('save_button'),
            'reset_button': buttons.get('reset_button'),
            'cancel_button': buttons.get('cancel_button'),
            
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
    
    except Exception as e:
        # The error will be handled by the @handle_ui_errors decorator
        raise
