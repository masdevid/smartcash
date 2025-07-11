"""
File: smartcash/ui/dataset/preprocess/components/preprocess_ui.py
Description: Main UI components for preprocessing using standard container components
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.errors.enums import ErrorLevel

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel

# Import preprocessing components
from smartcash.ui.dataset.preprocess.components.input_options import create_preprocessing_input_options
from smartcash.ui.dataset.preprocess.constants import UI_CONFIG, BUTTON_CONFIG, PREPROCESSING_TIPS

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets for the preprocessing module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing form widgets and components
    """
    input_options = create_preprocessing_input_options(config)
    
    return {
        'components': {
            'input_options': input_options
        },
        'input_options': input_options
    }


def _create_module_summary_content(components: Dict[str, Any]) -> str:
    """Create summary content for the module.
    
    Args:
        components: Dictionary of UI components
        
    Returns:
        HTML string containing the summary content
    """
    return "<p>Current preprocessing settings and YOLO preset configuration will be displayed here.</p>"


def _create_module_info_box() -> widgets.Widget:
    """Create an info box with module documentation.
    
    Returns:
        Info box widget
    """
    tips_html = ''.join([f"<li>{tip}</li>" for tip in PREPROCESSING_TIPS])
    
    return widgets.HTML(
        value=f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8; margin: 10px 0;">
            <h5>ℹ️ Tips Preprocessing</h5>
            <ul>
                {tips_html}
            </ul>
            <div style="margin-top: 10px; padding: 8px; background-color: #e7f3ff; border-radius: 4px;">
                <strong>YOLO Presets:</strong>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    <li><strong>yolov5s/m:</strong> 640x640 - Standard untuk training cepat</li>
                    <li><strong>yolov5l:</strong> 832x832 - Higher accuracy, lebih lambat</li>
                    <li><strong>yolov5x:</strong> 1024x1024 - Maximum accuracy</li>
                </ul>
            </div>
        </div>
        """
    )


@handle_ui_errors(
    error_component_title=f"{UI_CONFIG['module_name']} Error",
    log_error=True,
    return_type=dict,
    level=ErrorLevel.ERROR,
    fail_fast=False,
    create_ui=True
)
def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create preprocessing UI using standard container components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of UI components
    """
    # Initialize config if not provided
    config = config or {}
    
    # Initialize UI components dictionary
    ui_components = {}
    
    try:
        # === CORE COMPONENTS ===
        
        # 1. Create form widgets
        form_widgets = _create_module_form_widgets(config)
        input_options = form_widgets['input_options']
        
        # 2. Create Header Container
        header_container = create_header_container(
            title=UI_CONFIG['title'],
            subtitle=UI_CONFIG['subtitle'],
            icon=UI_CONFIG['icon']
        )
        ui_components['header_container'] = header_container.container
        
        # 3. Create Form Container
        form_container = create_form_container()
        
        # Place input options in the form container
        form_container['get_form_container']().children = (input_options,)
        ui_components['form_container'] = form_container['container']
        
        # 4. Create Action Container with primary button for main preprocess action
        action_container = create_action_container(
            buttons=[
                {
                    "id": "check",
                    "text": BUTTON_CONFIG['check']['text'],
                    "style": BUTTON_CONFIG['check']['style'],
                    "tooltip": BUTTON_CONFIG['check']['tooltip'],
                    "order": BUTTON_CONFIG['check']['order']
                },
                {
                    "id": "cleanup",
                    "text": BUTTON_CONFIG['cleanup']['text'],
                    "style": BUTTON_CONFIG['cleanup']['style'],
                    "tooltip": BUTTON_CONFIG['cleanup']['tooltip'],
                    "order": BUTTON_CONFIG['cleanup']['order']
                }
            ],
            title="🚀 Preprocessing Operations",
            alignment="left",
            show_save_reset=True  # Use default save/reset buttons
        )
        
        # Configure the primary button for main preprocessing action
        primary_button = action_container['primary_button']
        if primary_button:
            primary_button.description = BUTTON_CONFIG['preprocess']['text']
            primary_button.tooltip = BUTTON_CONFIG['preprocess']['tooltip']
        
        ui_components['action_container'] = action_container
        ui_components['preprocess_btn'] = primary_button
        ui_components['check_btn'] = action_container['buttons'].get('check')
        ui_components['cleanup_btn'] = action_container['buttons'].get('cleanup')
    
        # 4. Create Summary Container (STANDARD ORDER)
        summary_content = _create_module_summary_content({})
        summary_container = create_summary_container(
            title="Preprocessing Configuration",
            theme="info",
            icon="⚙️"
        )
        summary_container.set_content(summary_content)
        ui_components['summary_container'] = summary_container.container
        
        # 5. Create Operation Container (includes progress tracker, dialogs, log accordion)
        operation_container = create_operation_container(
            title=f"📊 {UI_CONFIG['module_name']} Status",
            show_progress=True,
            show_dialog=True,
            show_logs=True,
            log_module_name=UI_CONFIG['module_name'],
            log_namespace_filter='preprocess'  # Filter logs for preprocess namespace only
        )
        ui_components['operation_container'] = operation_container
        
        # Extract components from operation container for backward compatibility
        ui_components['progress_tracker'] = operation_container['progress_tracker']
        ui_components['progress'] = ui_components['progress_tracker']  # Alias
        ui_components['log_accordion'] = operation_container['log_accordion']
        ui_components['log_output'] = ui_components['log_accordion']  # Alias
        
        # 6. Create Info Box with Tips
        info_box = _create_module_info_box()
        
        # 7. Create Footer Container with info only
        footer_container = create_footer_container(
            show_buttons=False,  # No buttons in footer
            info_box=info_box
        )
        ui_components['footer_container'] = footer_container.container
        
        # === MAIN UI ASSEMBLY ===
        
        # 8. Assemble Main Container (STANDARD ORDER)
        main_container = create_main_container(
            header_container=header_container.container,     # 1. Header Container
            form_container=form_container['container'],      # 2. Form Container  
            action_container=action_container['container'],  # 3. Action Container
            summary_container=summary_container.container,   # 4. Summary Container
            operation_container=operation_container['container'], # 5. Operation Container
            footer_container=footer_container.container      # 6. Footer Container
        )
        ui_components['main_container'] = main_container.container
        ui_components['ui'] = main_container.container  # Alias for compatibility
    
        # === HELPER METHODS ===
        
        def update_status(message: str, status_type: str = "info", show: bool = True) -> None:
            """
            Update the status panel with a new message.
            
            Args:
                message: New status message
                status_type: Status type (info, success, warning, error)
                show: Whether to show the status panel
            """
            header_container.update_status(message, status_type, show)
        
        def update_title(title: str, subtitle: Optional[str] = None) -> None:
            """
            Update the header title and subtitle.
            
            Args:
                title: New title text
                subtitle: New subtitle text (or None to keep current)
            """
            header_container.update_title(title, subtitle)
        
        def update_section(section_name: str, new_content: widgets.Widget) -> None:
            """
            Update a section of the main container.
            
            Args:
                section_name: Name of the section to update ('header', 'form', etc.)
                new_content: New widget to replace the current section
            """
            main_container.update_section(section_name, new_content)
    
        # === BUTTON MAPPING ===
        
        # Extract buttons from action container using standard approach
        preprocess_btn = action_container.get('preprocess')
        check_btn = action_container.get('check')
        cleanup_btn = action_container.get('cleanup')
        
        # Add helper methods and components to ui_components
        ui_components.update({
            # UPDATE METHODS
            'update_status': update_status,
            'update_title': update_title,
            'update_section': update_section,
            
            # BUTTONS with consistent naming for handlers
            'preprocess_btn': preprocess_btn,
            'check_btn': check_btn, 
            'cleanup_btn': cleanup_btn,
            
            # ALIASES for backward compatibility
            'preprocess_button': preprocess_btn,
            'check_button': check_btn,
            'cleanup_button': cleanup_btn,
            
            # INPUT FORM COMPONENTS
            'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
            'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None),
            'preserve_aspect_checkbox': getattr(input_options, 'preserve_aspect_checkbox', None),
            'target_splits_select': getattr(input_options, 'target_splits_select', None),
            'batch_size_input': getattr(input_options, 'batch_size_input', None),
            'validation_checkbox': getattr(input_options, 'validation_checkbox', None),
            'move_invalid_checkbox': getattr(input_options, 'move_invalid_checkbox', None),
            'invalid_dir_input': getattr(input_options, 'invalid_dir_input', None),
            'cleanup_target_dropdown': getattr(input_options, 'cleanup_target_dropdown', None),
            'backup_checkbox': getattr(input_options, 'backup_checkbox', None),
            
            # METADATA
            'module_name': UI_CONFIG['module_name'],
            'parent_module': UI_CONFIG['parent_module'],
            'ui_initialized': True,
            'api_integration': True
        })
        
        return ui_components
    
    except Exception as e:
        # The error will be handled by the @handle_ui_errors decorator
        raise


# Old function removed - replaced by _create_module_info_box()