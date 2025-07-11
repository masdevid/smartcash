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
        
        # 3. Create Form Container with full width and no horizontal margin
        try:
            form_container = create_form_container(
                container_margin="0",  # No margin
                container_padding="16px 0",  # Only vertical padding
                layout_kwargs={
                    'width': '100%',
                    'max_width': '100%',
                    'margin': '0',
                    'padding': '0'
                }
            )
            
            # Safely get form container and set children
            get_form_container = form_container.get('get_form_container')
            if callable(get_form_container):
                form_widget = get_form_container()
                if hasattr(form_widget, 'children'):
                    form_widget.children = (input_options,)
            
            # Store form container with fallback
            form_container_widget = form_container.get('container')
            if form_container_widget is None:
                form_container_widget = input_options  # Fallback to just show inputs
                
        except Exception as e:
            # Create a fallback form container
            error_msg = f"Failed to create form container: {str(e)}"
            form_container_widget = widgets.VBox([
                widgets.HTML(f"<div style='color: red; padding: 10px;'>{error_msg}</div>"),
                input_options  # Still try to show the input options
            ])
            
            # Log the error for debugging
            if 'log_accordion' in ui_components:
                ui_components['log_accordion'].log(f"UI Error: {error_msg}", level="error")
        
        # Store in components with fallback
        ui_components['form_container'] = form_container_widget
        
        # 4. Create Action Container with primary button for main preprocess action
        try:
            action_container = create_action_container(
                buttons=[
                    {
                        "id": "preprocess",
                        "text": BUTTON_CONFIG['preprocess']['text'],
                        "style": BUTTON_CONFIG['preprocess']['style'],
                        "tooltip": BUTTON_CONFIG['preprocess']['tooltip'],
                        "order": BUTTON_CONFIG['preprocess']['order'],
                        "primary": True  # Make this the primary button
                    },
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
            
            # Get all buttons with fallbacks
            buttons = getattr(action_container, 'buttons', {}) or {}
            primary_button = buttons.get('preprocess')
            check_btn = buttons.get('check')
            cleanup_btn = buttons.get('cleanup')
            
            # Create disabled fallback buttons if needed
            def create_disabled_button(text, tooltip="This feature is not available"):
                btn = widgets.Button(description=text, disabled=True, tooltip=tooltip)
                btn.add_class('disabled-button')
                return btn
            
            if not primary_button:
                primary_button = create_disabled_button(
                    BUTTON_CONFIG['preprocess']['text'],
                    "Preprocessing action is not available"
                )
            if not check_btn:
                check_btn = create_disabled_button(
                    BUTTON_CONFIG['check']['text'],
                    "Check action is not available"
                )
            if not cleanup_btn:
                cleanup_btn = create_disabled_button(
                    BUTTON_CONFIG['cleanup']['text'],
                    "Cleanup action is not available"
                )
            
            ui_components['action_container'] = action_container
            ui_components['preprocess_btn'] = primary_button
            ui_components['check_btn'] = check_btn
            ui_components['cleanup_btn'] = cleanup_btn
            
        except Exception as e:
            # Create fallback buttons if action container creation fails
            error_msg = f"Failed to create action container: {str(e)}"
            error_button = widgets.Button(description="Error", disabled=True, tooltip=error_msg)
            error_button.add_class('error-button')
            
            ui_components['action_container'] = {'container': widgets.HTML(f"<div style='color: red;'>{error_msg}</div>")}
            ui_components['preprocess_btn'] = error_button
            ui_components['check_btn'] = error_button
            ui_components['cleanup_btn'] = error_button
    
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
        try:
            operation_container = create_operation_container(
                title=f"📊 {UI_CONFIG['module_name']} Status",
                show_progress=True,
                show_dialog=True,
                show_logs=True,
                log_module_name=UI_CONFIG['module_name'],
                log_namespace_filter='preprocess'  # Filter logs for preprocess namespace only
            )
            
            # Safely extract components with fallbacks
            progress_tracker = operation_container.get('progress_tracker')
            log_accordion = operation_container.get('log_accordion')
            
            # Set up fallback for progress updates
            if not progress_tracker:
                progress_tracker = type('DummyProgress', (), {
                    'set_progress': lambda *_, **__: None,
                    'error': lambda *_, **__: None,
                    'warning': lambda *_, **__: None,
                    'success': lambda *_, **__: None,
                    'widget': widgets.HTML()
                })()
            
            # Set up fallback for logging
            if not log_accordion:
                log_accordion = type('DummyLogAccordion', (), {
                    'log': lambda *_, **__: None,
                    'clear': lambda: None,
                    'widget': widgets.HTML()
                })()
            
            ui_components['operation_container'] = operation_container
            ui_components['progress_tracker'] = progress_tracker
            ui_components['progress'] = progress_tracker  # Alias
            ui_components['log_accordion'] = log_accordion
            ui_components['log_output'] = log_accordion  # Alias
            
        except Exception as e:
            # Create minimal fallback UI components
            error_widget = widgets.HTML(f"<div style='color: red; padding: 10px;'>Failed to initialize operation container: {str(e)}</div>")
            dummy_component = type('DummyComponent', (), {'widget': error_widget})()
            
            ui_components['operation_container'] = {'container': error_widget}
            ui_components['progress_tracker'] = dummy_component
            ui_components['progress'] = dummy_component
            ui_components['log_accordion'] = dummy_component
            ui_components['log_output'] = dummy_component
        
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
        try:
            # Safely get all container widgets with fallbacks
            header_widget = getattr(header_container, 'container', None)
            form_widget = form_container if not hasattr(form_container, 'get') else form_container.get('container', form_container)
            action_widget = action_container.get('container', None) if hasattr(action_container, 'get') else action_container
            summary_widget = getattr(summary_container, 'container', None)
            operation_widget = operation_container.get('container', None) if hasattr(operation_container, 'get') else operation_container
            footer_widget = getattr(footer_container, 'container', None)
            
            # Create a simple fallback widget for any missing container
            def create_fallback_widget(name):
                return widgets.HTML(f"<div style='padding: 10px; border: 1px dashed #ccc;'>[{name} Container Not Available]</div>")
            
            # Create main container with fallbacks
            main_container = create_main_container(
                header_container=header_widget or create_fallback_widget("Header"),
                form_container=form_widget or create_fallback_widget("Form"),
                action_container=action_widget or create_fallback_widget("Actions"),
                summary_container=summary_widget or create_fallback_widget("Summary"),
                operation_container=operation_widget or create_fallback_widget("Operations"),
                footer_container=footer_widget or create_fallback_widget("Footer")
            )
            
            # Store main container with fallback
            main_widget = getattr(main_container, 'container', None)
            if main_widget is None:
                # Create a simple fallback UI if main container creation fails
                main_widget = widgets.VBox([
                    header_widget or create_fallback_widget("Header"),
                    form_widget or create_fallback_widget("Form"),
                    action_widget or create_fallback_widget("Actions"),
                    summary_widget or create_fallback_widget("Summary"),
                    operation_widget or create_fallback_widget("Operations"),
                    footer_widget or create_fallback_widget("Footer")
                ])
                
                # Add some error styling
                main_widget.add_class('fallback-ui')
                
                # Log the error
                if 'log_accordion' in ui_components:
                    ui_components['log_accordion'].log(
                        "Warning: Using fallback UI due to main container creation failure",
                        level="warning"
                    )
            
            ui_components['main_container'] = main_widget
            ui_components['ui'] = main_widget  # Alias for compatibility
            
        except Exception as e:
            # If everything fails, create a minimal error UI
            error_widget = widgets.HTML(
                f"<div style='color: red; padding: 20px;'>"
                f"<h3>⚠️ UI Initialization Error</h3>"
                f"<p>Failed to initialize the preprocessing UI: {str(e)}</p>"
                f"<p>Please check the logs for more details.</p>"
                f"</div>"
            )
            
            ui_components['main_container'] = error_widget
            ui_components['ui'] = error_widget  # Alias for compatibility
            
            # Try to log the error if possible
            if 'log_accordion' in ui_components:
                try:
                    ui_components['log_accordion'].log(
                        f"Critical UI Error: {str(e)}\n{traceback.format_exc()}",
                        level="error"
                    )
                except:
                    pass  # If logging fails, we can't do much more
    
        # === HELPER METHODS ===
        
        def update_ui_status(message: str, status_type: str = "info", show: bool = True) -> None:
            """
            Update the status panel with a new message.
            
            Args:
                message: New status message
                status_type: Status type (info, success, warning, error)
                show: Whether to show the status panel
            """
            if hasattr(header_container, 'update_status'):
                header_container.update_status(message, status_type, show)
            else:
                # Fallback to logging if header container doesn't support status updates
                logger = getattr(header_container, 'logger', None)
                if logger:
                    logger.log(message, level=status_type.upper())
        
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
            'update_ui_status': update_ui_status,
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