"""
File: smartcash/ui/setup/env_config/components/ui_components.py
Deskripsi: Komponen UI untuk environment configuration dengan layout lengkap
"""

import ipywidgets
from typing import Dict, Any

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel

# Import env_config specific components
from smartcash.ui.setup.env_config.components.setup_summary import create_setup_summary
from smartcash.ui.setup.env_config.components.env_info_panel import create_env_info_panel
from smartcash.ui.setup.env_config.components.tips_panel import create_tips_requirements

def create_env_config_ui() -> Dict[str, Any]:
    """
    🎨 Buat komponen UI untuk environment configuration
    
    Returns:
        Dictionary berisi semua komponen UI
    """
    # Initialize components dictionary
    ui_components = {}
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="🔧 Environment Setup",
        subtitle="Konfigurasi lingkungan untuk SmartCash YOLOv5-EfficientNet"
    )
    ui_components['header_container'] = header_container
    
    # 2. Create Action Container with setup button
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "setup",
                "text": "▶️ Setup Environment",
                "style": "primary",
                "order": 1
            }
        ],
        title="🔧 Environment Setup Actions",
        alignment="center"
    )
    
    # Get setup button from action container and connect it to handlers
    setup_button = action_container['buttons']['setup']
    
    # Store the button for later handler connection
    ui_components['setup_button'] = setup_button
    
    # Create a placeholder for the setup handler function that will be replaced
    # by EnvConfigHandler.handle_setup_button_click during initialization
    def placeholder_setup_handler(btn):
        # Get the log accordion if available
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'append_log'):
            log_output.append_log("Setup button clicked - handler not yet connected", "info")
        else:
            # Fallback to print if log not available
            print("Setup button clicked - handler not yet connected")
    
    # Attach the placeholder handler - will be replaced by the real handler during initialization
    setup_button.on_click(placeholder_setup_handler)
    
    # Create form container without save/reset buttons
    form_components = create_form_container(
        show_buttons=False,
        container_margin="15px 0",
        container_padding="10px"
    )
    
    # Add the action container to the form container
    form_components['form_container'].children = (action_container['container'],)
    
    # Store components in ui_components
    ui_components['form_container'] = form_components['container']
    ui_components['action_container'] = action_container
    ui_components['setup_button'] = setup_button
    ui_components['confirmation_area'] = action_container['dialog_area']
    ui_components['show_dialog'] = action_container['show_dialog']
    ui_components['show_info'] = action_container['show_info']
    ui_components['clear_dialog'] = action_container['clear_dialog']
    ui_components['is_dialog_visible'] = action_container['is_dialog_visible']
    
    # 3. Create Summary Container for setup summary
    setup_summary = create_setup_summary()
    summary_container = create_summary_container(
        theme="info",
        icon="📋"
    )
    summary_container.set_content(setup_summary.value)
    ui_components['setup_summary'] = setup_summary
    ui_components['summary_container'] = summary_container
    
    # 4. Create Progress Tracker
    progress_tracker = ProgressTracker()
    progress_tracker.show(level=ProgressLevel.DUAL)
    ui_components['progress_tracker'] = progress_tracker
    
    # 5. Environment Info Panel
    env_info_panel = create_env_info_panel()
    ui_components['env_info_panel'] = env_info_panel
    
    # 6. Tips & Requirements
    tips_requirements = create_tips_requirements()
    ui_components['tips_requirements'] = tips_requirements
    
    # 7. Create footer container with logs (no tips panel)
    footer_container = create_footer_container(
        show_progress=False,
        show_logs=True,
        show_info=False,
        show_tips=False,  # Explicitly disable tips panel
        log_module_name="Environment"
    )
    ui_components['footer_container'] = footer_container
    
    # Use the modern LogAccordion implementation directly
    from smartcash.ui.components.log_accordion import LogAccordion
    
    # Get the log accordion from the footer container
    # The footer container already creates a log accordion internally
    log_accordion = footer_container.log_accordion
    
    # Expose log components via footer container for easy access
    ui_components['log_accordion'] = log_accordion
    ui_components['log_output'] = footer_container.log_output
    
    # Add the log method for convenience
    ui_components['log'] = footer_container.log
    
    # 8. Create the final UI layout using main_container
    
    # Fix status panel width to prevent horizontal scrollbar
    if hasattr(header_container, 'status_panel') and header_container.status_panel is not None:
        # Fix the status panel HTML element directly
        if hasattr(header_container.status_panel, 'children') and len(header_container.status_panel.children) > 0:
            status_html = header_container.status_panel.children[0]
            # Apply custom CSS to fix the status panel styling
            if hasattr(status_html, 'value') and isinstance(status_html.value, str):
                # Extract the existing HTML content
                html_content = status_html.value
                # Replace the inline style with a style that doesn't overflow
                html_content = html_content.replace(
                    'padding: 8px 12px;', 
                    'padding: 8px 12px; word-wrap: break-word; white-space: normal;'
                )
                status_html.value = html_content
            
        # Also fix the container
        header_container.status_panel.layout.width = '100%'
        header_container.status_panel.layout.max_width = '100%'
    
    # Fix summary container title duplication and vertical layout
    if hasattr(setup_summary, 'title_widget') and hasattr(setup_summary, 'value'):
        # Create a new widget without the title
        setup_content = setup_summary.value
        summary_container.set_content(setup_content)
    
    # Ensure summary container has vertical layout
    if hasattr(summary_container, 'container'):
        summary_container.container.layout.flex_flow = 'column'
        summary_container.container.layout.align_items = 'stretch'
        if hasattr(summary_container.container, 'children'):
            for child in summary_container.container.children:
                if hasattr(child, 'layout'):
                    child.layout.width = '100%'
    
    # Make sure log accordion is inside the footer container and remove duplicates
    if 'log_accordion' in ui_components and ui_components['log_accordion'] is not None:
        # Check if it's already in the footer container
        if footer_container.container.children and ui_components['log_accordion'] not in footer_container.container.children:
            # Add it to the footer if needed
            footer_children = list(footer_container.container.children)
            footer_children.append(ui_components['log_accordion'])
            footer_container.container.children = tuple(footer_children)
        
        # Ensure we only have one reference to the log accordion
        # This prevents the mysterious duplicate accordion from appearing
        ui_components['log_accordion'] = footer_container.log_accordion
        ui_components['log_output'] = footer_container.log_accordion
        ui_components['log_components'] = footer_container.log_accordion
        
        # Make sure the log accordion is visible
        if hasattr(footer_container.log_accordion, 'selected_index'):
            footer_container.log_accordion.selected_index = 0  # Expand first accordion item
    
    # Create the main container with proper layout
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=form_components['container'],
        action_container=action_container['container'],
        footer_container=footer_container.container
    )
    
    # Create a list of components that should be in the main container
    all_components = [
        header_container.container,
        form_components['container'],
        summary_container.container,
        env_info_panel,
        tips_requirements,
        footer_container.container
    ]
    
    # Filter out any None components
    all_components = [c for c in all_components if c is not None]
    
    # Replace the main container's children with our custom ordered components
    main_container.container.children = all_components
    
    # Store the main UI container
    ui_components['ui'] = main_container.container
    ui_components['main_container'] = main_container
    
    return ui_components