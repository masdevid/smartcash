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
from smartcash.ui.components import create_log_accordion
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
    
    # Get setup button from action container
    setup_button = action_container['buttons']['setup']
    
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
        title="Setup Summary",
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
    
    # 7. Create footer container with logs
    footer_container = create_footer_container(
        show_progress=False,
        show_logs=True,
        show_info=False,
        show_tips=False,
        log_module_name="Environment"
    )
    ui_components['footer_container'] = footer_container

    # Expose log components via footer container
    if footer_container.log_accordion:
        ui_components['log_accordion'] = footer_container.log_accordion
        if getattr(footer_container, 'log_output', None):
            ui_components['log_output'] = footer_container.log_output
            ui_components['log_components'] = footer_container.log_output
        else:
            ui_components['log_output'] = footer_container.log_accordion
            ui_components['log_components'] = footer_container.log_accordion
    
    # 8. Create the final UI layout using main_container
    
    # Fix status panel width to prevent horizontal scrollbar
    if hasattr(header_container, 'status_panel') and header_container.status_panel is not None:
        header_container.status_panel.layout.width = '100%'
        header_container.status_panel.layout.max_width = '100%'
    
    # Fix summary container title duplication - remove the title from the component
    # if it already has a title in the container
    if hasattr(setup_summary, 'title_widget') and hasattr(setup_summary, 'value'):
        # Create a new widget without the title
        setup_content = setup_summary.value
        summary_container.set_content(setup_content)
    
    # Make sure log accordion is inside the footer container
    if 'log_accordion' in ui_components and ui_components['log_accordion'] is not None:
        # Check if it's already in the footer container
        if footer_container.container.children and ui_components['log_accordion'] not in footer_container.container.children:
            # Add it to the footer if needed
            footer_children = list(footer_container.container.children)
            footer_children.append(ui_components['log_accordion'])
            footer_container.container.children = tuple(footer_children)
    
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
    ui_components['ui'] = main_ui
    
    return ui_components