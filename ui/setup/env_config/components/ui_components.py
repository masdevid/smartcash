"""
File: smartcash/ui/setup/env_config/components/ui_components.py
Deskripsi: Komponen UI untuk environment configuration dengan layout lengkap
"""

import ipywidgets
from typing import Dict, Any
from smartcash.ui.components import (
    create_header, create_status_panel,
    create_log_accordion, create_divider
)
from smartcash.ui.setup.env_config.utils.dual_progress_tracker import track_setup_progress
from smartcash.ui.setup.env_config.components.setup_summary import create_setup_summary
from smartcash.ui.setup.env_config.components.env_info_panel import create_env_info_panel
from smartcash.ui.setup.env_config.components.tips_panel import create_tips_requirements

def create_env_config_ui() -> Dict[str, Any]:
    """
    🎨 Buat komponen UI untuk environment configuration
    
    Returns:
        Dictionary berisi semua komponen UI
    """
    # 1. Header
    header = create_header(
        "🔧 Environment Setup",
        "Konfigurasi lingkungan untuk SmartCash YOLOv5-EfficientNet",
        "🚀"
    )
    
    # 2. Setup Button (Centered)
    setup_button = ipywidgets.Button(
        description="▶️ Setup Environment",
        button_style='primary',
        layout=ipywidgets.Layout(
            width='200px',
            height='45px',
            margin='15px 0',
            border_radius='4px'
        )
    )
    
    setup_button_container = ipywidgets.HBox(
        [setup_button],
        layout=ipywidgets.Layout(
            justify_content='center',
            align_items='center',
            width='100%'
        )
    )
    
    # 3. Status Panel
    status_panel = create_status_panel("Siap untuk setup environment", "info")
    
    # 4. Progress Tracker - Initialize with proper container and styling
    ui_components = {}
    
    # Create a container for the progress bar with proper styling
    progress_container = ipywidgets.VBox(
        layout=ipywidgets.Layout(
            width='100%',
            margin='10px 0',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='5px',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        )
    )
    
    # Store the container in ui_components first
    ui_components['progress_container'] = progress_container
    
    # Create progress tracker - it will register itself in the components
    progress_tracker = track_setup_progress(ui_components=ui_components)
    
    # Ensure the progress bar is visible by default
    if hasattr(progress_tracker, 'show'):
        progress_tracker.show()
    
    # Make sure the progress container is visible
    progress_container.layout.visibility = 'visible'
    progress_container.layout.display = 'flex'
    
    # 5. Log Accordion
    log_accordion = create_log_accordion(
        module_name="Setup Environment",
        height="150px",
        width="100%",
        max_logs=1000,
        show_timestamps=True,
        show_level_icons=True,
        auto_scroll=True
    )
    # Expand the accordion by default
    if 'log_accordion' in log_accordion and hasattr(log_accordion['log_accordion'], 'selected_index'):
        log_accordion['log_accordion'].selected_index = 0  # Expand first accordion item
    
    # 6. Setup Summary
    setup_summary = create_setup_summary()
    
    # 7. Environment & Colab Info Panel
    env_info_panel = create_env_info_panel()
    
    # 8. Tips & Requirements (2 kolom)
    tips_requirements = create_tips_requirements()
    
    # Assemble main UI
    ui_components = {
        'header': header,
        'setup_button': setup_button,
        'setup_button_container': setup_button_container,
        'status_panel': status_panel,
        'progress_tracker': progress_tracker,
        'log_accordion': log_accordion,
        'log_output': log_accordion,  # Alias untuk kompatibilitas
        'setup_summary': setup_summary,
        'env_info_panel': env_info_panel,
        'tips_requirements': tips_requirements
    }
    
    # Ensure progress container exists and is valid
    progress_container = ui_components.get('progress_container')
    if not isinstance(progress_container, ipywidgets.VBox):
        progress_container = ipywidgets.VBox()
        ui_components['progress_container'] = progress_container
    
    # Create main container
    main_container = ipywidgets.VBox([
        header,
        status_panel,
        setup_button_container,
        progress_container,  # Use the validated progress container
        log_accordion['log_accordion'],  # Use the accordion widget from the dictionary
        create_divider(),
        setup_summary,
        env_info_panel,
        tips_requirements
    ], layout=ipywidgets.Layout(
        width='100%',
        padding='15px',
        border='1px solid #ddd',
        border_radius='8px'
    ))
    
    # Store the log components for later use
    ui_components['log_components'] = log_accordion
    ui_components['ui'] = main_container
    
    return ui_components