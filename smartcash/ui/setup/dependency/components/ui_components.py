"""
File: smartcash/ui/setup/dependency/components/ui_components.py
Deskripsi: UI components tanpa check/uncheck buttons integration
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components import create_dual_progress_tracker, create_header, create_status_panel, create_log_accordion, create_save_reset_buttons, create_action_buttons
from smartcash.ui.setup.dependency.components.ui_package_selector import create_package_selector_grid

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create dependency installer UI tanpa check/uncheck buttons"""
    
    get_icon = lambda key, fallback="📦": ICONS.get(key, fallback) if 'ICONS' in globals() else fallback
    
    # Header
    header = create_header(
        f"{get_icon('download', '📦')} Dependency Installer", 
        "Setup packages yang diperlukan untuk SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel("Pilih packages yang akan diinstall dan klik tombol install", "info")
    
    # Package selector grid
    package_selector = create_package_selector_grid(config)
    
    # Custom packages input
    custom_packages = widgets.Textarea(
        placeholder='Package tambahan (satu per baris)\ncontoh: numpy>=1.21.0\nopencv-python>=4.5.0',
        layout=widgets.Layout(
            width='100%',
            height='90px',
            margin='8px 0',
            border='1px solid #ddd',
            border_radius='4px',
            overflow='hidden'  # Prevent horizontal scroll
        )
    )
    
    # Action buttons
    action_buttons = create_action_buttons(
        primary_label="Install",
        primary_icon='download',
        secondary_buttons=[
            ('Analyze', 'search', 'info'),
            ('Status Check', 'check-circle', 'info'),
            ('System Report', 'info', 'info')
        ],
        button_width='120px'
    )
    
    # Map button names for backward compatibility
    action_buttons['status_check_button'] = action_buttons['check_button']
    
    # Action buttons container
    action_container = widgets.HBox([
        action_buttons['download_button'],
        action_buttons['status_check_button'],
        action_buttons['cleanup_button']
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        justify_content='flex-start',
        width='100%',
        gap='8px',
        overflow='hidden'
    ))
    
    # Save/reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        button_width='100px'
    )
    
    # Save/reset container
    save_reset_container = widgets.HBox([
        save_reset_buttons['save_button'],
        save_reset_buttons['reset_button']
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        justify_content='flex-start',
        width='100%',
        gap='8px',
        overflow='hidden'
    ))
    
    # Progress tracker
    progress_tracker = create_dual_progress_tracker("Package Installation")
    
    # Log accordion
    log_accordion = create_log_accordion()
    
    # Create main UI components
    main_ui_children = [
        header,
        status_panel,
        widgets.HTML(value="<h4>📦 Package Selection</h4>"),
        package_selector.get('container', widgets.VBox()),
        widgets.HTML(value="<h4>➕ Custom Packages</h4>"),
        custom_packages,
        widgets.HTML('<hr style="margin: 12px 0; border: 0; border-top: 1px solid #eee;">'),
        action_container,
        save_reset_container,
        widgets.HTML('<hr style="margin: 12px 0; border: 0; border-top: 1px solid #eee;">'),
        progress_tracker.ui_manager.container
    ]
    
    # Add log accordion if available
    if log_accordion and 'widget' in log_accordion and log_accordion['widget'] is not None:
        main_ui_children.append(log_accordion['widget'])
    
    # Filter out None values
    main_ui_children = [c for c in main_ui_children if c is not None]
    
    # Create main container
    main_ui = widgets.VBox(
        main_ui_children,
        layout=widgets.Layout(
            display='flex',
            flex_direction='column',
            align_items='stretch',
            width='100%',
            padding='12px',
            gap='12px',
            overflow='hidden',
            box_sizing='border-box'
        )
    )
    
    # Prepare components dictionary
    components = {
        # Core UI components
        'ui': main_ui,
        'container': main_ui,  # Alias for backward compatibility
        'main_container': main_ui,  # Alias for backward compatibility
        'header': header,
        'status_panel': status_panel,
        'package_selector': package_selector.get('container', None),
        'custom_packages': custom_packages,
        
        # Action buttons
        'install_button': action_buttons.get('download_button'),
        'analyze_button': action_buttons.get('check_button'),
        'check_button': action_buttons.get('check_button'),  # Alias for analyze_button
        'status_check_button': action_buttons.get('check_button'),  # Alias for backward compatibility
        'system_report_button': action_buttons.get('cleanup_button'),
        
        # Form controls
        'save_button': save_reset_buttons.get('save_button'),
        'reset_button': save_reset_buttons.get('reset_button'),
        
        # Progress and logging
        'progress_tracker': progress_tracker.ui_manager,
        'log_accordion': log_accordion.get('widget'),
        'log_output': log_accordion.get('output'),
        
        # Additional references
        'action_buttons': action_buttons,
        'save_reset_buttons': save_reset_buttons,
        'package_selector_widget': package_selector
    }
    
    # Log missing components if needed
    try:
        from smartcash.ui.utils.logging_utils import log_missing_components
        log_missing_components(components)
    except ImportError:
        pass  # Skip if logging utils not available
    
    return components