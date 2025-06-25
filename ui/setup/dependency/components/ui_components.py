"""
File: smartcash/ui/setup/dependency/components/ui_components.py
Deskripsi: UI components dengan flexbox layout dan tanpa horizontal scrollbar
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create dependency installer UI dengan flexbox layout"""
    config = config or {}
    
    # Import components
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.components import (
        create_header,
        create_action_buttons,
        create_dual_progress_tracker,
        create_status_panel,
        create_log_accordion,
        create_save_reset_buttons,
        create_divider
    )
    from smartcash.ui.setup.dependency.components.ui_package_selector import create_package_selector_grid
    
    # Helper untuk icons
    get_icon = lambda key, fallback="📦": ICONS.get(key, fallback)
    
    # Header
    header = create_header(
        f"{get_icon('download')} Dependency Installer",
        "Setup packages yang diperlukan untuk SmartCash"
    )
    
    # Status panel
    status_panel = create_status_panel(
        "Pilih packages yang akan diinstall dan klik tombol install",
        "info"
    )
    
    # Package selector dengan flexbox
    package_selector = create_package_selector_grid(config)
        
    # Custom packages textarea
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
    
    # Action buttons dengan flexbox
    action_buttons = create_action_buttons(
        primary_label="Install",
        secondary_label="Analyze", 
        warning_label="System Report",
        primary_icon=get_icon('download'),
        secondary_icon=get_icon('search'),
        warning_icon=get_icon('info'),
        button_width='120px'
    )
    
    # Button container dengan flexbox
    action_container = widgets.HBox([
        action_buttons['download_button'],
        action_buttons['check_button'], 
        action_buttons['cleanup_button']
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        justify_content='flex-start',
        width='100%',
        overflow='hidden'
    ))
    
    # Save/reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        button_width='100px'
    )
    
    # Save/reset container dengan flexbox
    save_reset_container = widgets.HBox([
        save_reset_buttons['save_button'],
        save_reset_buttons['reset_button']
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        align_items='center',
        justify_content='flex-start',
        width='100%',
        overflow='hidden'
    ))
    
    # Progress tracker
    progress_tracker = create_dual_progress_tracker("Package Installation")
    
    # Log accordion
    log_accordion = create_log_accordion()
    
    # Main UI dengan flexbox layout
    main_ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(value="<h4>📦 Package Selection</h4>"),
        package_selector['widget'],
        widgets.HTML(value="<h4>➕ Custom Packages</h4>"),
        custom_packages,
        create_divider(),
        action_container,
        save_reset_container,
        create_divider(),
        progress_tracker.container,
        log_accordion['widget']
    ], layout=widgets.Layout(
        display='flex',
        flex_direction='column',
        align_items='stretch',
        width='100%',
        overflow='hidden',  # Prevent horizontal scroll
        box_sizing='border-box'
    ))
    
    # Return components with all required keys
    components = {
        # Core UI components
        'ui': main_ui,
        'header': header,
        'status_panel': status_panel,
        'package_selector': package_selector.get('widget'),
        'custom_packages': custom_packages,
        
        # Action buttons
        'install_button': action_buttons.get('download_button'),
        'analyze_button': action_buttons.get('check_button'),
        'check_button': action_buttons.get('check_button'),  # Duplicate of analyze_button for backward compatibility
        'system_report_button': action_buttons.get('cleanup_button'),
        
        # Form controls
        'save_button': save_reset_buttons.get('save_button'),
        'reset_button': save_reset_buttons.get('reset_button'),
        
        # Progress and logging
        'progress_tracker': progress_tracker,
        'log_accordion': log_accordion.get('widget'),
        'log_output': log_accordion.get('output'),
        
        # Additional components for reference
        'action_buttons': action_buttons,
        'save_reset_buttons': save_reset_buttons,
        'package_selector_widget': package_selector  # Keep full selector for advanced usage
    }
    
    # Log any missing critical components
    missing = [k for k, v in components.items() if v is None and k not in ['logger']]
    if missing:
        import logging
        logging.warning(f"Missing optional components: {missing}")

    return components