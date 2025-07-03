"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: UI components for preprocessing using shared container components
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components import create_log_accordion
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """🎨 Create preprocessing UI using shared container components"""
    config = config or {}
    
    # Initialize UI components dictionary
    ui_components = {}
    
    # Import preprocessing input options
    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options
    
    # === CORE COMPONENTS ===
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="Dataset Preprocessing",
        subtitle="Preprocessing dataset dengan YOLO normalization dan real-time progress",
        icon="🚀"
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Create Form Components
    input_options = create_preprocessing_input_options(config)
    
    # 3. Create Form Container
    form_container = create_form_container()
    
    # Place input options in the form container
    form_container['form_container'].children = (input_options,)
    ui_components['form_container'] = form_container['container']
    
    # 4. Create Log Accordion
    log_accordion = create_log_accordion(
        module_name='preprocessing',
        height='200px'
    )
    ui_components['log_accordion'] = log_accordion
    ui_components['log_output'] = log_accordion  # Alias for compatibility
    
    # 5. Create Progress Tracker
    progress_tracker = ProgressTracker(
        operation="Dataset Preprocessing",
        level=ProgressLevel.DUAL,
        auto_hide=False
    )
    ui_components['progress_tracker'] = progress_tracker
    ui_components['progress'] = progress_tracker  # Alias for compatibility
    
    # Show progress tracker initially
    if hasattr(progress_tracker, 'show'):
        progress_tracker.show()
    
    # 6. Create Footer Container with log_output and info_box
    footer_container = create_footer_container(
        show_buttons=False,  # No buttons in footer
        log_accordion=log_accordion,
        info_box=widgets.HTML(
            value="""
            <div style="padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8; margin: 10px 0;">
                <h5>ℹ️ Tips</h5>
                <ul>
                    <li>Gunakan resolusi yang sesuai dengan model target</li>
                    <li>Min-Max normalization (0-1) direkomendasikan untuk YOLO</li>
                    <li>Aktifkan validasi untuk memastikan dataset berkualitas</li>
                </ul>
            </div>
            """
        )
    )
    ui_components['footer_container'] = footer_container.container
    
    # 7. Create Action Container
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "preprocess",
                "text": "🚀 Mulai Preprocessing",
                "style": "primary",
                "order": 1
            },
            {
                "button_id": "check",
                "text": "🔍 Check Dataset",
                "style": "info",
                "order": 2
            },
            {
                "button_id": "cleanup",
                "text": "🗑️ Cleanup",
                "style": "warning",
                "tooltip": "Hapus data preprocessing yang sudah ada",
                "order": 3
            }
        ],
        title="🚀 Preprocessing Operations",
        alignment="left"
    )
    ui_components['action_container'] = action_container
    
    # === MAIN UI ASSEMBLY ===
    
    # 8. Assemble Main Container
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=ui_components['form_container'],
        footer_container=footer_container.container,
        action_container=action_container.container
    )
    ui_components['main_container'] = main_container.container
    ui_components['ui'] = main_container.container  # Alias for compatibility
    
    # === HELPER METHODS ===
    
    def update_status(message: str, status_type: str = "info", show: bool = True) -> None:
        """Update the status panel with a new message.
        
        Args:
            message: New status message
            status_type: Status type (info, success, warning, error)
            show: Whether to show the status panel
        """
        header_container.update_status(message, status_type, show)
    
    def update_title(title: str, subtitle: Optional[str] = None) -> None:
        """Update the header title and subtitle.
        
        Args:
            title: New title text
            subtitle: New subtitle text (or None to keep current)
        """
        header_container.update_title(title, subtitle)
    
    def update_section(section_name: str, new_content: widgets.Widget) -> None:
        """Update a section of the main container.
        
        Args:
            section_name: Name of the section to update ('header', 'form', etc.)
            new_content: New widget to replace the current section
        """
        main_container.update_section(section_name, new_content)
    
    # === BUTTON MAPPING ===
    
    # Extract buttons from action container using standard approach
    preprocess_btn = action_container.get_button('preprocess')
    check_btn = action_container.get_button('check')
    cleanup_btn = action_container.get_button('cleanup')
    
    # Add helper methods to ui_components
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
        'module_name': 'preprocessing',
        'ui_initialized': True,
        'api_integration': True
    })
    
    return ui_components