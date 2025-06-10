"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Simplified UI components dengan dual progress tracker
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from .input_options import create_preprocessing_input_options

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simplified preprocessing UI dengan dual progress tracker"""
    
    config = config or {}
    
    # === CORE COMPONENTS ===
    header = create_header("🔧 Dataset Preprocessing", "Preprocessing dataset dengan validasi")
    status_panel = create_status_panel("🚀 Siap memulai preprocessing dataset", "info")
    input_options = create_preprocessing_input_options(config)
    
    # === ACTION BUTTONS ===
    action_buttons = create_action_buttons(
        primary_label="🚀 Mulai Preprocessing",
        secondary_buttons=[("🔍 Check Dataset", "search", "info")],
        cleanup_enabled=True
    )
    
    # === SAVE/RESET BUTTONS ===
    save_reset_buttons = create_save_reset_buttons()
    
    # === CONFIRMATION AREA ===
    confirmation_area = widgets.Output(layout=widgets.Layout(
        width='100%', min_height='50px', max_height='200px', 
        margin='10px 0', border='1px solid #e0e0e0',
        border_radius='4px', overflow='auto'
    ))
    
    # === DUAL PROGRESS TRACKER ===
    try:
        progress_components = create_dual_progress_tracker(operation="Preprocessing", auto_hide=False)
    except Exception:
        # Fallback manual progress tracker
        pass
    
    # === LOG ACCORDION ===
    try:
        log_components = create_log_accordion(module_name='preprocessing')
    except Exception:
        # Fallback log
        pass
    
    # === MAIN UI ASSEMBLY ===
    ui = widgets.VBox([
        header,
        status_panel,
        input_options,
        save_reset_buttons['container'],
        action_buttons['container'],
        confirmation_area,
        progress_components.get('container') if isinstance(progress_components, dict) else widgets.VBox([]),
        log_components['log_accordion']
    ], layout=widgets.Layout(width='100%', padding='8px'))
    
    # === SIMPLIFIED COMPONENTS MAPPING ===
    return {
        'ui': ui,
        
        # Action buttons - CRITICAL
        'preprocess_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Save/reset - CRITICAL
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Confirmation
        'confirmation_area': confirmation_area,
        
        # Progress tracker dengan safe access
        'progress_tracker': progress_components.get('tracker') if isinstance(progress_components, dict) else progress_components,
        'progress_container': progress_components.get('container') if isinstance(progress_components, dict) else widgets.VBox([]),
        
        # Log - CRITICAL
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        'status': log_components['log_output'],
        
        # Input components (optional)
        'input_options': input_options,
        'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
        'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None),
        'target_splits_select': getattr(input_options, 'target_splits_select', None),
        'batch_size_input': getattr(input_options, 'batch_size_input', None),
        'validation_checkbox': getattr(input_options, 'validation_checkbox', None),
        
        # Module info
        'module_name': 'preprocessing'
    }
