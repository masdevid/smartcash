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
    """Simplified preprocessing UI dengan dual progress tracker dan fallback"""
    
    try:
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
            progress_components = _create_fallback_progress_tracker()
        
        # === LOG ACCORDION ===
        try:
            log_components = create_log_accordion(module_name='preprocessing')
        except Exception:
            # Fallback log
            log_output = widgets.Output(layout=widgets.Layout(
                width='100%', height='200px', border='1px solid #ddd',
                overflow='auto', margin='10px 0'
            ))
            log_components = {
                'log_output': log_output,
                'log_accordion': widgets.Accordion([log_output])
            }
            log_components['log_accordion'].set_title(0, "📋 Log Preprocessing")
        
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
        
    except Exception as e:
        # === FALLBACK MINIMAL UI ===
        return _create_minimal_fallback_ui(str(e))

def _create_fallback_progress_tracker() -> Dict[str, Any]:
    """Create minimal fallback progress tracker"""
    
    # Single progress bar
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Progress:',
        layout=widgets.Layout(width='100%')
    )
    
    # Minimal tracker object
    class MinimalTracker:
        def __init__(self):
            self.bar = progress_bar
            
        def show(self, operation, **kwargs):
            pass
            
        def update_overall(self, value, message=None):
            self.bar.value = min(100, max(0, value))
                
        def update_current(self, value, message=None):
            self.bar.value = min(100, max(0, value))
            
        def complete(self, message="Complete"):
            self.bar.value = 100
            
        def error(self, message="Error"):
            pass
            
        def reset(self):
            self.bar.value = 0
    
    return {
        'tracker': MinimalTracker(),
        'container': progress_bar
    }

def _create_minimal_fallback_ui(error_msg: str) -> Dict[str, Any]:
    """Create minimal fallback UI jika main UI gagal"""
    
    # Minimal components
    fallback_header = widgets.HTML(f"""
        <div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:4px; margin:10px 0;'>
            <h4>⚠️ Preprocessing UI (Fallback Mode)</h4>
            <p>Error: {error_msg}</p>
        </div>
    """)
    
    # Basic buttons
    preprocess_button = widgets.Button(
        description="🚀 Mulai Preprocessing",
        button_style='primary',
        layout=widgets.Layout(width='160px', margin='5px')
    )
    
    save_button = widgets.Button(
        description="💾 Simpan",
        button_style='primary',
        layout=widgets.Layout(width='100px', margin='5px')
    )
    
    # Basic log
    log_output = widgets.Output(layout=widgets.Layout(
        width='100%', height='200px', border='1px solid #ddd',
        overflow='auto', margin='10px 0'
    ))
    
    # Minimal UI
    ui = widgets.VBox([
        fallback_header,
        widgets.HBox([preprocess_button, save_button]),
        log_output
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Display error in log
    with log_output:
        from IPython.display import display, HTML
        display(HTML(f"<div style='color:red'>❌ UI Error: {error_msg}</div>"))
        display(HTML("<div style='color:orange'>⚠️ Running in fallback mode with minimal features</div>"))
    
    return {
        'ui': ui,
        'preprocess_button': preprocess_button,
        'save_button': save_button,
        'log_output': log_output,
        'status': log_output,
        'module_name': 'preprocessing',
        'fallback_mode': True
    }