"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Enhanced UI components dengan full Progress Bridge integration dan real-time backend status sync
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
    """Enhanced preprocessing UI dengan full Progress Bridge integration dan backend status sync"""
    
    config = config or {}
    
    # === CORE COMPONENTS ===
    header = create_header("🔧 Dataset Preprocessing", "Preprocessing dataset dengan validasi dan real-time progress")
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
    
    # === ENHANCED DUAL PROGRESS TRACKER ===
    try:
        progress_components = create_dual_progress_tracker(
            operation="Preprocessing", 
            auto_hide=False,
            steps=["Validation", "Processing", "Completion"]
        )
        progress_tracker = progress_components.get('tracker') if isinstance(progress_components, dict) else progress_components
        progress_container = progress_components.get('container') if isinstance(progress_components, dict) else widgets.VBox([])
    except Exception:
        # Enhanced fallback dual progress tracker
        progress_tracker = _create_fallback_dual_tracker()
        progress_container = widgets.VBox([progress_tracker])
    
    # === LOG ACCORDION ===
    try:
        log_components = create_log_accordion(module_name='preprocessing', height='250px')
        log_output = log_components['log_output']
        log_accordion = log_components['log_accordion']
    except Exception:
        # Enhanced fallback log
        log_output = widgets.Output(layout=widgets.Layout(
            width='100%', height='200px', border='1px solid #ddd', 
            overflow='auto', padding='10px'
        ))
        log_accordion = widgets.Accordion([log_output])
        log_accordion.set_title(0, "📋 Log Preprocessing")
    
    # === MAIN UI ASSEMBLY ===
    ui = widgets.VBox([
        header,
        status_panel,
        input_options,
        save_reset_buttons['container'],
        action_buttons['container'],
        confirmation_area,
        progress_container,
        log_accordion
    ], layout=widgets.Layout(width='100%', padding='8px'))
    
    # === ENHANCED COMPONENTS MAPPING dengan Backend Integration ===
    ui_components = {
        'ui': ui,
        
        # Action buttons - CRITICAL dengan backend awareness
        'preprocess_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Save/reset - CRITICAL dengan config sync
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Communication areas
        'confirmation_area': confirmation_area,
        'status_panel': status_panel,
        
        # Enhanced progress tracking dengan backend integration
        'progress_tracker': progress_tracker,
        'progress_container': progress_container,
        
        # Log - CRITICAL dengan namespace awareness
        'log_output': log_output,
        'log_accordion': log_accordion,
        'status': log_output,  # Compatibility alias
        
        # Input components (validated)
        'input_options': input_options,
        'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
        'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None),
        'target_splits_select': getattr(input_options, 'target_splits_select', None),
        'batch_size_input': getattr(input_options, 'batch_size_input', None),
        'validation_checkbox': getattr(input_options, 'validation_checkbox', None),
        'preserve_aspect_checkbox': getattr(input_options, 'preserve_aspect_checkbox', None),
        'move_invalid_checkbox': getattr(input_options, 'move_invalid_checkbox', None),
        'invalid_dir_input': getattr(input_options, 'invalid_dir_input', None),
        
        # Module metadata
        'module_name': 'preprocessing',
        'ui_initialized': True
    }
    
    # === SETUP ENHANCED PROGRESS BRIDGE INTEGRATION ===
    _setup_progress_bridge_integration(ui_components)
    
    # === SETUP BACKEND BUTTON MANAGEMENT ===
    _setup_backend_button_management(ui_components)
    
    # === REGISTER REAL-TIME STATUS UPDATES ===
    _setup_real_time_status_updates(ui_components)
    
    return ui_components

def _create_fallback_dual_tracker():
    """📊 Create fallback dual progress tracker dengan manual dual bars"""
    overall_bar = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Overall:',
        bar_style='',
        style={'bar_color': '#28a745', 'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    current_bar = widgets.IntProgress(
        value=0, min=0, max=100,
        description='Current:',
        bar_style='',
        style={'bar_color': '#007bff', 'description_width': '80px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )
    
    status_label = widgets.HTML(
        value="<div style='color: #6c757d; font-size: 12px; margin: 5px 0;'>Ready</div>",
        layout=widgets.Layout(width='100%')
    )
    
    container = widgets.VBox([
        widgets.HTML("<h5 style='margin: 5px 0; color: #495057;'>📊 Progress Tracking</h5>"),
        overall_bar,
        current_bar,
        status_label
    ], layout=widgets.Layout(
        width='100%', border='1px solid #dee2e6', 
        border_radius='5px', padding='10px', margin='10px 0'
    ))
    
    # Add methods untuk compatibility
    def update_overall(progress: int, message: str = None):
        overall_bar.value = max(0, min(progress, 100))
        if message:
            status_label.value = f"<div style='color: #28a745; font-size: 12px; margin: 5px 0;'>{message}</div>"
    
    def update_current(progress: int, message: str = None):
        current_bar.value = max(0, min(progress, 100))
        if message:
            status_label.value = f"<div style='color: #007bff; font-size: 12px; margin: 5px 0;'>{message}</div>"
    
    def show(operation_name: str = "Processing"):
        container.layout.display = 'flex'
        status_label.value = f"<div style='color: #495057; font-size: 12px; margin: 5px 0;'>🚀 {operation_name}</div>"
    
    def hide():
        container.layout.display = 'none'
    
    def complete(message: str = "Completed"):
        overall_bar.value = 100
        current_bar.value = 100
        overall_bar.bar_style = 'success'
        current_bar.bar_style = 'success'
        status_label.value = f"<div style='color: #28a745; font-size: 12px; margin: 5px 0;'>✅ {message}</div>"
    
    def error(message: str = "Error"):
        overall_bar.bar_style = 'danger'
        current_bar.bar_style = 'danger'
        status_label.value = f"<div style='color: #dc3545; font-size: 12px; margin: 5px 0;'>❌ {message}</div>"
    
    def reset():
        overall_bar.value = 0
        current_bar.value = 0
        overall_bar.bar_style = ''
        current_bar.bar_style = ''
        status_label.value = "<div style='color: #6c757d; font-size: 12px; margin: 5px 0;'>Ready</div>"
    
    # Attach methods
    container.update_overall = update_overall
    container.update_current = update_current
    container.show = show
    container.hide = hide
    container.complete = complete
    container.error = error
    container.reset = reset
    
    return container

def _setup_progress_bridge_integration(ui_components: Dict[str, Any]):
    """🌉 Setup Progress Bridge integration dengan dual tracker"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.progress_utils import create_dual_progress_callback
        
        # Create progress callback untuk backend integration
        progress_callback = create_dual_progress_callback(ui_components)
        ui_components['progress_callback'] = progress_callback
        
        # Setup progress manager
        if 'progress_manager' in ui_components:
            progress_manager = ui_components['progress_manager']
            # Auto-register callback ke backend jika ada
            if hasattr(progress_manager, 'create_backend_callback'):
                backend_callback = progress_manager.create_backend_callback()
                ui_components['backend_progress_callback'] = backend_callback
        
    except Exception as e:
        # Fallback progress callback
        def fallback_callback(level: str, current: int, total: int, message: str):
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    if level in ['overall', 'step']:
                        progress_tracker.update_overall(int((current/total)*100), message)
                    else:
                        progress_tracker.update_current(int((current/total)*100), message)
            except Exception:
                pass
        
        ui_components['progress_callback'] = fallback_callback

def _setup_backend_button_management(ui_components: Dict[str, Any]):
    """🔘 Setup enhanced button management dengan backend operation awareness"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.button_manager import setup_backend_button_management
        
        button_manager = setup_backend_button_management(ui_components)
        ui_components['button_manager'] = button_manager
        
        # Register button state sync dengan progress updates
        if 'progress_callback' in ui_components:
            enhanced_callback = button_manager.register_progress_updates(ui_components['progress_callback'])
            ui_components['progress_callback'] = enhanced_callback
        
    except Exception as e:
        # Fallback button management
        def simple_disable_all():
            for key in ['preprocess_button', 'check_button', 'cleanup_button']:
                button = ui_components.get(key)
                if button and hasattr(button, 'disabled'):
                    button.disabled = True
        
        def simple_enable_all():
            for key in ['preprocess_button', 'check_button', 'cleanup_button']:
                button = ui_components.get(key)
                if button and hasattr(button, 'disabled'):
                    button.disabled = False
        
        ui_components['disable_all_buttons'] = simple_disable_all
        ui_components['enable_all_buttons'] = simple_enable_all

def _setup_real_time_status_updates(ui_components: Dict[str, Any]):
    """📡 Setup real-time status updates dengan backend sync"""
    try:
        def update_status_panel_safe(message: str, status_type: str = "info"):
            """Update status panel dengan safe error handling"""
            try:
                from smartcash.ui.components.status_panel import update_status_panel
                status_panel = ui_components.get('status_panel')
                if status_panel:
                    update_status_panel(status_panel, message, status_type)
            except Exception:
                pass
        
        def log_with_progress_sync(message: str, level: str = "info"):
            """Log dengan progress sync integration"""
            try:
                from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
                log_to_accordion(ui_components, message, level)
                
                # Sync dengan progress callback jika milestone
                if 'progress_callback' in ui_components and level in ['success', 'error']:
                    progress_val = 100 if level == 'success' else 0
                    ui_components['progress_callback']('overall', progress_val, 100, message)
                    
            except Exception:
                pass
        
        # Register enhanced functions
        ui_components['update_status_safe'] = update_status_panel_safe
        ui_components['log_with_progress'] = log_with_progress_sync
        
        # Setup config handler progress sync
        def sync_config_operations():
            """Sync config operations dengan progress tracking"""
            config_handler = ui_components.get('config_handler')
            if config_handler and hasattr(config_handler, 'set_progress_callback'):
                progress_callback = ui_components.get('progress_callback')
                if progress_callback:
                    config_handler.set_progress_callback(progress_callback)
        
        ui_components['sync_config_operations'] = sync_config_operations
        
    except Exception as e:
        # Silent fallback untuk prevent UI break
        ui_components['update_status_safe'] = lambda msg, level: None
        ui_components['log_with_progress'] = lambda msg, level: None
        ui_components['sync_config_operations'] = lambda: None

def register_backend_service_integration(ui_components: Dict[str, Any], backend_service) -> bool:
    """🔗 Register backend service integration dengan UI components"""
    try:
        # Register progress callback ke backend service
        if hasattr(backend_service, 'register_progress_callback'):
            progress_callback = ui_components.get('progress_callback')
            if progress_callback:
                backend_service.register_progress_callback(progress_callback)
        
        # Setup button manager dengan backend service
        button_manager = ui_components.get('button_manager')
        if button_manager and hasattr(button_manager, 'sync_with_backend_status'):
            button_manager.sync_with_backend_status(backend_service)
        
        # Store backend service reference
        ui_components['backend_service'] = backend_service
        
        return True
        
    except Exception as e:
        from smartcash.common.logger import get_logger
        get_logger('ui_components').warning(f"⚠️ Backend service integration warning: {str(e)}")
        return False

def setup_config_progress_sync(ui_components: Dict[str, Any], config_handler) -> bool:
    """🔄 Setup config handler progress sync"""
    try:
        # Call sync function jika ada
        if 'sync_config_operations' in ui_components:
            ui_components['sync_config_operations']()
        
        # Direct setup
        if hasattr(config_handler, 'set_progress_callback'):
            progress_callback = ui_components.get('progress_callback')
            if progress_callback:
                config_handler.set_progress_callback(progress_callback)
                return True
        
        return False
        
    except Exception:
        return False

# Export untuk backward compatibility
def create_preprocessing_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return create_preprocessing_main_ui(config)