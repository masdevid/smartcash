"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: Enhanced UI utilities dengan log accordion control dan proper status updates
"""

from typing import Dict, Any, Optional, List, Callable
from IPython.display import display, HTML
import datetime

# === LOG ACCORDION CONTROL ===

def reset_and_expand_log_accordion(ui_components: Dict[str, Any]) -> None:
    """Reset dan expand log accordion untuk operation baru"""
    log_accordion = ui_components.get('log_accordion')
    if not log_accordion:
        return
    
    try:
        # Reset content di log output
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'clear_output'):
            with log_output:
                log_output.clear_output(wait=True)
        
        # Force expand accordion untuk show logs
        if hasattr(log_accordion, 'selected_index'):
            log_accordion.selected_index = 0  # Expand first (dan satu-satunya) tab
        
        # Alternative approach untuk accordion expansion
        if hasattr(log_accordion, 'children') and log_accordion.children:
            first_child = log_accordion.children[0]
            if hasattr(first_child, 'layout'):
                first_child.layout.visibility = 'visible'
                first_child.layout.display = 'flex'
        
        logger_bridge = get_logger_bridge(ui_components)
        if logger_bridge:
            logger_bridge.debug("🔄 Log accordion direset dan diperluas")
            
    except Exception as e:
        logger_bridge = get_logger_bridge(ui_components)
        if logger_bridge:
            logger_bridge.debug(f"🔍 Log accordion reset warning: {str(e)}")

def disable_log_accordion(ui_components: Dict[str, Any]) -> None:
    """Disable log accordion selama proses berjalan"""
    log_accordion = ui_components.get('log_accordion')
    if not log_accordion or not hasattr(log_accordion, 'disabled'):
        return
    
    try:
        log_accordion.disabled = True
        logger_bridge = get_logger_bridge(ui_components)
        if logger_bridge:
            logger_bridge.debug("⏸️ Log accordion dinonaktifkan selama proses")
    except Exception:
        pass

def enable_log_accordion(ui_components: Dict[str, Any]) -> None:
    """Enable log accordion setelah proses selesai"""
    log_accordion = ui_components.get('log_accordion')
    if not log_accordion or not hasattr(log_accordion, 'disabled'):
        return
    
    try:
        log_accordion.disabled = False
        logger_bridge = get_logger_bridge(ui_components)
        if logger_bridge:
            logger_bridge.debug("▶️ Log accordion diaktifkan kembali")
    except Exception:
        pass

# === ENHANCED STATUS PANEL UPDATES ===

def update_status_panel_enhanced(ui_components: Dict[str, Any], message: str, 
                                status_type: str, force_update: bool = True) -> None:
    """Enhanced status panel update dengan force refresh"""
    status_panel = ui_components.get('status_panel')
    if not status_panel:
        return
    
    try:
        # Method 1: Direct update_status method
        if hasattr(status_panel, 'update_status'):
            status_panel.update_status(message, status_type)
        
        # Method 2: Direct value update jika method tidak ada
        elif hasattr(status_panel, 'value'):
            color_map = {
                'success': '#28a745', 'info': '#007bff', 
                'warning': '#ffc107', 'error': '#dc3545'
            }
            color = color_map.get(status_type, '#495057')
            
            icon_map = {
                'success': '✅', 'info': 'ℹ️', 
                'warning': '⚠️', 'error': '❌'
            }
            icon = icon_map.get(status_type, 'ℹ️')
            
            status_panel.value = f"""
            <div style="padding: 8px 12px; background-color: {color}; color: white; 
                       border-radius: 4px; margin: 5px 0; font-weight: 500;">
                {icon} {message}
            </div>
            """
        
        # Force refresh jika diperlukan
        if force_update and hasattr(status_panel, 'hold_sync'):
            status_panel.hold_sync()
            status_panel.sync()
        
        logger_bridge = get_logger_bridge(ui_components)
        if logger_bridge:
            logger_bridge.debug(f"📊 Status panel diperbarui: {status_type} - {message}")
            
    except Exception as e:
        logger_bridge = get_logger_bridge(ui_components)
        if logger_bridge:
            logger_bridge.debug(f"🔍 Status panel update warning: {str(e)}")

# === ENHANCED OPERATION FLOW ===

def start_operation_flow(ui_components: Dict[str, Any], operation_name: str) -> None:
    """Start operation flow dengan proper UI state management"""
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.info(f"🚀 Memulai {operation_name}...")
    
    # 1. Clear semua outputs
    clear_outputs(ui_components)
    
    # 2. Reset dan expand log accordion
    reset_and_expand_log_accordion(ui_components)
    
    # 3. Disable buttons
    disable_buttons(ui_components)
    
    # 4. Disable log accordion
    disable_log_accordion(ui_components)
    
    # 5. Update status panel
    update_status_panel_enhanced(ui_components, f"🚀 {operation_name} dimulai...", 'info')
    
    # 6. Setup progress
    setup_progress(ui_components, f"🚀 {operation_name}...")

def complete_operation_flow(ui_components: Dict[str, Any], operation_name: str, 
                          success: bool, message: str) -> None:
    """Complete operation flow dengan proper cleanup"""
    status_type = 'success' if success else 'error'
    icon = '✅' if success else '❌'
    
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        log_method = logger_bridge.success if success else logger_bridge.error
        log_method(f"{icon} {operation_name}: {message}")
    
    # 1. Complete progress
    if success:
        complete_progress(ui_components, f"{icon} {message}")
    else:
        error_progress(ui_components, f"{icon} {message}")
    
    # 2. Update status panel dengan force refresh
    update_status_panel_enhanced(ui_components, f"{icon} {message}", status_type, force_update=True)
    
    # 3. Enable buttons
    enable_buttons(ui_components)
    
    # 4. Enable log accordion
    enable_log_accordion(ui_components)
    
    # 5. Clear confirmation flags
    confirmation_flags = ['_preprocessing_confirmed', '_cleanup_confirmed']
    for flag in confirmation_flags:
        ui_components.pop(flag, None)

# === EXISTING UTILITIES ===

def get_logger_bridge(ui_components: Dict[str, Any]) -> Optional[Any]:
    """Get logger bridge dengan safe validation"""
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        print("[WARNING] Logger bridge not initialized in UI components")
    return logger_bridge

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info") -> None:
    """Enhanced log message ke UI dengan fallback dan proper styling"""
    logger_bridge = get_logger_bridge(ui_components)
    
    # Primary: use logger bridge jika tersedia
    if logger_bridge:
        log_method = getattr(logger_bridge, level.lower(), logger_bridge.info)
        log_method(message)
        return
    
    # Fallback: direct UI logging
    _log_to_ui_direct(ui_components, message, level)

def _log_to_ui_direct(ui_components: Dict[str, Any], message: str, level: str) -> None:
    """Direct UI logging fallback dengan styling"""
    log_output = ui_components.get('log_output')
    if not log_output:
        print(f"[{level.upper()}] {message}")
        return
    
    # Color dan styling untuk different levels
    level_styles = {
        'info': {'color': '#2196F3', 'icon': 'ℹ️'},
        'success': {'color': '#4CAF50', 'icon': '✅'}, 
        'warning': {'color': '#FF9800', 'icon': '⚠️'},
        'error': {'color': '#F44336', 'icon': '❌'},
        'debug': {'color': '#9E9E9E', 'icon': '🔍'}
    }
    
    style = level_styles.get(level, level_styles['info'])
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    
    html = f"""
    <div style='margin: 2px 0; padding: 4px 8px; border-left: 3px solid {style['color']}; 
                background-color: rgba(248,249,250,0.8); border-radius: 4px;'>
        <span style='color: #666; font-size: 11px;'>[{timestamp}]</span>
        <span style='color: {style['color']}; margin-left: 4px;'>{style['icon']} {message.strip()}</span>
    </div>
    """
    
    with log_output:
        display(HTML(html))

# === VISIBILITY CONTROL ===

def update_widget_visibility(widget: Any, visible: bool, height: Optional[str] = None) -> None:
    """Generic function untuk update widget visibility"""
    if not widget or not hasattr(widget, 'layout'):
        return
    
    try:
        widget.layout.visibility = 'visible' if visible else 'hidden'
        widget.layout.height = height or ('auto' if visible else '0px')
    except Exception as e:
        logger_bridge = get_logger_bridge({})
        if logger_bridge:
            logger_bridge.debug(f"🔍 Widget visibility update warning: {str(e)}")

def hide_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Hide confirmation area dengan proper validation"""
    confirmation_area = ui_components.get('confirmation_area')
    update_widget_visibility(confirmation_area, False, '0px')

def show_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Show confirmation area dengan proper validation"""
    confirmation_area = ui_components.get('confirmation_area')
    update_widget_visibility(confirmation_area, True, 'auto')

def clear_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear outputs dan hide confirmation area"""
    hide_confirmation_area(ui_components)

# === BUTTON STATE MANAGEMENT ===

def update_buttons_state(ui_components: Dict[str, Any], disabled: bool, 
                        button_ids: Optional[List[str]] = None) -> int:
    """Generic function untuk update multiple button states"""
    if not button_ids:
        button_ids = ['preprocess_btn', 'check_btn', 'cleanup_btn']
    
    updated_count = 0
    action_buttons = ui_components.get('action_buttons', {})
    
    for btn_id in button_ids:
        button = action_buttons.get(btn_id) or ui_components.get(btn_id)
        if button and hasattr(button, 'disabled'):
            try:
                button.disabled = disabled
                updated_count += 1
            except Exception:
                pass  # Silent fail untuk compatibility
    
    return updated_count

def disable_buttons(ui_components: Dict[str, Any]) -> None:
    """Disable operation buttons during processing"""
    updated_count = update_buttons_state(ui_components, True)
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug(f"⏸️ {updated_count} tombol dinonaktifkan")

def enable_buttons(ui_components: Dict[str, Any]) -> None:
    """Enable operation buttons after processing"""
    updated_count = update_buttons_state(ui_components, False)
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug(f"▶️ {updated_count} tombol diaktifkan")

# === PROGRESS MANAGEMENT ===

def update_progress_state(ui_components: Dict[str, Any], bar_style: str, 
                         value: int, description: str) -> None:
    """Generic function untuk update progress state"""
    progress = ui_components.get('progress')
    if not progress:
        return
    
    try:
        if hasattr(progress, 'bar_style'):
            progress.bar_style = bar_style
        if hasattr(progress, 'value'):
            progress.value = value
        if hasattr(progress, 'description'):
            progress.description = description
    except Exception as e:
        logger_bridge = get_logger_bridge(ui_components)
        if logger_bridge:
            logger_bridge.debug(f"🔍 Progress update warning: {str(e)}")

def setup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Setup progress tracker untuk operation"""
    update_progress_state(ui_components, '', 0, message)

def complete_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete progress tracker dengan success message"""
    update_progress_state(ui_components, 'success', 100, message)

def error_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state pada progress tracker"""
    update_progress_state(ui_components, 'danger', 0, f"Error: {message}")

# === ERROR HANDLING ===

def handle_error(ui_components: Dict[str, Any], error_msg: str, 
                logger: Optional[Any] = None) -> None:
    """Enhanced error handler dengan proper cleanup dan state management"""
    # Use provided logger atau get dari ui_components
    logger_bridge = logger or get_logger_bridge(ui_components)
    
    # Log error ke UI
    if logger_bridge:
        logger_bridge.error(f"❌ {error_msg}")
    else:
        log_to_ui(ui_components, error_msg, 'error')
    
    # Set error state pada progress jika ada
    error_progress(ui_components, error_msg)
    
    # Enable buttons untuk ensure UI tidak terkunci
    enable_buttons(ui_components)
    
    # Enable log accordion
    enable_log_accordion(ui_components)
    
    # Update status panel dengan force refresh
    update_status_panel_enhanced(ui_components, f"❌ {error_msg}", 'error', force_update=True)

# === ONE-LINER UTILITIES ===

# Status shortcuts
log_info = lambda ui_components, msg: log_to_ui(ui_components, msg, 'info')
log_success = lambda ui_components, msg: log_to_ui(ui_components, msg, 'success')
log_warning = lambda ui_components, msg: log_to_ui(ui_components, msg, 'warning')
log_error = lambda ui_components, msg: log_to_ui(ui_components, msg, 'error')

# Progress shortcuts
start_progress = lambda ui_components, msg: setup_progress(ui_components, msg)
finish_progress = lambda ui_components, msg: complete_progress(ui_components, msg)
fail_progress = lambda ui_components, msg: error_progress(ui_components, msg)

# Enhanced operation flow shortcuts
start_operation = lambda ui_components, op_name: start_operation_flow(ui_components, op_name)
complete_operation = lambda ui_components, op_name, success, msg: complete_operation_flow(ui_components, op_name, success, msg)