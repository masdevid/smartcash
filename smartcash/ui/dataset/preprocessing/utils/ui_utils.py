"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: Refactored UI utilities dengan DRY principles dan better error handling
"""

from typing import Dict, Any, Optional, List, Callable
from IPython.display import display, HTML
import datetime

# === CORE UI UTILITIES ===

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
    
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug("Menyembunyikan area konfirmasi")

def show_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Show confirmation area dengan proper validation"""
    confirmation_area = ui_components.get('confirmation_area')
    update_widget_visibility(confirmation_area, True, 'auto')
    
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug("Menampilkan area konfirmasi")

def clear_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear outputs dan hide confirmation area"""
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug("Membersihkan output dan menyembunyikan area konfirmasi")
    
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
    logger_bridge = get_logger_bridge(ui_components)
    
    updated_count = update_buttons_state(ui_components, True)
    
    if logger_bridge:
        logger_bridge.debug(f"Menonaktifkan {updated_count} tombol operasi")

def enable_buttons(ui_components: Dict[str, Any]) -> None:
    """Enable operation buttons after processing"""
    logger_bridge = get_logger_bridge(ui_components)
    
    updated_count = update_buttons_state(ui_components, False)
    
    if logger_bridge:
        logger_bridge.debug(f"Mengaktifkan {updated_count} tombol operasi")

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
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug(f"Menyiapkan progress tracker: {message}")
    
    update_progress_state(ui_components, '', 0, message)

def complete_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete progress tracker dengan success message"""
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug(f"Menyelesaikan progress tracker: {message}")
    
    update_progress_state(ui_components, 'success', 100, message)

def error_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state pada progress tracker"""
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.error(f"Error pada progress tracker: {message}")
    
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
    _safe_enable_buttons(ui_components, logger_bridge)
    
    # Update status panel jika tersedia
    _update_status_panel(ui_components, f"Error: {error_msg}", 'error')

def _safe_enable_buttons(ui_components: Dict[str, Any], logger_bridge: Optional[Any]) -> None:
    """Safe button enabling dengan error handling"""
    try:
        enable_buttons(ui_components)
        if logger_bridge:
            logger_bridge.debug("Tombol operasi diaktifkan ulang setelah error")
    except Exception as btn_error:
        error_msg = f"Gagal mengaktifkan tombol setelah error: {str(btn_error)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        else:
            print(f"[ERROR] {error_msg}")

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str) -> None:
    """Update status panel dengan message"""
    status_panel = ui_components.get('status_panel')
    if status_panel and hasattr(status_panel, 'update_status'):
        try:
            status_panel.update_status(message, status_type)
        except Exception:
            pass  # Silent fail untuk compatibility

# === UI STATE MANAGEMENT ===

def reset_ui_state(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke initial state setelah operation"""
    logger_bridge = get_logger_bridge(ui_components)
    if logger_bridge:
        logger_bridge.debug("🔄 Mereset UI state ke kondisi awal")
    
    # Enable buttons
    enable_buttons(ui_components)
    
    # Hide confirmation area
    hide_confirmation_area(ui_components)
    
    # Clear confirmation flags
    confirmation_flags = ['_preprocessing_confirmed', '_cleanup_confirmed']
    for flag in confirmation_flags:
        ui_components.pop(flag, None)
    
    # Reset progress ke idle state
    update_progress_state(ui_components, '', 0, "Siap untuk operasi")

def show_error_ui(ui_components: Dict[str, Any], error_msg: str) -> None:
    """Show error di UI dengan proper formatting"""
    handle_error(ui_components, error_msg)

# === ACCORDION LOGGING (Legacy Support) ===

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = "info") -> None:
    """Fallback log function untuk accordion output (legacy support)"""
    logger_bridge = get_logger_bridge(ui_components)
    
    # Primary: gunakan logger bridge
    if logger_bridge:
        log_method = getattr(logger_bridge, level.lower(), logger_bridge.info)
        log_method(f"[Accordion] {message}")
        return
    
    # Fallback: direct accordion logging
    accordion = ui_components.get('log_accordion')
    if accordion and hasattr(accordion, 'children') and accordion.children:
        log_output = accordion.children[0]
        if hasattr(log_output, 'append_stdout'):
            log_output.append_stdout(f"[{level.upper()}] {message}\n")

# === ONE-LINER UTILITIES ===

# Quick access functions dengan one-liner pattern
get_confirmation_area = lambda ui_components: ui_components.get('confirmation_area')
get_status_panel = lambda ui_components: ui_components.get('status_panel')
get_progress_widget = lambda ui_components: ui_components.get('progress')
get_action_buttons = lambda ui_components: ui_components.get('action_buttons', {})

# Status shortcuts
log_info = lambda ui_components, msg: log_to_ui(ui_components, msg, 'info')
log_success = lambda ui_components, msg: log_to_ui(ui_components, msg, 'success')
log_warning = lambda ui_components, msg: log_to_ui(ui_components, msg, 'warning')
log_error = lambda ui_components, msg: log_to_ui(ui_components, msg, 'error')

# Progress shortcuts
start_progress = lambda ui_components, msg: setup_progress(ui_components, msg)
finish_progress = lambda ui_components, msg: complete_progress(ui_components, msg)
fail_progress = lambda ui_components, msg: error_progress(ui_components, msg)