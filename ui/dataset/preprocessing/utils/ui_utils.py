"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: UI utilities untuk preprocessing handlers dengan error handling dan logging
"""

from typing import Dict, Any
from IPython.display import display, HTML
import datetime

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Log message ke UI dengan timestamp dan styling"""
    try:
        log_output = ui_components.get('log_output')
        if not log_output:
            return
            
        # Color mapping untuk different levels
        colors = {
            'info': '#2196F3', 'success': '#4CAF50', 
            'warning': '#FF9800', 'error': '#F44336'
        }
        color = colors.get(level, '#2196F3')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Clean message dari duplicate emoji
        clean_msg = message.strip()
        
        html = f"""
        <div style='margin: 2px 0; padding: 4px 8px; border-left: 3px solid {color}; 
                    background-color: rgba(248,249,250,0.8); border-radius: 4px;'>
            <span style='color: #666; font-size: 11px;'>[{timestamp}]</span>
            <span style='color: {color}; margin-left: 4px;'>{clean_msg}</span>
        </div>
        """
        
        with log_output:
            display(HTML(html))
            
    except Exception:
        # Fallback ke print jika UI logging gagal
        print(f"[{level.upper()}] {message}")

def hide_confirmation_area(ui_components: Dict[str, Any]):
    """Hide confirmation area dengan visibility control"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'hidden'
        confirmation_area.layout.height = '0px'

def show_confirmation_area(ui_components: Dict[str, Any]):
    """Show confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'

def clear_outputs(ui_components: Dict[str, Any]):
    """Clear outputs dengan hiding confirmation area"""
    hide_confirmation_area(ui_components)

def disable_buttons(ui_components: Dict[str, Any]):
    """Disable operation buttons during processing"""
    button_keys = ['preprocess_button', 'check_button', 'cleanup_button']
    for btn_key in button_keys:
        if button := ui_components.get(btn_key):
            if hasattr(button, 'disabled'):
                button.disabled = True

def enable_buttons(ui_components: Dict[str, Any]):
    """Enable operation buttons after processing"""
    button_keys = ['preprocess_button', 'check_button', 'cleanup_button']
    for btn_key in button_keys:
        if button := ui_components.get(btn_key):
            if hasattr(button, 'disabled'):
                button.disabled = False

def handle_error(ui_components: Dict[str, Any], error_msg: str):
    """Handle error dengan logging dan cleanup"""
    log_to_ui(ui_components, error_msg, "error")
    error_progress(ui_components, error_msg)
    enable_buttons(ui_components)

def setup_progress(ui_components: Dict[str, Any], message: str):
    """Setup progress tracker untuk operation"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'show'):
            progress_tracker.show()
        if hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, message)

def complete_progress(ui_components: Dict[str, Any], message: str):
    """Complete progress tracker dengan success message"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(100, message)

def error_progress(ui_components: Dict[str, Any], message: str):
    """Set error state pada progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        if hasattr(progress_tracker, 'error'):
            progress_tracker.error(message)
        elif hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(0, f"❌ {message}")

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Fallback log function untuk accordion output"""
    try:
        log_output = ui_components.get('log_accordion', {}).get('log_output')
        if log_output and hasattr(log_output, 'append_log'):
            log_output.append_log(message, level)
        else:
            log_to_ui(ui_components, message, level)
    except Exception:
        print(f"[{level.upper()}] {message}")

# Convenience aliases
_log_to_ui = log_to_ui
_hide_confirmation_area = hide_confirmation_area
_show_confirmation_area = show_confirmation_area
_clear_outputs = clear_outputs
_disable_buttons = disable_buttons
_enable_buttons = enable_buttons
_handle_error = handle_error
_setup_progress = setup_progress
_complete_progress = complete_progress
_error_progress = error_progress