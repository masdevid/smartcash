"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: Fixed UI utilities dengan scroll fix dan simplified logging
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from datetime import datetime
from smartcash.common.logger import get_logger

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Fixed log to accordion tanpa scroll_to_bottom error"""
    try:
        # Get logger dengan fallback
        logger = ui_components.get('logger') or get_logger('preprocessing_ui')
        
        # Clean message untuk console
        console_message = _clean_html_message(message)
        
        # Log ke console
        log_methods = {
            'info': logger.info,
            'success': getattr(logger, 'success', logger.info),
            'warning': logger.warning,
            'error': logger.error,
            'debug': logger.debug
        }
        
        log_method = log_methods.get(level.lower(), logger.info)
        log_method(console_message)
        
        # Update UI log dengan safe handling
        _update_ui_log_safe(ui_components, message, level)
        
        # Auto-expand untuk important messages
        if level.lower() in ['error', 'warning']:
            _expand_log_accordion(ui_components)
                
    except Exception as e:
        # Safe fallback
        print(f"[{level.upper()}] {_clean_html_message(message)}")
        print(f"Log error: {str(e)}")

def _update_ui_log_safe(ui_components: Dict[str, Any], message: str, level: str):
    """Safe UI log update tanpa scroll errors"""
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    try:
        # Create styled message
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            'info': '#2196F3',
            'success': '#4CAF50',
            'warning': '#FF9800', 
            'error': '#F44336',
            'debug': '#9E9E9E'
        }
        
        color = colors.get(level.lower(), '#2196F3')
        
        formatted_msg = f"""
        <div style='margin: 3px 0; padding: 6px; border-left: 3px solid {color}; 
                    background-color: rgba(248, 249, 250, 0.8);'>
            <span style='color: #757575; font-size: 11px;'>{timestamp}</span>
            <div style='margin-top: 2px; font-size: 13px;'>{message}</div>
        </div>
        """
        
        # Display tanpa scroll method
        with log_output:
            display(HTML(formatted_msg))
            
    except Exception:
        # Silent fail to prevent UI break
        pass

def _expand_log_accordion(ui_components: Dict[str, Any]):
    """Safely expand log accordion"""
    try:
        log_accordion = ui_components.get('log_accordion')
        if log_accordion and hasattr(log_accordion, 'selected_index'):
            log_accordion.selected_index = 0
    except Exception:
        pass

def _clean_html_message(message: str) -> str:
    """Remove HTML tags dari message untuk console logging"""
    if '<' in message and '>' in message:
        try:
            import re
            return re.sub(r'<[^>]+>', '', message).strip()
        except Exception:
            return message
    return message

def clear_outputs(ui_components: Dict[str, Any], clear_logs: bool = True, clear_confirm: bool = True):
    """Clear UI outputs dengan safe handling"""
    try:
        # Clear log output
        if clear_logs and 'log_output' in ui_components:
            log_output = ui_components['log_output']
            if hasattr(log_output, 'clear_output'):
                with log_output:
                    clear_output(wait=True)
        
        # Clear confirmation area
        if clear_confirm and 'confirmation_area' in ui_components:
            from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import clear_confirmation_area
            clear_confirmation_area(ui_components)
        
        # Reset progress tracker
        _reset_progress_tracker_safe(ui_components)
        
        return True
        
    except Exception as e:
        get_logger('preprocessing_ui').error(f"Error clearing outputs: {str(e)}")
        return False

def _reset_progress_tracker_safe(ui_components: Dict[str, Any]):
    """Safely reset progress tracker"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
    except Exception:
        pass

def handle_ui_error(ui_components: Dict[str, Any], error_msg: str, exception: Optional[Exception] = None):
    """Handle error dengan UI updates"""
    try:
        logger = get_logger('preprocessing_ui')
        
        # Format error message
        full_error_msg = f"❌ {error_msg}"
        if exception:
            full_error_msg += f"\nDetail: {str(exception)}"
        
        # Log error
        logger.error(full_error_msg)
        
        # Update progress tracker
        _update_progress_error(ui_components, error_msg)
        
        # Show error in UI
        error_html = f"""
        <div style='margin: 10px 0; padding: 12px; 
                   background-color: #ffebee; 
                   border-left: 4px solid #f44336;
                   border-radius: 4px;'>
            <h4 style='margin: 0 0 8px 0; color: #c62828;'>⚠️ Terjadi Kesalahan</h4>
            <p style='margin: 0; color: #d32f2f;'>{error_msg}</p>
        </div>
        """
        
        log_to_accordion(ui_components, error_html, 'error')
        
        # Update status panel
        _update_status_panel_safe(ui_components, error_msg, 'error')
        
        return False
        
    except Exception as e:
        # Fallback error handling
        print(f"❌ {error_msg}")
        if exception:
            print(f"Detail: {str(exception)}")
        print(f"UI Error handler failed: {str(e)}")
        return False

def show_ui_success(ui_components: Dict[str, Any], message: str):
    """Show success message dengan UI updates"""
    try:
        logger = get_logger('preprocessing_ui')
        
        # Log success
        success_msg = f"✅ {message}"
        logger.info(success_msg)
        
        # Update progress tracker
        _update_progress_success(ui_components, message)
        
        # Show success in UI
        success_html = f"""
        <div style='margin: 10px 0; padding: 12px; 
                   background-color: #e8f5e9; 
                   border-left: 4px solid #4caf50;
                   border-radius: 4px;'>
            <h4 style='margin: 0 0 8px 0; color: #2e7d32;'>✅ Berhasil</h4>
            <p style='margin: 0; color: #388e3c;'>{message}</p>
        </div>
        """
        
        log_to_accordion(ui_components, success_html, 'success')
        
        # Update status panel
        _update_status_panel_safe(ui_components, message, 'success')
        
        # Show confirmation feedback
        from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import show_success_message
        show_success_message(ui_components, message)
        
        return True
        
    except Exception as e:
        print(f"✅ {message}")
        print(f"UI Success handler failed: {str(e)}")
        return False

def _update_progress_error(ui_components: Dict[str, Any], error_msg: str):
    """Update progress tracker untuk error"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(error_msg)
    except Exception:
        pass

def _update_progress_success(ui_components: Dict[str, Any], message: str):
    """Update progress tracker untuk success"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
    except Exception:
        pass

def _update_status_panel_safe(ui_components: Dict[str, Any], message: str, status_type: str):
    """Safe update status panel"""
    try:
        from smartcash.ui.components.status_panel import update_status_panel
        status_panel = ui_components.get('status_panel')
        if status_panel:
            update_status_panel(status_panel, message, status_type)
    except Exception:
        pass

def log_preprocessing_config(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Log preprocessing configuration"""
    try:
        preprocessing = config.get('preprocessing', {})
        normalization = preprocessing.get('normalization', {})
        target_size = normalization.get('target_size', [640, 640])
        
        config_lines = [
            "🔧 <b>Konfigurasi Preprocessing</b>",
            f"📐 <b>Resolusi:</b> {target_size[0]}x{target_size[1]}",
            f"🎨 <b>Normalisasi:</b> {normalization.get('method', 'minmax')}",
            f"🎯 <b>Target Split:</b> {', '.join(preprocessing.get('target_splits', ['train', 'valid']))}",
            f"✅ <b>Validasi:</b> {'Aktif' if preprocessing.get('validation', {}).get('enabled', True) else 'Nonaktif'}"
        ]
        
        log_to_accordion(ui_components, '<br>'.join(config_lines), 'info')
        
    except Exception as e:
        handle_ui_error(ui_components, f"Gagal menampilkan konfigurasi: {str(e)}")

# One-liner utilities
get_ui_logger = lambda ui_components: ui_components.get('logger') or get_logger('preprocessing_ui')
is_milestone_step = lambda step, progress: progress in [0, 25, 50, 75, 100] or 'error' in step.lower()
safe_log = lambda ui_components, msg, level='info': log_to_accordion(ui_components, msg, level)
safe_clear = lambda ui_components: clear_outputs(ui_components, clear_logs=False, clear_confirm=True)
safe_error = lambda ui_components, msg: handle_ui_error(ui_components, msg)
safe_success = lambda ui_components, msg: show_ui_success(ui_components, msg)