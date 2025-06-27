"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: UI utilities untuk preprocessing handlers dengan error handling dan logging
"""

from typing import Dict, Any, Optional, Union
from IPython.display import display, HTML
import datetime

# Import logger bridge with fallback
try:
    from smartcash.ui.utils.logger_bridge import UILoggerBridge
except ImportError:
    UILoggerBridge = None

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Log message ke UI dengan timestamp dan styling
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan
        level: Level log (info, success, warning, error)
    """
    try:
        # Try using logger_bridge if available
        if 'logger_bridge' in ui_components and ui_components['logger_bridge']:
            logger = ui_components['logger_bridge']
            log_method = getattr(logger, level.lower(), logger.info)
            log_method(message)
            return
            
        # Fallback to direct UI logging
        log_output = ui_components.get('log_output')
        if not log_output:
            print(f"[{level.upper()}] {message}")
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
            
    except Exception as e:
        # Fallback ke print jika UI logging gagal
        print(f"[{level.upper()}] {message}")
        print(f"Logging error: {str(e)}")

def hide_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Hide confirmation area dengan visibility control
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.visibility = 'hidden'
            confirmation_area.layout.height = '0px'
            if logger_bridge:
                logger_bridge.debug("Menyembunyikan area konfirmasi")
    except Exception as e:
        error_msg = f"Gagal menyembunyikan area konfirmasi: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def show_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Show confirmation area
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        confirmation_area = ui_components.get('confirmation_area')
        if confirmation_area and hasattr(confirmation_area, 'layout'):
            confirmation_area.layout.visibility = 'visible'
            confirmation_area.layout.height = 'auto'
            if logger_bridge:
                logger_bridge.debug("Menampilkan area konfirmasi")
    except Exception as e:
        error_msg = f"Gagal menampilkan area konfirmasi: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def clear_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear outputs dengan hiding confirmation area
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        if logger_bridge:
            logger_bridge.debug("Membersihkan output dan menyembunyikan area konfirmasi")
        hide_confirmation_area(ui_components)
    except Exception as e:
        error_msg = f"Gagal membersihkan output: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def disable_buttons(ui_components: Dict[str, Any]) -> None:
    """Disable operation buttons during processing
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        if logger_bridge:
            logger_bridge.debug("Menonaktifkan tombol operasi")
            
        button_ids = ['preprocess_btn', 'check_btn', 'cleanup_btn']
        disabled_count = 0
        
        for btn_id in button_ids:
            if btn_id in ui_components and hasattr(ui_components[btn_id], 'disabled'):
                ui_components[btn_id].disabled = True
                disabled_count += 1
                
        if logger_bridge and disabled_count > 0:
            logger_bridge.debug(f"Berhasil menonaktifkan {disabled_count} tombol")
            
    except Exception as e:
        error_msg = f"Gagal menonaktifkan tombol: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def enable_buttons(ui_components: Dict[str, Any]) -> None:
    """Enable operation buttons after processing
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        if logger_bridge:
            logger_bridge.debug("Mengaktifkan tombol operasi")
            
        button_ids = ['preprocess_btn', 'check_btn', 'cleanup_btn']
        enabled_count = 0
        
        for btn_id in button_ids:
            if btn_id in ui_components and hasattr(ui_components[btn_id], 'disabled'):
                ui_components[btn_id].disabled = False
                enabled_count += 1
                
        if logger_bridge and enabled_count > 0:
            logger_bridge.debug(f"Berhasil mengaktifkan {enabled_count} tombol")
            
    except Exception as e:
        error_msg = f"Gagal mengaktifkan tombol: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def handle_error(ui_components: Dict[str, Any], error_msg: str, logger: Optional[UILoggerBridge] = None) -> None:
    """Handle error dengan logging dan cleanup
    
    Args:
        ui_components: Dictionary berisi komponen UI
        error_msg: Pesan error yang akan ditampilkan
        logger: Optional logger instance (UILoggerBridge)
    """
    # Prioritize passed logger, fallback to ui_components logger
    logger_bridge = logger or (ui_components.get('logger_bridge') if ui_components else None)
    
    try:
        # Log error ke UI
        if logger_bridge:
            logger_bridge.error(f"❌ {error_msg}")
        else:
            # Fallback ke log_to_ui jika logger_bridge tidak tersedia
            log_to_ui(ui_components, error_msg, 'error')
        
        # Enable buttons untuk memastikan UI tidak terkunci
        try:
            enable_buttons(ui_components)
            if logger_bridge:
                logger_bridge.debug("Tombol operasi diaktifkan ulang setelah error")
        except Exception as btn_error:
            btn_error_msg = f"Gagal mengaktifkan tombol setelah error: {str(btn_error)}"
            if logger_bridge:
                logger_bridge.error(btn_error_msg)
            else:
                print(f"[ERROR] {btn_error_msg}")
                
    except Exception as e:
        # Fallback ke console logging jika terjadi error
        error_details = f"Gagal menangani error: {str(e)}\nError asli: {error_msg}"
        if logger_bridge:
            logger_bridge.critical(f"Kegagalan kritis dalam handle_error: {error_details}")
        else:
            print(f"[CRITICAL] {error_details}")

def setup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Setup progress tracker untuk operation
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan di progress tracker
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        if logger_bridge:
            logger_bridge.debug(f"Menyiapkan progress tracker: {message}")
            
        progress = ui_components.get('progress')
        if progress and hasattr(progress, 'value'):
            progress.bar_style = ''
            progress.value = 0
            progress.description = message
            
            if logger_bridge:
                logger_bridge.debug("Progress tracker berhasil disiapkan")
    except Exception as e:
        error_msg = f"Gagal menyiapkan progress tracker: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def complete_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete progress tracker dengan success message
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan ditampilkan di progress tracker
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        if logger_bridge:
            logger_bridge.debug(f"Menyelesaikan progress tracker: {message}")
            
        progress = ui_components.get('progress')
        if progress and hasattr(progress, 'value'):
            progress.bar_style = 'success'
            progress.value = 100
            progress.description = message
            
            if logger_bridge:
                logger_bridge.debug("Progress tracker berhasil diselesaikan")
    except Exception as e:
        error_msg = f"Gagal menyelesaikan progress tracker: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def error_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state pada progress tracker
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan error yang akan ditampilkan di progress tracker
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        if logger_bridge:
            logger_bridge.error(f"Error pada progress tracker: {message}")
            
        progress = ui_components.get('progress')
        if progress and hasattr(progress, 'bar_style'):
            progress.bar_style = 'danger'
            progress.description = f"Error: {message}"
            
            if logger_bridge:
                logger_bridge.debug("Status error berhasil ditetapkan pada progress tracker")
    except Exception as e:
        error_msg = f"Gagal menetapkan status error pada progress tracker: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        raise ValueError(error_msg) from e

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = "info") -> None:
    """Fallback log function untuk accordion output
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan dicatat
        level: Level log (info, warning, error, success)
    """
    logger_bridge = ui_components.get('logger_bridge') if ui_components else None
    
    try:
        if logger_bridge:
            # Gunakan logger bridge jika tersedia
            log_method = getattr(logger_bridge, level.lower(), logger_bridge.info)
            log_method(f"[Accordion] {message}")
            return
            
        # Fallback ke accordion logging
        accordion = ui_components.get('log_accordion')
        if accordion and hasattr(accordion, 'children') and accordion.children:
            log_output = accordion.children[0]
            if hasattr(log_output, 'append_stdout'):
                log_output.append_stdout(f"[{level.upper()}] {message}\n")
            
    except Exception as e:
        error_msg = f"Gagal mencatat ke accordion: {str(e)}"
        if logger_bridge:
            logger_bridge.error(error_msg)
        print(f"[ERROR] {error_msg}")

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