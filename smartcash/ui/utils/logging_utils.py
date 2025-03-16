"""
File: smartcash/ui/utils/logging_utils.py
Deskripsi: Utilitas untuk logging di notebook dengan tampilan berbasis UI alerts
"""

import logging
import sys
from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.utils.constants import ALERT_STYLES, ICONS

class UILogHandler(logging.Handler):
    """Custom logging handler yang menampilkan log di UI output widget."""
    
    def __init__(self, output_widget: widgets.Output):
        """
        Initialize UILogHandler.
        
        Args:
            output_widget: Widget output untuk menampilkan log
        """
        super().__init__()
        self.output_widget = output_widget
        self.setFormatter(logging.Formatter('%(message)s'))
    
    def emit(self, record):
        """Tampilkan log di widget output dengan styling alert."""
        try:
            # Maps logging levels to alert styles
            level_to_style = {
                logging.DEBUG: 'info',
                logging.INFO: 'info',
                logging.WARNING: 'warning', 
                logging.ERROR: 'error',
                logging.CRITICAL: 'error',
            }
            
            # Maps logging levels to icons
            level_to_icon = {
                logging.DEBUG: ICONS.get('info', 'ℹ️'),
                logging.INFO: ICONS.get('info', 'ℹ️'),
                logging.WARNING: ICONS.get('warning', '⚠️'),
                logging.ERROR: ICONS.get('error', '❌'),
                logging.CRITICAL: ICONS.get('error', '❌'),
            }
            
            style = level_to_style.get(record.levelno, 'info')
            alert_style = ALERT_STYLES.get(style, ALERT_STYLES['info'])
            icon = level_to_icon.get(record.levelno, ICONS.get('info', 'ℹ️'))
            
            # Format log message dengan styling alert
            message = self.format(record)
            log_html = f"""
            <div style="padding: 5px; margin: 2px 0; 
                      background-color: {alert_style['bg_color']}; 
                      color: {alert_style['text_color']}; 
                      border-left: 4px solid {alert_style['border_color']}; 
                      border-radius: 4px;">
                <span style="margin-right: 8px;">{icon}</span>
                <span>{message}</span>
            </div>
            """
            
            # Display di output widget
            with self.output_widget:
                display(HTML(log_html))
                
        except Exception:
            # Fallback ke default behavior
            super().emit(record)

class UILogger(logging.Logger):
    """Logger khusus dengan metode yang disesuaikan untuk alerts styling."""
    
    def success(self, msg, *args, **kwargs):
        """
        Log dengan level INFO tapi dengan style success.
        
        Args:
            msg: Pesan log
            *args, **kwargs: Argumen tambahan
        """
        if self.isEnabledFor(logging.INFO):
            # Tambahkan emoji success jika belum ada
            if not any(emoji in msg for emoji in ['✅', '✓', '🎉']):
                msg = f"✅ {msg}"
            self._log(logging.INFO, msg, args, **kwargs)

def setup_ipython_logging(
    ui_components: Dict[str, Any],
    logger_name: str = 'ui_logger',
    log_level: int = logging.INFO,
    clear_existing_handlers: bool = True
) -> Optional[logging.Logger]:
    """
    Setup logger untuk notebook dengan output ke widget.
    
    Args:
        ui_components: Dictionary berisi widget UI
        logger_name: Nama logger
        log_level: Level logging
        clear_existing_handlers: Hapus handler yang sudah ada
        
    Returns:
        Logger instance atau None jika gagal
    """
    # Register custom logger class
    logging.setLoggerClass(UILogger)
    
    # Dapatkan output widget
    output_widget = None
    for key in ['status', 'log_output', 'output']:
        if key in ui_components and isinstance(ui_components[key], widgets.Output):
            output_widget = ui_components[key]
            break
    
    if not output_widget:
        return None
    
    # Dapatkan logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Hapus handler existing jika perlu
    if clear_existing_handlers:
        logger.handlers = []
    
    # Tambahkan UI handler
    ui_handler = UILogHandler(output_widget)
    ui_handler.setLevel(log_level)
    logger.addHandler(ui_handler)
    
    # Disable propagation ke root logger agar tidak muncul di console
    logger.propagate = False
    
    return logger

def create_dummy_logger() -> logging.Logger:
    """
    Buat dummy logger yang tidak melakukan output.
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('dummy_logger')
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    
    return logger

def log_to_ui(
    ui_components: Dict[str, Any],
    message: str,
    level: str = 'info'
) -> None:
    """
    Log pesan langsung ke UI tanpa logger.
    
    Args:
        ui_components: Dictionary berisi widget UI
        message: Pesan yang akan ditampilkan
        level: Level log ('info', 'success', 'warning', 'error')
    """
    # Dapatkan output widget
    output_widget = None
    for key in ['status', 'log_output', 'output']:
        if key in ui_components and isinstance(ui_components[key], widgets.Output):
            output_widget = ui_components[key]
            break
    
    if not output_widget:
        return
    
    # Maps level ke alert style
    alert_style = ALERT_STYLES.get(level, ALERT_STYLES['info'])
    icon = alert_style.get('icon', ICONS.get(level, ICONS.get('info', 'ℹ️')))
    
    # Format log message
    log_html = f"""
    <div style="padding: 5px; margin: 2px 0; 
              background-color: {alert_style['bg_color']}; 
              color: {alert_style['text_color']}; 
              border-left: 4px solid {alert_style['border_color']}; 
              border-radius: 4px;">
        <span style="margin-right: 8px;">{icon}</span>
        <span>{message}</span>
    </div>
    """
    
    # Display di output widget
    with output_widget:
        display(HTML(log_html))