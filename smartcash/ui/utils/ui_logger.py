"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Utility untuk mengarahkan output logger ke UI widget
"""

import logging
import sys
from typing import Dict, Any, Optional, Union
from IPython.display import display, HTML

def create_ui_logger_handler(ui_components: Dict[str, Any]) -> Optional[logging.Handler]:
    """
    Buat handler logging yang mengarahkan output ke widget UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        
    Returns:
        Handler logging atau None jika tidak dapat dibuat
    """
    if 'status' not in ui_components:
        return None
        
    # Buat handler kustom yang mengirim output ke widget output
    class IPythonWidgetHandler(logging.Handler):
        def __init__(self, output_widget):
            logging.Handler.__init__(self)
            self.output_widget = output_widget
            
        def emit(self, record):
            try:
                # Format pesan
                msg = self.format(record)
                
                # Tambahkan warna berdasarkan level
                color_map = {
                    logging.DEBUG: 'gray',
                    logging.INFO: 'blue',
                    logging.WARNING: 'orange',
                    logging.ERROR: 'red',
                    logging.CRITICAL: 'darkred'
                }
                
                emoji_map = {
                    logging.DEBUG: '🐞',
                    logging.INFO: 'ℹ️',
                    logging.WARNING: '⚠️',
                    logging.ERROR: '❌',
                    logging.CRITICAL: '🔥'
                }
                
                # Dapatkan warna dan emoji
                color = color_map.get(record.levelno, 'black')
                emoji = emoji_map.get(record.levelno, '')
                
                # Create styled HTML
                html_msg = f"""<div style="margin: 2px 0; color: {color}; overflow-wrap: break-word;">
                    <span>{emoji} <b>{record.levelname}:</b></span> {msg}
                </div>"""
                
                # Tampilkan di widget
                with self.output_widget:
                    display(HTML(html_msg))
            except Exception:
                # Jangan biarkan error di handler menyebabkan masalah
                pass
    
    # Buat handler
    handler = IPythonWidgetHandler(ui_components['status'])
    
    # Set formatter
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    return handler

def redirect_logger_to_ui(logger: logging.Logger, ui_components: Dict[str, Any]) -> bool:
    """
    Arahkan logger ke output UI.
    
    Args:
        logger: Logger yang akan diarahkan
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        
    Returns:
        True jika berhasil, False jika gagal
    """
    # Buat handler
    handler = create_ui_logger_handler(ui_components)
    if not handler:
        return False
    
    # Set level handler
    handler.setLevel(logging.DEBUG)
    
    # Tambahkan handler ke logger
    logger.addHandler(handler)
    
    return True

def setup_root_logger_ui_redirect(ui_components: Dict[str, Any], level: int = logging.INFO) -> None:
    """
    Setup root logger untuk mengarahkan semua log ke UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        level: Level logging (default: INFO)
    """
    # Dapatkan root logger
    root_logger = logging.getLogger()
    
    # Set level
    root_logger.setLevel(level)
    
    # Tambahkan handler
    handler = create_ui_logger_handler(ui_components)
    if handler:
        handler.setLevel(level)
        root_logger.addHandler(handler)

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", emoji: str = "") -> None:
    """
    Log pesan langsung ke UI tanpa melalui logger.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, debug, success)
        emoji: Emoji untuk ditampilkan (opsional)
    """
    if 'status' not in ui_components:
        return
        
    # Map level ke warna
    color_map = {
        'debug': 'gray',
        'info': 'blue',
        'success': 'green',
        'warning': 'orange',
        'error': 'red'
    }
    
    # Default emoji berdasarkan level jika tidak disediakan
    if not emoji:
        emoji_map = {
            'debug': '🐞',
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        emoji = emoji_map.get(level, '')
    
    # Get color
    color = color_map.get(level, 'black')
    
    # Create HTML
    html_msg = f"""<div style="margin: 2px 0; color: {color}; overflow-wrap: break-word;">
        <span>{emoji} <b>{level.upper()}:</b></span> {message}
    </div>"""
    
    # Display in widget
    with ui_components['status']:
        display(HTML(html_msg))