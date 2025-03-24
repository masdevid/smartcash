"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Utility untuk mengarahkan output logger ke UI widget dengan styling komponen yang sudah ada
"""

import logging
import sys
from typing import Dict, Any, Optional, Union
from IPython.display import display, HTML
from smartcash.ui.utils.constants import ALERT_STYLES, ICONS
def create_direct_ui_logger(ui_components: Dict[str, Any], name: str = "ui_logger"):
    """
    Buat logger yang langsung menampilkan output ke UI tanpa addHandler.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        name: Nama logger
        
    Returns:
        Logger yang dikonfigurasi
    """
    # Import komponen UI yang sudah ada
    try:
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.common.logger import SmartCashLogger, LogLevel, get_logger
        
        # Extend SmartCashLogger untuk override metode log
        class UISmartCashLogger(SmartCashLogger):
            def __init__(self, name, *args, **kwargs):
                self.ui_components = ui_components
                super().__init__(name, *args, **kwargs)
                
            def log(self, level, message):
                # Tetap log di aslinya untuk file
                super().log(level, message)
                
                # Log ke UI juga menggunakan komponen yang ada
                if 'status' not in self.ui_components:
                    return
                
                level_name = level.name if hasattr(level, 'name') else 'INFO'
                level_str = level_name.lower()
                
                # Map level ke status type untuk create_status_indicator
                status_map = {
                    'debug': 'info',
                    'info': 'info', 
                    'success': 'success',
                    'warning': 'warning',
                    'error': 'error',
                    'critical': 'error'
                }
                status_type = status_map.get(level_str, 'info')
                
                # Log ke UI dengan create_status_indicator
                with self.ui_components['status']:
                    display(create_status_indicator(status_type, message))
        
        # Buat instance UISmartCashLogger
        return UISmartCashLogger(name, LogLevel.DEBUG, log_file=None)
    
    except ImportError:
        # Fallback ke console_logger jika SmartCashLogger tidak tersedia
        console_logger = logging.getLogger(name)
        
        try:
            from smartcash.ui.utils.alert_utils import create_status_indicator
            
            class UILogger:
                def __init__(self, name, ui_components):
                    self.name = name
                    self.ui_components = ui_components
                    self.console_logger = logging.getLogger(name)
                    
                def _log_to_ui(self, msg, level="info"):
                    if 'status' not in self.ui_components:
                        return
                        
                    # Map level ke status type
                    status_map = {
                        'debug': 'info',
                        'info': 'info', 
                        'success': 'success',
                        'warning': 'warning',
                        'error': 'error',
                        'critical': 'error'
                    }
                    status_type = status_map.get(level, 'info')
                    
                    # Display menggunakan komponen yang sudah ada
                    with self.ui_components['status']:
                        display(create_status_indicator(status_type, msg))
                        
                def debug(self, msg):
                    self.console_logger.debug(msg)
                    self._log_to_ui(msg, "debug")
                    
                def info(self, msg):
                    self.console_logger.info(msg)
                    self._log_to_ui(msg, "info")
                    
                def success(self, msg):
                    self.console_logger.info(msg)  # success maps to info in std logging
                    self._log_to_ui(msg, "success")
                    
                def warning(self, msg):
                    self.console_logger.warning(msg)
                    self._log_to_ui(msg, "warning")
                    
                def error(self, msg):
                    self.console_logger.error(msg)
                    self._log_to_ui(msg, "error")
                    
                def critical(self, msg):
                    self.console_logger.critical(msg)
                    self._log_to_ui(msg, "critical")
            
            # Return UI logger
            return UILogger(name, ui_components)
        
        except ImportError:
            # Jika create_status_indicator juga tidak tersedia, fallback ke logger biasa
            return console_logger

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", emoji: str = "") -> None:
    """
    Log pesan langsung ke UI menggunakan komponen UI yang sudah ada.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, debug, success)
        emoji: Emoji untuk ditampilkan (diabaikan, menggunakan styling dari alert_utils)
    """
    if 'status' not in ui_components:
        return
        
    try:
        # Gunakan create_status_indicator dari alert_utils
        from smartcash.ui.utils.alert_utils import create_status_indicator
        
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
        
        # Display ke UI widget
        with ui_components['status']:
            display(HTML(log_html))
            
    except ImportError:
        # Fallback jika komponen tidak tersedia
        with ui_components['status']:
            display(HTML(f"<div>{message}</div>"))
        
def intercept_cell_utils_logs(ui_components: Dict[str, Any]) -> None:
    """
    Intercept dan redirect output dari cell_utils ke UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    try:
        import sys
        from IPython import get_ipython
        
        # Import komponen UI yang sudah ada
        try:
            from smartcash.ui.utils.alert_utils import create_status_indicator
            
            # Redirect stdout/stderr ke UI handler
            class UIOutputHandler:
                def __init__(self, ui_components):
                    self.ui_components = ui_components
                    self.terminal = sys.stdout
                    self.buffer = ""
                    
                def write(self, message):
                    # Write to terminal
                    self.terminal.write(message)
                    
                    # Collect output until we get a complete line
                    self.buffer += message
                    
                    # Check if we have a complete line
                    if '\n' in self.buffer:
                        lines = self.buffer.split('\n')
                        self.buffer = lines[-1]  # Keep incomplete last line
                        
                        # Process complete lines
                        for line in lines[:-1]:
                            if line.strip() and 'status' in self.ui_components:
                                with self.ui_components['status']:
                                    display(create_status_indicator("info", line))
                            
                def flush(self):
                    self.terminal.flush()
                    
                    # Flush any remaining content
                    if self.buffer and 'status' in self.ui_components:
                        with self.ui_components['status']:
                            display(create_status_indicator("info", self.buffer))
                        self.buffer = ""
            
            # Simpan original stdout
            original_stdout = sys.stdout
            
            # Ganti stdout dengan handler kita
            sys.stdout = UIOutputHandler(ui_components)
            
            # Simpan di ui_components agar bisa dikembalikan nanti
            ui_components['original_stdout'] = original_stdout
            ui_components['custom_stdout'] = sys.stdout
        
        except ImportError:
            # Fallback ke implementasi simple
            class SimpleUIOutputHandler:
                def __init__(self, ui_components):
                    self.ui_components = ui_components
                    self.terminal = sys.stdout
                    
                def write(self, message):
                    # Write to terminal
                    self.terminal.write(message)
                    
                    # Write to UI if it's not an empty line
                    if message.strip() and 'status' in self.ui_components:
                        with self.ui_components['status']:
                            display(HTML(f"<div>{message}</div>"))
                            
                def flush(self):
                    self.terminal.flush()
            
            # Simpan original stdout
            original_stdout = sys.stdout
            
            # Ganti stdout dengan handler kita
            sys.stdout = SimpleUIOutputHandler(ui_components)
            
            # Simpan di ui_components agar bisa dikembalikan nanti
            ui_components['original_stdout'] = original_stdout
            ui_components['custom_stdout'] = sys.stdout
    
    except Exception:
        # Jangan gagalkan jika error
        pass