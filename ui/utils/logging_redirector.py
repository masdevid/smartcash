"""
File: smartcash/ui/utils/logging_redirector.py
Deskripsi: Utilitas untuk mengalihkan semua console log ke output log UI
"""

import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path

def redirect_all_logs_to_ui(ui_components: Dict[str, Any]) -> None:
    """
    Alihkan semua console log ke output log UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    if 'logger' not in ui_components:
        print("⚠️ Tidak dapat mengalihkan log: logger tidak ditemukan di ui_components")
        return
    
    # Dapatkan UI logger
    ui_logger = ui_components['logger']
    
    # Redirect stdout/stderr ke UI
    from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui
    intercept_stdout_to_ui(ui_components)
    
    # Redirect semua logger ke UI logger
    redirect_all_loggers_to_ui_logger(ui_logger)
    
    print("✅ Semua console log berhasil dialihkan ke output log UI")

def redirect_all_loggers_to_ui_logger(ui_logger) -> None:
    """
    Alihkan semua logger ke UI logger.
    
    Args:
        ui_logger: Instance UILogger
    """
    # Dapatkan root logger
    root_logger = logging.getLogger()
    
    # Hapus semua handler yang ada
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Buat custom handler yang mengalihkan ke UI logger
    class UILogHandler(logging.Handler):
        def __init__(self, ui_logger):
            super().__init__()
            self.ui_logger = ui_logger
        
        def emit(self, record):
            try:
                # Format pesan
                msg = self.format(record)
                
                # Log ke UI logger berdasarkan level
                if record.levelno >= logging.CRITICAL:
                    self.ui_logger.critical(msg)
                elif record.levelno >= logging.ERROR:
                    self.ui_logger.error(msg)
                elif record.levelno >= logging.WARNING:
                    self.ui_logger.warning(msg)
                else:
                    self.ui_logger.info(msg)
            except Exception:
                self.handleError(record)
    
    # Buat dan tambahkan handler ke root logger
    handler = UILogHandler(ui_logger)
    formatter = logging.Formatter('%(name)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Set level ke DEBUG agar semua log ditangkap
    root_logger.setLevel(logging.DEBUG)
    
    # Alihkan juga SmartCashLogger jika ada
    try:
        from smartcash.common.logger import get_logger, LogLevel
        
        # Dapatkan instance SmartCashLogger
        sc_logger = get_logger()
        
        # Tambahkan callback untuk mengalihkan log ke UI logger
        def ui_log_callback(level, message):
            if level.name == 'DEBUG':
                ui_logger.debug(message)
            elif level.name == 'INFO':
                ui_logger.info(message)
            elif level.name == 'SUCCESS':
                ui_logger.success(message)
            elif level.name == 'WARNING':
                ui_logger.warning(message)
            elif level.name in ('ERROR', 'CRITICAL'):
                ui_logger.error(message)
        
        # Tambahkan callback ke SmartCashLogger
        sc_logger.add_callback(ui_log_callback)
    except ImportError:
        # Tidak ada SmartCashLogger, lewati
        pass

def restore_console_logs(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan console log ke aslinya.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Kembalikan stdout/stderr ke aslinya
    from smartcash.ui.utils.ui_logger import restore_stdout
    restore_stdout(ui_components)
    
    # Reset semua logging handler
    from smartcash.ui.utils.ui_logger import _reset_logging_handlers
    _reset_logging_handlers()
    
    print("✅ Console log dikembalikan ke aslinya") 