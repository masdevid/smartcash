"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Utility untuk mengarahkan output logger ke UI widget dengan integrasi standar dan tanpa duplikasi
"""

import logging
import sys
import threading
from typing import Dict, Any, Optional, Union
from IPython.display import display, HTML

def create_direct_ui_logger(ui_components: Dict[str, Any], name: str = "ui_logger"):
    """
    Buat logger yang langsung menampilkan output ke UI tanpa addHandler.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        name: Nama logger
        
    Returns:
        Logger yang dikonfigurasi
    """
    # Import komponen secara strategis untuk menghindari circular imports
    try:
        from smartcash.common.logger import get_logger, SmartCashLogger, LogLevel
        
        # Sebelum membuat logger kustom, pastikan ada widget output
        if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
            # Fallback ke logger standar tanpa UI
            return get_logger(name)
        
        # Setup LogLevel ke DEBUG untuk mengizinkan semua level log
        logger = get_logger(name, LogLevel.DEBUG)
        
        # Override fungsi log
        original_log = logger.log
        
        def ui_log(level, message):
            # Call asli untuk konsistensi dan file logging
            original_log(level, message)
            
            # Log ke UI
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                # Map LogLevel ke status type
                level_name = level.name.lower() if hasattr(level, 'name') else 'info'
                status_type = "success" if level_name == "success" else \
                             "warning" if level_name in ("warning", "warn") else \
                             "error" if level_name in ("error", "critical") else "info"
                display(create_status_indicator(status_type, message))
        
        # Ganti implementasi log dengan versi kustom
        logger.log = ui_log
        return logger
    
    except ImportError:
        # Fallback ke UI Logger sederhana dengan logging standar
        class SimpleUILogger:
            def __init__(self, name, ui_components):
                self.name = name
                self.ui_components = ui_components
                self.logger = logging.getLogger(name)
                
            def _log_ui(self, message, level):
                if 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                    with self.ui_components['status']:
                        try:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            display(create_status_indicator(level, message))
                        except ImportError:
                            # Ultra-minimal fallback dengan emoji generik
                            emoji = "✅" if level == "success" else \
                                   "⚠️" if level == "warning" else \
                                   "❌" if level == "error" else "ℹ️"
                            display(HTML(f"<div><span>{emoji}</span> {message}</div>"))
            
            def debug(self, message):
                self.logger.debug(message)
                self._log_ui(message, "info")
                
            def info(self, message):
                self.logger.info(message)
                self._log_ui(message, "info")
                
            def success(self, message):
                self.logger.info(message)
                self._log_ui(message, "success")
                
            def warning(self, message):
                self.logger.warning(message)
                self._log_ui(message, "warning")
                
            def error(self, message):
                self.logger.error(message)
                self._log_ui(message, "error")
                
            def critical(self, message):
                self.logger.critical(message)
                self._log_ui(message, "error")
            
            # Kompatibilitas dengan SmartCashLogger
            def log(self, level, message):
                level_name = level.name.lower() if hasattr(level, 'name') else str(level).lower()
                if level_name in ("debug",):
                    self.debug(message)
                elif level_name in ("info",):
                    self.info(message)
                elif level_name in ("success",):
                    self.success(message)
                elif level_name in ("warning", "warn"):
                    self.warning(message)
                elif level_name in ("error", "critical"):
                    self.error(message)
                else:
                    self.info(message)
            
        return SimpleUILogger(name, ui_components)


def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", emoji: str = "") -> None:
    """
    Log pesan langsung ke UI widget menggunakan komponen terpadu.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, debug, success)
        emoji: Emoji untuk ditampilkan (akan ditambahkan ke pesan)
    """
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        # Fallback sederhana: print pesan
        print(f"{emoji} {message}")
        return
    
    with ui_components['status']:
        try:
            # Gunakan komponen alert_utils jika tersedia
            from smartcash.ui.utils.alert_utils import create_status_indicator
            # Tambahkan emoji ke pesan jika disediakan
            full_message = f"{emoji} {message}" if emoji else message
            display(create_status_indicator(level, full_message))
        except ImportError:
            # Fallback minimal tanpa alert_utils - reuse kode
            emoji_map = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌", "debug": "🔍"}
            icon = emoji or emoji_map.get(level, "ℹ️")
            display(HTML(f"<div><span>{icon}</span> {message}</div>"))


def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """
    Intercept stdout dan arahkan ke UI widget.
    Metode yang lebih clean dengan satu implementasi dan thread lock.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
    """
    # Pastikan ada status output widget
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        return
    
    # Pastikan tidak terjadi multiple intercepts
    if 'custom_stdout' in ui_components and ui_components.get('custom_stdout') == sys.stdout:
        return
    
    # Buat stdout interceptor dengan thread-safety
    class UIStdoutInterceptor:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.terminal = sys.stdout
            self.buffer = ""
            self.lock = threading.RLock()
            self.buffer_limit = 1000  # Batasi buffer untuk mencegah memory leak
            
        def write(self, message):
            # Write ke terminal asli
            self.terminal.write(message)
            
            # Buffer output sampai ada newline, dengan thread-safety
            with self.lock:
                # Batasi ukuran buffer
                if len(self.buffer) > self.buffer_limit:
                    self.buffer = self.buffer[-self.buffer_limit:]
                
                self.buffer += message
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]  # Simpan baris terakhir yang belum lengkap
                    
                    # Tampilkan setiap baris lengkap
                    for line in lines[:-1]:
                        if line.strip():  # Cek jika bukan baris kosong
                            try:
                                with self.ui_components['status']:
                                    try:
                                        from smartcash.ui.utils.alert_utils import create_status_indicator
                                        display(create_status_indicator("info", line))
                                    except ImportError:
                                        display(HTML(f"<div>{line}</div>"))
                            except Exception:
                                # Jika ada error saat menampilkan ke UI, kirim ke stdout asli
                                self.terminal.write(f"[UI STDOUT ERROR] {line}\n")
        
        def flush(self):
            self.terminal.flush()
            # Flush buffer jika perlu, dengan thread-safety
            with self.lock:
                if self.buffer:
                    try:
                        with self.ui_components['status']:
                            try:
                                from smartcash.ui.utils.alert_utils import create_status_indicator
                                display(create_status_indicator("info", self.buffer))
                            except ImportError:
                                display(HTML(f"<div>{self.buffer}</div>"))
                    except Exception:
                        # Fallback ke stdout asli
                        self.terminal.write(f"[UI STDOUT ERROR] {self.buffer}\n")
                    self.buffer = ""
        
        # Kebutuhan IOBase lainnya
        def isatty(self):
            return False
            
        def fileno(self):
            return self.terminal.fileno()
    
    # Simpan stdout original dan replace dengan interceptor
    original_stdout = sys.stdout
    ui_components['original_stdout'] = original_stdout
    
    # Pasang interceptor
    interceptor = UIStdoutInterceptor(ui_components)
    sys.stdout = interceptor
    ui_components['custom_stdout'] = interceptor