"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Utility untuk mengarahkan output logger ke UI widget dengan integrasi standar dan optimasi buffer
"""

import logging
import sys
import threading
import time
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
        
        # Batasi log per detik untuk mengurangi flooding
        logger._last_log_time = 0
        logger._log_buffer = []
        logger._min_log_interval = 0.2  # minimal 200ms antara log
        
        def ui_log(level, message):
            # Batasi frekuensi log dengan buffer
            current_time = time.time()
            if hasattr(logger, '_last_log_time'):
                time_diff = current_time - logger._last_log_time
                if time_diff < logger._min_log_interval:
                    # Tambahkan ke buffer jika terlalu sering
                    if not hasattr(logger, '_log_buffer'):
                        logger._log_buffer = []
                    logger._log_buffer.append((level, message))
                    
                    # Flush buffer jika sudah menunggu cukup lama
                    if len(logger._log_buffer) > 5 or time_diff > 0.5:
                        _flush_log_buffer(logger, original_log, ui_components)
                    
                    return
            
            # Call asli untuk konsistensi dan file logging
            original_log(level, message)
            
            # Log ke UI
            _display_ui_log(level, message, ui_components)
            
            # Update last log time
            logger._last_log_time = current_time
        
        # Ganti implementasi log dengan versi kustom
        logger.log = ui_log
        
        # Tambahkan fungsi flush
        def flush_buffer():
            if hasattr(logger, '_log_buffer') and logger._log_buffer:
                _flush_log_buffer(logger, original_log, ui_components)
        
        logger.flush_buffer = flush_buffer
        
        return logger
    
    except ImportError:
        # Fallback ke UI Logger sederhana dengan logging standar
        class SimpleUILogger:
            def __init__(self, name, ui_components):
                self.name = name
                self.ui_components = ui_components
                self.logger = logging.getLogger(name)
                self._last_log_time = 0
                self._log_buffer = []
                self._min_log_interval = 0.2
                
            def _log_ui(self, message, level):
                # Batasi frekuensi log
                current_time = time.time()
                time_diff = current_time - self._last_log_time
                
                if time_diff < self._min_log_interval:
                    # Tambahkan ke buffer jika terlalu sering
                    self._log_buffer.append((level, message))
                    
                    # Flush buffer jika sudah menunggu cukup lama atau buffer penuh
                    if len(self._log_buffer) > 5 or time_diff > 0.5:
                        self._flush_buffer()
                    
                    return
                
                # Log ke UI component
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
                
                # Update timestamp
                self._last_log_time = current_time
            
            def _flush_buffer(self):
                if not self._log_buffer:
                    return
                    
                # Group similar logs
                grouped_messages = {}
                for level, msg in self._log_buffer:
                    if msg in grouped_messages:
                        grouped_messages[msg] = (level, grouped_messages[msg][1] + 1)
                    else:
                        grouped_messages[msg] = (level, 1)
                
                # Log grouped messages
                for msg, (level, count) in grouped_messages.items():
                    if count > 1:
                        self._log_ui(f"{msg} (x{count})", level)
                    else:
                        self._log_ui(msg, level)
                
                # Clear buffer
                self._log_buffer = []
            
            def flush_buffer(self):
                self._flush_buffer()
            
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

def _flush_log_buffer(logger, original_log, ui_components):
    """Flush buffer log dengan grouping untuk mengurangi spam."""
    if not hasattr(logger, '_log_buffer') or not logger._log_buffer:
        return
        
    # Group similar logs
    grouped_messages = {}
    for level, msg in logger._log_buffer:
        if msg in grouped_messages:
            grouped_messages[msg] = (level, grouped_messages[msg][1] + 1)
        else:
            grouped_messages[msg] = (level, 1)
    
    # Log grouped messages
    for msg, (level, count) in grouped_messages.items():
        if count > 1:
            # Log dengan counter untuk pesan yang duplikat
            counter_msg = f"{msg} (x{count})"
            original_log(level, counter_msg)
            _display_ui_log(level, counter_msg, ui_components)
        else:
            original_log(level, msg)
            _display_ui_log(level, msg, ui_components)
    
    # Clear buffer
    logger._log_buffer = []
    logger._last_log_time = time.time()

def _display_ui_log(level, message, ui_components):
    """Display log message to UI component."""
    with ui_components['status']:
        from smartcash.ui.utils.alert_utils import create_status_indicator
        # Map LogLevel ke status type
        level_name = level.name.lower() if hasattr(level, 'name') else 'info'
        status_type = "success" if level_name == "success" else \
                     "warning" if level_name in ("warning", "warn") else \
                     "error" if level_name in ("error", "critical") else "info"
        display(create_status_indicator(status_type, message))

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
    Metode yang lebih clean dengan satu implementasi dan buffer untuk mengurangi spam.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
    """
    # Pastikan ada status output widget
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        return
    
    # Pastikan tidak terjadi multiple intercepts
    if 'custom_stdout' in ui_components and ui_components.get('custom_stdout') == sys.stdout:
        return
        
    # Hapus handler logging lain untuk mencegah duplikasi output
    try:
        import logging
        root_logger = logging.getLogger()
        
        # Hapus handler stdout untuk mencegah duplikasi
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.__stdout__:
                root_logger.removeHandler(handler)
    except Exception:
        pass
    
    # Buat stdout interceptor dengan thread-safety dan buffer
    class UIStdoutInterceptor:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.terminal = sys.stdout
            self.buffer = ""
            self.lock = threading.RLock()
            self.buffer_limit = 1000  # Batasi buffer untuk mencegah memory leak
            self.last_flush_time = time.time()
            self.min_flush_interval = 0.5  # Minimal 500ms antara flush
            self.buffered_lines = 0
            
        def write(self, message):
            # Write ke terminal asli
            self.terminal.write(message)
            
            # Buffer output sampai ada newline, dengan thread-safety
            with self.lock:
                # Batasi ukuran buffer
                if len(self.buffer) > self.buffer_limit:
                    self.buffer = self.buffer[-self.buffer_limit:]
                
                self.buffer += message
                
                # Deteksi baris baru untuk buffer flush
                new_lines = message.count('\n')
                self.buffered_lines += new_lines
                
                # Kondisi untuk flush: 
                # 1. Ada newline dan sudah cukup waktu sejak flush terakhir
                # 2. Sudah terlalu banyak baris di buffer
                current_time = time.time()
                time_since_flush = current_time - self.last_flush_time
                
                should_flush = False
                if '\n' in self.buffer and time_since_flush >= self.min_flush_interval:
                    should_flush = True
                elif self.buffered_lines >= 5:  # Flush setelah 5 baris
                    should_flush = True
                
                if should_flush:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]  # Simpan baris terakhir yang belum lengkap
                    
                    # Reset counter
                    self.buffered_lines = 0
                    self.last_flush_time = current_time
                    
                    # Group similar lines untuk mengurangi spam
                    grouped_lines = {}
                    for line in lines[:-1]:
                        if line.strip():  # Cek jika bukan baris kosong
                            if line in grouped_lines:
                                grouped_lines[line] += 1
                            else:
                                grouped_lines[line] = 1
                    
                    # Tampilkan setiap grup baris
                    if grouped_lines:
                        try:
                            with self.ui_components['status']:
                                for line, count in grouped_lines.items():
                                    try:
                                        from smartcash.ui.utils.alert_utils import create_status_indicator
                                        display_line = f"{line} (x{count})" if count > 1 else line
                                        display(create_status_indicator("info", display_line))
                                    except ImportError:
                                        display(HTML(f"<div>{line}</div>"))
                        except Exception:
                            # Jika ada error saat menampilkan ke UI, kirim ke stdout asli
                            for line, count in grouped_lines.items():
                                self.terminal.write(f"[{count}x] {line}\n" if count > 1 else f"{line}\n")
        
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
                        self.terminal.write(f"{self.buffer}\n")
                    self.buffer = ""
                    self.buffered_lines = 0
                    self.last_flush_time = time.time()
        
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