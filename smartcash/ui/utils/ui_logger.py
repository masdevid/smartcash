"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Logger khusus untuk UI yang dapat mengarahkan output ke widget UI dan file
"""

import logging
import sys
import threading
import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List, Union
from IPython.display import display, HTML
import ipywidgets as widgets
from datetime import datetime

# Ekspor fungsi-fungsi utama
__all__ = [
    'UILogger', 
    'create_ui_logger', 
    'get_current_ui_logger',
    'log_to_ui',
    'intercept_stdout_to_ui',
    'restore_stdout'
]

class UILogger:
    """
    Logger khusus untuk UI yang dapat mengarahkan output ke widget UI dan file.
    """
    
    def __init__(self, 
                ui_components: Dict[str, Any], 
                name: str = "ui_logger",
                log_to_file: bool = False,
                log_dir: str = "logs",
                log_level: int = logging.INFO):
        """
        Inisialisasi UILogger.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            name: Nama logger
            log_to_file: Flag untuk mengaktifkan logging ke file
            log_dir: Direktori untuk menyimpan file log
            log_level: Level logging (default: INFO)
        """
        self.ui_components = ui_components
        self.name = name
        self.log_level = log_level
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Hapus handler yang sudah ada untuk mencegah duplikasi
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Tambahkan console handler untuk output standar
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            '%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Tambahkan file handler jika diperlukan
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            
            log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Simpan path log file untuk referensi
            self.log_file_path = log_file
        else:
            self.log_file_path = None
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """
        Log pesan ke UI components.
        
        Args:
            message: Pesan yang akan di-log
            level: Level log (info, warning, error, success)
        """
        # Skip pesan kosong
        if not message or not message.strip():
            return
            
        # Tambahkan timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Tambahkan emoji sesuai level
        emoji_map = {
            "debug": "🔍",
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🔥"
        }
        emoji = emoji_map.get(level, "ℹ️")
        
        # Format pesan dengan timestamp dan emoji
        formatted_message = f"[{timestamp}] {emoji} {message}"
        
        # Prioritas log output:
        # 1. log_output widget jika ada
        # 2. status widget jika ada
        # 3. fallback ke stdout
        
        if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
            with self.ui_components['log_output']:
                print(formatted_message)
        elif 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
            with self.ui_components['status']:
                try:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    display(create_status_indicator(level, message))
                except ImportError:
                    # Fallback jika tidak ada alert_utils
                    color = {
                        "debug": "gray",
                        "info": "blue",
                        "success": "green",
                        "warning": "orange",
                        "error": "red",
                        "critical": "darkred"
                    }.get(level, "black")
                    display(HTML(f"<div style='color:{color}'>{formatted_message}</div>"))
        else:
            # Fallback ke stdout jika tidak ada UI components
            print(formatted_message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        if not message or not message.strip():
            return
        self.logger.debug(message)
        # Debug messages hanya ditampilkan di UI jika level log DEBUG
        if self.log_level <= logging.DEBUG:
            self._log_to_ui(message, "debug")
    
    def info(self, message: str) -> None:
        """Log info message."""
        if not message or not message.strip():
            return
        self.logger.info(message)
        self._log_to_ui(message, "info")
    
    def success(self, message: str) -> None:
        """Log success message."""
        if not message or not message.strip():
            return
        self.logger.info(f"SUCCESS: {message}")
        self._log_to_ui(message, "success")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        if not message or not message.strip():
            return
        self.logger.warning(message)
        self._log_to_ui(message, "warning")
    
    def error(self, message: str) -> None:
        """Log error message."""
        if not message or not message.strip():
            return
        self.logger.error(message)
        self._log_to_ui(message, "error")
    
    def critical(self, message: str) -> None:
        """Log critical message."""
        if not message or not message.strip():
            return
        self.logger.critical(message)
        self._log_to_ui(message, "critical")
    
    def progress(self, iterable=None, desc="Processing", **kwargs):
        """
        Buat progress bar dan log progress.
        
        Args:
            iterable: Iterable untuk diiterasi
            desc: Deskripsi progress
            **kwargs: Arguments tambahan untuk tqdm
            
        Returns:
            tqdm progress bar atau iterable asli jika tqdm tidak ada
        """
        try:
            from tqdm.auto import tqdm
            progress_bar = tqdm(iterable, desc=desc, **kwargs)
            return progress_bar
        except ImportError:
            self.warning("tqdm tidak ditemukan, progress tracking tidak aktif")
            return iterable

    def set_level(self, level: int) -> None:
        """
        Atur level logging untuk logger dan semua handlers.
        
        Args:
            level: Level logging untuk diatur
        """
        self.log_level = level
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
            
        # Log informasi perubahan level
        level_names = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL"
        }
        level_name = level_names.get(level, str(level))
        self.info(f"Log level diubah ke {level_name}")


def create_ui_logger(ui_components: Dict[str, Any], 
                    name: str = "ui_logger",
                    log_to_file: bool = False,
                    redirect_stdout: bool = True,
                    log_dir: str = "logs",
                    log_level: int = logging.INFO) -> UILogger:
    """
    Buat UILogger dan setup integrasi dengan UI components.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        name: Nama logger
        log_to_file: Flag untuk mengaktifkan logging ke file
        redirect_stdout: Flag untuk mengalihkan stdout ke UI
        log_dir: Direktori untuk menyimpan file log
        log_level: Level logging (default: INFO)
        
    Returns:
        UILogger instance
    """
    # Buat logger
    logger = UILogger(ui_components, name, log_to_file, log_dir, log_level)
    
    # Redirect stdout ke UI jika diperlukan
    if redirect_stdout and 'status' in ui_components:
        intercept_stdout_to_ui(ui_components)
    
    # Reset logging untuk menghindari duplikasi
    _reset_logging_handlers()
    
    # Integrasikan logger dengan SmartCashLogger jika ada
    try:
        from smartcash.common.logger import get_logger, LogLevel
        
        # Mapping level log antara UILogger dan SmartCashLogger
        level_mapping = {
            logging.DEBUG: LogLevel.DEBUG,
            logging.INFO: LogLevel.INFO,
            logging.WARNING: LogLevel.WARNING,
            logging.ERROR: LogLevel.ERROR,
            logging.CRITICAL: LogLevel.CRITICAL
        }
        
        # Gunakan level yang sesuai dengan UILogger
        sc_level = level_mapping.get(log_level, LogLevel.INFO)
        std_logger = get_logger(name, level=sc_level)
        
        # Callback untuk meneruskan log dari SmartCashLogger ke UI
        def ui_log_callback(level, message):
            if not message or not message.strip():
                return
            if level.name == 'DEBUG':
                logger.debug(message)
            elif level.name == 'INFO':
                logger.info(message)
            elif level.name == 'SUCCESS':
                logger.success(message)
            elif level.name == 'WARNING':
                logger.warning(message)
            elif level.name in ('ERROR', 'CRITICAL'):
                logger.error(message)
        
        # Tambahkan callback ke std_logger
        std_logger.add_callback(ui_log_callback)
        
        # Simpan referensi ke smartcash_logger di ui_components
        ui_components['smartcash_logger'] = std_logger
        
    except ImportError:
        # Tidak ada SmartCashLogger, hanya gunakan UILogger
        pass
    
    # Simpan referensi ke logger di ui_components
    ui_components['logger'] = logger
    
    # Register logger untuk digunakan secara global
    _register_current_ui_logger(logger)
    
    return logger

# Variabel global untuk menyimpan referensi ke logger UI saat ini
_current_ui_logger = None

def _register_current_ui_logger(logger: UILogger) -> None:
    """
    Register logger UI saat ini untuk digunakan secara global.
    
    Args:
        logger: UILogger instance
    """
    global _current_ui_logger
    _current_ui_logger = logger

def get_current_ui_logger() -> Optional[UILogger]:
    """
    Dapatkan logger UI yang sedang aktif.
    
    Returns:
        UILogger instance atau None jika tidak ada
    """
    return _current_ui_logger

def _reset_logging_handlers():
    """Reset semua logging handlers untuk menghindari duplikasi."""
    root_logger = logging.getLogger()
    
    # Hapus semua StreamHandler untuk stdout/stderr
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            root_logger.removeHandler(handler)
            
    # Reset level logging untuk root logger
    root_logger.setLevel(logging.WARNING)
    
    # Tambahkan handler minimal untuk root logger
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """
    Intercept stdout dan arahkan ke UI widget.
    
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
            # Prefiks untuk messages yang akan difilter
            self.ignore_prefixes = [
                'DEBUG:', '[DEBUG]', 'INFO:', '[INFO]',
                'Using TensorFlow backend', 'Colab notebook',
                'Your session crashed', 'Executing in eager mode',
                'TensorFlow', 'NumExpr', 'Running on',
                '/usr/local/lib', 'WARNING:', '[WARNING]',
                'Config file not found in repo'  # Filter pesan error konfigurasi yang tidak ada
            ]
            
        def write(self, message):
            # Write ke terminal asli
            self.terminal.write(message)
            
            # Skip semua pesan yang tidak penting
            msg_strip = message.strip()
            if not msg_strip or len(msg_strip) < 2:  # Skip pesan kosong atau terlalu pendek
                return
                
            # Filter berdasarkan prefiks
            if any(prefix in msg_strip for prefix in self.ignore_prefixes):
                return
                
            # Filter messages seperti inisialisasi, setup, dll
            if ('inisialisasi' in msg_strip.lower() or 'setup' in msg_strip.lower() or 
                'handler' in msg_strip.lower() or 'initializing' in msg_strip.lower()):
                return
                
            # Buffer output sampai ada newline, dengan thread-safety
            with self.lock:
                # Batasi ukuran buffer
                if len(self.buffer) > self.buffer_limit:
                    self.buffer = self.buffer[-self.buffer_limit:]
                
                self.buffer += message
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]  # Simpan baris terakhir yang belum lengkap
                    
                    # Tampilkan setiap baris lengkap yang tidak kosong
                    for line in lines[:-1]:
                        if line.strip():  # Cek jika bukan baris kosong
                            try:
                                # Prioritaskan log_output jika ada
                                if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                                    with self.ui_components['log_output']:
                                        print(line)
                                else:
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
                if self.buffer and self.buffer.strip():  # Hanya flush jika buffer tidak kosong setelah strip
                    try:
                        # Prioritaskan log_output jika ada
                        if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
                            with self.ui_components['log_output']:
                                print(self.buffer)
                        else:
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


def restore_stdout(ui_components: Dict[str, Any]) -> None:
    """
    Kembalikan stdout ke aslinya.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    if 'original_stdout' in ui_components:
        # Simpan custom stdout untuk dibersihkan
        custom_stdout = ui_components.get('custom_stdout')
        
        # Kembalikan ke aslinya
        sys.stdout = ui_components['original_stdout']
        
        # Hapus referensi di ui_components
        ui_components.pop('original_stdout', None)
        ui_components.pop('custom_stdout', None)
        
        # Flush buffer stdout custom untuk memastikan tidak ada pesan yang tertinggal
        if custom_stdout and hasattr(custom_stdout, 'flush'):
            try:
                custom_stdout.flush()
            except:
                pass

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """
    Log pesan ke UI components dengan icon opsional.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        message: Pesan yang akan di-log
        level: Level log (info, warning, error, success)
        icon: Icon opsional untuk ditambahkan ke pesan
    """
    # Jika pesan kosong, jangan log
    if not message or not message.strip():
        return
        
    # Jika ada logger di ui_components, gunakan itu
    if 'logger' in ui_components:
        logger = ui_components['logger']
        
        # Format pesan dengan icon jika ada
        formatted_message = f"{icon} {message}" if icon else message
        
        # Log ke logger berdasarkan level
        if level == "error":
            logger.error(formatted_message)
        elif level == "warning":
            logger.warning(formatted_message)
        elif level == "success":
            logger.success(formatted_message)
        elif level == "critical":
            logger.critical(formatted_message)
        else:  # info
            logger.info(formatted_message)
        return
    
    # Fallback jika tidak ada logger: gunakan status atau log_output widget
    if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
        from IPython.display import display, HTML
        
        # Tentukan warna berdasarkan level
        color = {
            "info": "blue",
            "success": "green",
            "warning": "orange",
            "error": "red",
            "critical": "darkred"
        }.get(level, "black")
        
        # Format pesan dengan icon jika ada
        formatted_message = f"{icon} {message}" if icon else message
        
        # Tampilkan ke status widget
        with ui_components['status']:
            try:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator(level, formatted_message))
            except ImportError:
                display(HTML(f"<div style='color:{color}'>{formatted_message}</div>"))
    
    # Jika ada log_output, gunakan itu juga
    elif 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        from IPython.display import display
        from datetime import datetime
        
        # Format pesan dengan timestamp dan icon
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {icon + ' ' if icon else ''}{message}"
        
        # Tampilkan ke log_output widget
        with ui_components['log_output']:
            print(formatted_message)
    
    # Fallback ke stdout jika tidak ada UI components
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {icon + ' ' if icon else ''}{message}"
        print(formatted_message)