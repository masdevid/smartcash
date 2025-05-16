"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Utility untuk mengarahkan output logger ke UI widget dengan integrasi standar dan tanpa duplikasi
"""

import logging
import sys
import threading
import time
import os
import io
import traceback
from typing import Dict, Any, Callable, Optional, List, Union, Set, Tuple
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Buffer global untuk menyimpan log awal sebelum UI terender
_EARLY_LOG_BUFFER = []
_BUFFER_LOCK = threading.RLock()
_UI_READY = False
_INTERCEPT_ACTIVE = False
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_ORIGINAL_LOGGERS = {}  # Untuk menyimpan handler asli dari logger

# Untuk menyimpan referensi ke semua interceptor yang aktif
_ACTIVE_INTERCEPTORS = set()

# Konfigurasi global
_CONFIG = {
    'buffer_limit': 5000,  # Batasi buffer untuk mencegah memory leak
    'log_to_file': True,   # Apakah log juga disimpan ke file
    'log_file': 'logs/smartcash.log',  # Path file log
    'log_level': logging.INFO,  # Level log default
    'ui_log_level': logging.INFO,  # Level log untuk UI
    'auto_intercept': True,  # Otomatis intercept saat UI ready
}

# Prefiks untuk mengidentifikasi pesan yang tidak perlu ditampilkan di UI
_IGNORE_PREFIXES = [
    '[DEBUG]', 'DEBUG:', 'INFO:', '[INFO]',
    'Using TensorFlow backend', 'Colab notebook',
    'Your session crashed', 'Executing in eager mode',
    'TensorFlow', 'NumExpr', 'Running on',
    '/usr/local/lib', 'WARNING:', '[WARNING]',
    'This TensorFlow binary', 'For more info',
    'Welcome to', 'This notebook', 'The Jupyter',
    'IPython', 'torch._C', 'matplotlib', 'numpy',
    'Requirement already satisfied', 'Downloading',
    'Collecting', 'Building wheel', 'Installing',
    'Successfully installed', 'Preparing metadata',
    'Extracting', 'Processing', 'Looking in',
    'Copying', 'Cloning', 'Pulling', 'Pushing',
    'Fetching', 'Checking out', 'HEAD is now at',
    'Already up to date', 'Your branch is',
    'Mounted at', 'Drive already mounted',
    'FutureWarning', 'DeprecationWarning',
    'UserWarning', 'RuntimeWarning',
    # Filter tambahan untuk mengurangi log INFO dari config_sync
    'INFO:config_sync', 'INFO:root', 'INFO:smartcash.ui.setup',
    'Environment config handlers', 'berhasil diinisialisasi',
    # Filter tambahan untuk meredam log sinkronisasi drive config
    'Menyinkronkan konfigurasi', 'Konfigurasi berhasil disinkronkan',
    'Memuat konfigurasi dari Drive', 'Sinkronisasi konfigurasi',
    'config_sync:', 'drive_sync:'
]

def set_ui_ready(ready: bool = True) -> None:
    """
    Set status UI ready secara manual.
    
    Args:
        ready: Status UI ready
    """
    global _UI_READY
    with _BUFFER_LOCK:
        _UI_READY = ready
        
def is_ui_ready() -> bool:
    """
    Cek apakah UI sudah siap.
    
    Returns:
        Status UI ready
    """
    global _UI_READY
    with _BUFFER_LOCK:
        return _UI_READY

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
    from smartcash.common.logger import get_logger, SmartCashLogger, LogLevel
    
    # Sebelum membuat logger kustom, pastikan ada widget output
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        # Fallback ke logger standar tanpa UI
        return get_logger(name)
    
    # Set UI ready karena sudah ada widget output
    set_ui_ready(True)
    
    # Setup LogLevel ke INFO untuk mengurangi debug log
    logger = get_logger(name, LogLevel.INFO)
    
    # Simpan referensi ke fungsi log asli
    original_log = logger.log
    
    # Dapatkan referensi ke logger Python standar
    python_logger = None
    if hasattr(logger, 'logger'):
        python_logger = logger.logger
        
        # Hapus semua console handlers dari logger
        for handler in list(python_logger.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                python_logger.removeHandler(handler)
    
    def ui_log(level, message):
        # Jangan log debug message ke UI untuk mengurangi noise
        level_name = level.name.lower() if hasattr(level, 'name') else 'info'
        
        # Konversi ke level logging standar
        std_level = logger.LEVEL_MAPPING[level] if hasattr(logger, 'LEVEL_MAPPING') else logging.INFO
        
        # Format pesan untuk file
        if hasattr(logger, '_format_message'):
            _, file_msg = logger._format_message(level, message)
        else:
            file_msg = message
            
        # Log ke file melalui Python logger (tanpa console output karena handlers sudah dihapus)
        if python_logger:
            python_logger.log(std_level, file_msg)
        
        # Skip UI display untuk debug messages
        if level_name == 'debug':
            return
            
        # Log ke UI untuk non-debug messages
        with ui_components['status']:
            from smartcash.ui.utils.alert_utils import create_status_indicator
            # Map LogLevel ke status type
            status_type = "success" if level_name == "success" else \
                         "warning" if level_name in ("warning", "warn") else \
                         "error" if level_name in ("error", "critical") else "info"
            display(create_status_indicator(status_type, message))
    
    # Ganti implementasi log dengan versi kustom
    logger.log = ui_log
    return logger


def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", emoji: str = "") -> None:
    """
    Log pesan langsung ke UI widget menggunakan komponen terpadu.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, debug, success)
        emoji: Emoji untuk ditampilkan (akan ditambahkan ke pesan)
    """
    global _EARLY_LOG_BUFFER
    
    # Skip debug messages untuk UI
    if level == "debug":
        return
    
    # Format pesan dengan emoji jika ada
    formatted_message = f"{emoji} {message}" if emoji else message
    
    # Cek apakah UI sudah siap
    if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
        set_ui_ready(True)  # UI sudah siap
        with ui_components['status']:
            try:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator(level, formatted_message))
            except ImportError:
                display(HTML(f"<div>{formatted_message}</div>"))
    else:
        # Simpan ke buffer jika UI belum siap
        with _BUFFER_LOCK:
            if not is_ui_ready():
                _EARLY_LOG_BUFFER.append((level, formatted_message))


def flush_early_logs(ui_components: Dict[str, Any], test_mode: bool = False) -> None:
    """
    Flush log awal yang tersimpan di buffer ke UI setelah UI terender.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
        test_mode: Flag untuk mode testing, jika True akan mengosongkan buffer tanpa menampilkan
    """
    global _EARLY_LOG_BUFFER
    
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        return
    
    with _BUFFER_LOCK:
        if not _EARLY_LOG_BUFFER:
            return
            
        # Simpan salinan buffer untuk ditampilkan
        buffer_copy = _EARLY_LOG_BUFFER.copy()
        
        # Kosongkan buffer terlebih dahulu untuk menghindari kehilangan log jika terjadi error
        _EARLY_LOG_BUFFER = []
        
        # Jika mode test, hanya kosongkan buffer dan set UI ready
        if test_mode:
            set_ui_ready(True)
            return
        
        try:
            with ui_components['status']:
                try:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    for level, message in buffer_copy:
                        display(create_status_indicator(level, message))
                except ImportError:
                    for _, message in buffer_copy:
                        display(HTML(f"<div>{message}</div>"))
            # Set UI ready setelah berhasil menampilkan
            set_ui_ready(True)
        except Exception as e:
            print(f"[ERROR] Gagal flush early logs: {str(e)}")
            # Jika gagal, kembalikan log ke buffer
            with _BUFFER_LOCK:
                _EARLY_LOG_BUFFER = buffer_copy + _EARLY_LOG_BUFFER


class UIStdoutInterceptor:
    """Kelas untuk mengalihkan stdout ke UI widget."""
    
    def __init__(self, ui_components, stream_type='stdout'):
        self.ui_components = ui_components
        self.stream_type = stream_type
        self.terminal = _ORIGINAL_STDOUT if stream_type == 'stdout' else _ORIGINAL_STDERR
        self.buffer = ""
        self.lock = threading.RLock()
        self.buffer_limit = _CONFIG['buffer_limit']
        self.ignore_debug_prefix = _IGNORE_PREFIXES
        
    def write(self, message):
        global _EARLY_LOG_BUFFER
        
        # Write ke terminal asli
        self.terminal.write(message)
        
        # Skip pesan yang tidak perlu ditampilkan di UI
        msg_strip = message.strip()
        if not msg_strip or len(msg_strip) < 2:  # Skip pesan kosong atau terlalu pendek
            return
            
        # Filter berdasarkan prefiks yang umum untuk log yang tidak perlu
        if any(prefix in msg_strip for prefix in self.ignore_debug_prefix):
            return
            
        # Filter tambahan untuk pesan pertama kali cell dijalankan
        if 'Executing:' in msg_strip or 'In [' in msg_strip:
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
                
                # Tampilkan setiap baris lengkap
                for line in lines[:-1]:
                    if line.strip():  # Cek jika bukan baris kosong
                        try:
                            # Cek apakah UI sudah siap
                            if 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                                with self.ui_components['status']:
                                    try:
                                        from smartcash.ui.utils.alert_utils import create_status_indicator
                                        level = "error" if self.stream_type == 'stderr' else "info"
                                        display(create_status_indicator(level, line))
                                        set_ui_ready(True)  # UI sudah siap
                                    except ImportError:
                                        display(HTML(f"<div>{line}</div>"))
                            else:
                                # Simpan ke buffer jika UI belum siap
                                with _BUFFER_LOCK:
                                    if not is_ui_ready():
                                        level = "error" if self.stream_type == 'stderr' else "info"
                                        _EARLY_LOG_BUFFER.append((level, line))
                        except Exception:
                            # Jika ada error saat menampilkan ke UI, kirim ke stdout asli
                            self.terminal.write(f"[UI {self.stream_type.upper()} ERROR] {line}\n")
                            # Simpan ke buffer untuk ditampilkan nanti
                            with _BUFFER_LOCK:
                                if not is_ui_ready():
                                    level = "error" if self.stream_type == 'stderr' else "info"
                                    _EARLY_LOG_BUFFER.append((level, line))
    
    def flush(self):
        self.terminal.flush()
        # Flush buffer jika perlu, dengan thread-safety
        with self.lock:
            if self.buffer:
                try:
                    with self.ui_components['status']:
                        try:
                            from smartcash.ui.utils.alert_utils import create_status_indicator
                            level = "error" if self.stream_type == 'stderr' else "info"
                            display(create_status_indicator(level, self.buffer))
                        except ImportError:
                            display(HTML(f"<div>{self.buffer}</div>"))
                except Exception:
                    # Fallback ke stdout asli
                    self.terminal.write(f"[UI {self.stream_type.upper()} ERROR] {self.buffer}\n")
                self.buffer = ""
    
    # Kebutuhan IOBase lainnya
    def isatty(self):
        return self.terminal.isatty()
        
    def fileno(self):
        return self.terminal.fileno()


class UILoggingHandler(logging.Handler):
    """Handler logging yang mengarahkan output ke UI widget."""
    
    def __init__(self, ui_components):
        super().__init__()
        self.ui_components = ui_components
        self.setLevel(_CONFIG['ui_log_level'])
        self.ignore_debug_prefix = _IGNORE_PREFIXES
        
    def emit(self, record):
        global _EARLY_LOG_BUFFER
        
        try:
            # Format record
            msg = self.format(record)
            
            # Skip pesan yang tidak perlu ditampilkan di UI
            if any(prefix in msg for prefix in self.ignore_debug_prefix):
                return
                
            # Tentukan level UI berdasarkan level logging
            if record.levelno >= logging.ERROR:
                level = "error"
            elif record.levelno >= logging.WARNING:
                level = "warning"
            else:
                level = "info"
                
            # Cek apakah UI sudah siap
            if 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                with self.ui_components['status']:
                    try:
                        from smartcash.ui.utils.alert_utils import create_status_indicator
                        display(create_status_indicator(level, msg))
                        set_ui_ready(True)  # UI sudah siap
                    except ImportError:
                        display(HTML(f"<div>{msg}</div>"))
            else:
                # Simpan ke buffer jika UI belum siap
                with _BUFFER_LOCK:
                    if not is_ui_ready():
                        _EARLY_LOG_BUFFER.append((level, msg))
        except Exception:
            # Jika ada error, log ke stderr asli
            traceback.print_exc()


def setup_file_logging():
    """Setup logging ke file."""
    if not _CONFIG['log_to_file']:
        return None
        
    try:
        # Buat direktori log jika belum ada
        log_dir = os.path.dirname(_CONFIG['log_file'])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Buat file handler
        file_handler = logging.FileHandler(_CONFIG['log_file'])
        file_handler.setLevel(_CONFIG['log_level'])
        
        # Format log file
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        return file_handler
    except Exception:
        traceback.print_exc()
        return None


def intercept_logging(ui_components: Dict[str, Any]):
    """
    Intercept logging module dan arahkan ke UI widget.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
    """
    global _ORIGINAL_LOGGERS, _ACTIVE_INTERCEPTORS
    
    # Pastikan ada status output widget
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        return
    
    # Setup file logging
    file_handler = setup_file_logging()
    
    # Buat UI handler
    ui_handler = UILoggingHandler(ui_components)
    
    # Intercept root logger
    root_logger = logging.getLogger()
    
    # Simpan handler asli
    if root_logger not in _ORIGINAL_LOGGERS:
        _ORIGINAL_LOGGERS[root_logger] = list(root_logger.handlers)
    
    # Hapus semua handler
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    
    # Tambahkan UI handler
    root_logger.addHandler(ui_handler)
    
    # Tambahkan file handler jika ada
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Simpan referensi ke UI handler
    _ACTIVE_INTERCEPTORS.add(ui_handler)


def restore_logging():
    """Kembalikan logging ke keadaan semula."""
    global _ORIGINAL_LOGGERS, _ACTIVE_INTERCEPTORS
    
    # Kembalikan handler asli
    for logger, handlers in _ORIGINAL_LOGGERS.items():
        # Hapus semua handler
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        
        # Tambahkan handler asli
        for handler in handlers:
            logger.addHandler(handler)
    
    # Reset
    _ORIGINAL_LOGGERS = {}
    _ACTIVE_INTERCEPTORS = set()


def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """
    Intercept stdout dan stderr dan arahkan ke UI widget.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
    """
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _INTERCEPT_ACTIVE, _ACTIVE_INTERCEPTORS
    
    # Flush log awal yang tersimpan di buffer
    flush_early_logs(ui_components)
    
    # Pastikan ada status output widget
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        return
    
    # Pastikan tidak terjadi multiple intercepts
    if _INTERCEPT_ACTIVE:
        return
    
    # Intercept logging
    intercept_logging(ui_components)
    
    # Simpan stdout dan stderr asli
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    
    # Pasang interceptor untuk stdout
    stdout_interceptor = UIStdoutInterceptor(ui_components, 'stdout')
    sys.stdout = stdout_interceptor
    ui_components['custom_stdout'] = stdout_interceptor
    _ACTIVE_INTERCEPTORS.add(stdout_interceptor)
    
    # Pasang interceptor untuk stderr
    stderr_interceptor = UIStdoutInterceptor(ui_components, 'stderr')
    sys.stderr = stderr_interceptor
    ui_components['custom_stderr'] = stderr_interceptor
    _ACTIVE_INTERCEPTORS.add(stderr_interceptor)
    
    # Set flag intercept aktif
    _INTERCEPT_ACTIVE = True
    
    # Log informasi
    print("📝 UI logger berhasil diaktifkan dan mengalihkan semua log ke UI")


def restore_stdout():
    """Kembalikan stdout dan stderr ke keadaan semula."""
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _INTERCEPT_ACTIVE, _ACTIVE_INTERCEPTORS
    
    # Kembalikan stdout dan stderr
    if _INTERCEPT_ACTIVE:
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
        _INTERCEPT_ACTIVE = False
        
        # Kembalikan logging
        restore_logging()
        
        # Reset interceptors
        _ACTIVE_INTERCEPTORS = set()
        
        # Log informasi
        print("📝 UI logger dinonaktifkan, log dikembalikan ke console")


def configure_ui_logger(config: Dict[str, Any]) -> None:
    """
    Konfigurasi UI logger.
    
    Args:
        config: Dictionary berisi konfigurasi
    """
    global _CONFIG
    
    # Update konfigurasi
    for key, value in config.items():
        if key in _CONFIG:
            _CONFIG[key] = value


def auto_intercept_when_ready(ui_components: Dict[str, Any]) -> None:
    """
    Otomatis intercept stdout dan stderr saat UI sudah siap.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
    """
    if not _CONFIG['auto_intercept']:
        return
        
    # Cek apakah UI sudah siap
    if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
        # Intercept stdout dan stderr
        intercept_stdout_to_ui(ui_components)
        return True
    
    return False