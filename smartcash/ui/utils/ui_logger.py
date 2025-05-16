"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Utility untuk mengarahkan output logger ke UI widget dengan integrasi standar dan tanpa duplikasi
"""

import logging
import sys
import threading
import time
from typing import Dict, Any, Callable, Optional, List, Union
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Buffer global untuk menyimpan log awal sebelum UI terender
# Format: [(level, message, timestamp), ...]
_EARLY_LOG_BUFFER = []
_BUFFER_LOCK = threading.RLock()
_UI_READY = False
_LOGS_DISPLAYED = set()  # Set untuk melacak log yang sudah ditampilkan di UI

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
    global _EARLY_LOG_BUFFER, _UI_READY, _LOGS_DISPLAYED
    
    # Skip debug messages untuk UI
    if level == "debug":
        return
    
    # Format pesan dengan emoji jika ada
    formatted_message = f"{emoji} {message}" if emoji else message
    
    # Buat timestamp unik untuk log ini
    timestamp = time.time()
    log_id = hash(f"{formatted_message}_{timestamp}")
    
    # Selalu tampilkan di console terlebih dahulu
    print(f"[{level.upper()}] {formatted_message}")
    
    # Cek apakah UI sudah siap
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        # Simpan ke buffer jika UI belum siap
        with _BUFFER_LOCK:
            _EARLY_LOG_BUFFER.append((level, formatted_message, log_id))
        return
    
    # Coba tampilkan ke UI
    try:
        # Tandai log ini sebagai sudah ditampilkan
        _LOGS_DISPLAYED.add(log_id)
        
        with ui_components['status']:
            try:
                # Gunakan komponen alert_utils jika tersedia
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator(level, formatted_message))
                _UI_READY = True  # UI sudah siap
            except ImportError:
                # Fallback sederhana jika alert_utils tidak tersedia
                style_map = {
                    "info": "color: #0c5460; background-color: #d1ecf1; padding: 10px; border-radius: 4px;",
                    "success": "color: #155724; background-color: #d4edda; padding: 10px; border-radius: 4px;",
                    "warning": "color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 4px;",
                    "error": "color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 4px;"
                }
                style = style_map.get(level, style_map["info"])
                display(HTML(f"<div style=\"{style}\">{formatted_message}</div>"))
    except Exception as e:
        # Jika gagal menampilkan ke UI, simpan ke buffer
        with _BUFFER_LOCK:
            if log_id not in _LOGS_DISPLAYED:
                _EARLY_LOG_BUFFER.append((level, formatted_message, log_id))
        # Tambahkan info error
        print(f"[ERROR] Gagal menampilkan log ke UI: {str(e)}")

def flush_early_logs(ui_components: Dict[str, Any]) -> None:
    """
    Flush log awal yang tersimpan di buffer ke UI setelah UI terender.
    Hanya menampilkan log yang belum ditampilkan di UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
    """
    global _EARLY_LOG_BUFFER, _UI_READY, _LOGS_DISPLAYED
    
    if 'status' not in ui_components or not hasattr(ui_components['status'], 'clear_output'):
        return
    
    with _BUFFER_LOCK:
        if not _EARLY_LOG_BUFFER:
            return
            
        # Set UI ready
        _UI_READY = True
        
        # Filter log yang belum ditampilkan
        logs_to_display = []
        for level, message, log_id in _EARLY_LOG_BUFFER:
            if log_id not in _LOGS_DISPLAYED:
                logs_to_display.append((level, message, log_id))
                _LOGS_DISPLAYED.add(log_id)
        
        # Kosongkan buffer
        _EARLY_LOG_BUFFER = []
        
        # Jika tidak ada log yang perlu ditampilkan, return
        if not logs_to_display:
            return
        
        try:
            with ui_components['status']:
                try:
                    from smartcash.ui.utils.alert_utils import create_status_indicator
                    for level, message, _ in logs_to_display:
                        display(create_status_indicator(level, message))
                except ImportError:
                    for _, message, _ in logs_to_display:
                        display(HTML(f"<div>{message}</div>"))
        except Exception as e:
            print(f"[ERROR] Gagal flush early logs: {str(e)}")

def intercept_stdout_to_ui(ui_components: Dict[str, Any]) -> None:
    """
    Intercept stdout dan arahkan ke UI widget.
    Metode yang lebih clean dengan satu implementasi dan thread lock.
    
    Args:
        ui_components: Dictionary berisi komponen UI dengan kunci 'status'
    """
    # Flush log awal yang tersimpan di buffer
    flush_early_logs(ui_components)
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
        
        # Hapus semua stream handlers untuk mencegah output ke console
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)
                
        # Tambahkan file handler jika diperlukan untuk tetap menyimpan log
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        if not has_file_handler:
            try:
                from pathlib import Path
                log_dir = Path('logs')
                log_dir.mkdir(exist_ok=True)
                file_handler = logging.FileHandler(log_dir / 'colab.log')
                file_handler.setLevel(logging.INFO)
                root_logger.addHandler(file_handler)
            except Exception:
                pass
    except Exception:
        pass
    
    # Buat stdout interceptor dengan thread-safety
    class UIStdoutInterceptor:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.terminal = sys.stdout
            self.buffer = ""
            self.lock = threading.RLock()
            self.buffer_limit = 1000  # Batasi buffer untuk mencegah memory leak
            # Tambahkan lebih banyak prefiks untuk difilter
            self.ignore_debug_prefix = [
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
            ]  # Prefiks untuk mengidentifikasi pesan yang tidak perlu ditampilkan di UI
            
        def write(self, message):
            global _EARLY_LOG_BUFFER, _UI_READY, _LOGS_DISPLAYED
            
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
                            # Buat timestamp unik untuk log ini
                            timestamp = time.time()
                            log_id = hash(f"{line}_{timestamp}")
                            
                            try:
                                # Cek apakah UI sudah siap
                                if 'status' in self.ui_components and hasattr(self.ui_components['status'], 'clear_output'):
                                    # Tandai log ini sebagai sudah ditampilkan
                                    _LOGS_DISPLAYED.add(log_id)
                                    
                                    with self.ui_components['status']:
                                        try:
                                            from smartcash.ui.utils.alert_utils import create_status_indicator
                                            display(create_status_indicator("info", line))
                                            _UI_READY = True  # UI sudah siap
                                        except ImportError:
                                            display(HTML(f"<div>{line}</div>"))
                                else:
                                    # Simpan ke buffer jika UI belum siap
                                    with _BUFFER_LOCK:
                                        _EARLY_LOG_BUFFER.append(("info", line, log_id))
                            except Exception as e:
                                # Jika ada error saat menampilkan ke UI, kirim ke stdout asli
                                self.terminal.write(f"[UI STDOUT ERROR] {line} ({str(e)})\n")
        
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