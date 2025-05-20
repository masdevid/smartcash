"""
File: smartcash/ui/utils/ui_logger.py
Deskripsi: Utility untuk mengarahkan output logger ke UI widget dengan integrasi standar dan tanpa duplikasi
"""

import logging
import sys
import threading
from typing import Dict, Any, Callable, Optional, List, Union
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

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
    
    # Dapatkan referensi ke logger Python standar tapi skip semua log non-critical
    python_logger = None
    if hasattr(logger, 'logger'):
        python_logger = logger.logger
        
        # Hapus semua handlers dari logger untuk menghilangkan semua log
        for handler in list(python_logger.handlers):
            python_logger.removeHandler(handler)
    
    def ui_log(level, message):
        # Jangan log debug message ke UI untuk mengurangi noise
        level_name = level.name.lower() if hasattr(level, 'name') else 'info'
        
        # Filter semua log inisialisasi dan non-critical
        if ('inisialisasi' in message.lower() or 'setup' in message.lower() or 
            'handler' in message.lower() or 'initializing' in message.lower() or 
            'module' in message.lower() or 'preprocessing' in message.lower()):
            return
            
        # Konversi ke level logging standar
        std_level = logger.LEVEL_MAPPING[level] if hasattr(logger, 'LEVEL_MAPPING') else logging.INFO
        
        # Format pesan untuk file
        if hasattr(logger, '_format_message'):
            _, file_msg = logger._format_message(level, message)
        else:
            file_msg = message
        
        # Skip UI display untuk debug messages dan info yang tidak penting
        if level_name == 'debug' or level_name == 'info':
            return
            
        # Hanya log error dan warning ke UI
        if level_name in ('error', 'critical', 'warning', 'warn'):
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                # Map LogLevel ke status type
                status_type = "warning" if level_name in ("warning", "warn") else "error"
                display(create_status_indicator(status_type, message))
    
    # Ganti implementasi log dengan versi kustom
    logger.log = ui_log
    return logger


def log_to_ui(ui_components: Optional[Dict[str, Any]], message: str, level: str = 'info', emoji: str = None) -> None:
    """
    Log pesan ke UI dengan level dan emoji yang sesuai.
    
    Args:
        ui_components: Dictionary komponen UI atau None
        message: Pesan yang akan ditampilkan
        level: Level log (info, warning, error, success)
        emoji: Emoji opsional untuk pesan
    """
    # Jika ui_components None, gunakan print biasa
    if ui_components is None:
        print(f"{emoji or ''} {message}")
        return
        
    # Pastikan ui_components adalah dictionary
    if not isinstance(ui_components, dict):
        print(f"{emoji or ''} {message}")
        return
    
    # Set emoji berdasarkan level jika tidak disediakan
    if emoji is None:
        emoji = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'success': '✅'
        }.get(level, '')
    
    # Coba log ke status output jika tersedia
    if 'status' in ui_components and hasattr(ui_components['status'], 'clear_output'):
        with ui_components['status']:
            display(create_status_indicator(level, f"{emoji} {message}"))
        return
    
    # Coba log ke log output jika tersedia
    if 'log_output' in ui_components:
        if hasattr(ui_components['log_output'], 'append_stdout'):
            if level == 'error':
                ui_components['log_output'].append_stderr(f"{emoji} {message}")
            else:
                ui_components['log_output'].append_stdout(f"{emoji} {message}")
            return
    
    # Fallback: print pesan
    print(f"{emoji} {message}")

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

            ]  # Prefiks untuk mengidentifikasi pesan yang tidak perlu ditampilkan di UI
            
        def write(self, message):
            # Write ke terminal asli
            self.terminal.write(message)
            
            # Skip semua pesan yang tidak kritis untuk cell_1_x, cell_2_1, dan cell_2_2
            msg_strip = message.strip()
            if not msg_strip or len(msg_strip) < 2:  # Skip pesan kosong atau terlalu pendek
                return
                
            # Filter semua log inisialisasi handler dan log non-critical
            if any(prefix in msg_strip for prefix in self.ignore_debug_prefix) or \
               'inisialisasi' in msg_strip.lower() or 'setup' in msg_strip.lower() or \
               'handler' in msg_strip.lower() or 'initializing' in msg_strip.lower() or \
               'module' in msg_strip.lower() or 'preprocessing' in msg_strip.lower():
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