# CM01 - SmartCash Common Module Architecture Guide

Dokumen ini menjelaskan arsitektur modul `common` yang menyediakan komponen utilitas terpusat yang dapat digunakan oleh semua modul lain dalam project SmartCash.

## Struktur Direktori

```
smartcash/common/
│
├── __init__.py             # Ekspor utilitas umum
├── config.py               # Utilitas konfigurasi
├── constants.py            # Konstanta global
├── logger.py               # Sistem logging
├── utils.py                # Utilitas umum
├── types.py                # Type definitions
└── exceptions.py           # Custom exceptions
```

## Komponen Utama

### Config Manager

`config.py` menyediakan ConfigManager yang menghandle loading, saving, dan merging konfigurasi dari file YAML atau sumber lainnya:

```python
# smartcash/common/config.py
"""
File: smartcash/common/config.py
Deskripsi: Manager konfigurasi dengan dukungan YAML dan environment variables
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

class ConfigManager:
    """
    Manager untuk konfigurasi aplikasi dengan dukungan untuk:
    - Loading dari file YAML/JSON
    - Environment variable overrides
    - Hierarki konfigurasi
    """
    
    DEFAULT_CONFIG_DIR = 'config'
    
    def __init__(self, 
                base_dir: Optional[str] = None, 
                config_file: Optional[str] = None,
                env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager.
        
        Args:
            base_dir: Direktori root project
            config_file: Path file konfigurasi utama (relatif ke base_dir)
            env_prefix: Prefix untuk environment variables
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config_dir = self.base_dir / self.DEFAULT_CONFIG_DIR
        self.env_prefix = env_prefix
        
        # Konfigurasi dasar
        self.config = {}
        
        # Muat konfigurasi dari file jika disediakan
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load konfigurasi dari file.
        
        Args:
            config_file: Nama file konfigurasi atau path lengkap
            
        Returns:
            Dictionary konfigurasi
        """
        # Tentukan path konfigurasi
        config_path = self._resolve_config_path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        # Load berdasarkan ekstensi file
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Format file konfigurasi tidak didukung: {config_path.suffix}")
        
        # Override dengan environment variables
        self._override_with_env_vars()
        
        return self.config
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """Resolve path konfigurasi relatif atau absolut."""
        config_path = Path(config_file)
        
        # Jika path absolut, gunakan langsung
        if config_path.is_absolute():
            return config_path
            
        # Jika config ada di config_dir, gunakan
        if (self.config_dir / config_path).exists():
            return self.config_dir / config_path
            
        # Jika config ada di direktori kerja, gunakan
        if (Path.cwd() / config_path).exists():
            return Path.cwd() / config_path
            
        # Default ke config_dir
        return self.config_dir / config_path
    
    def _override_with_env_vars(self) -> None:
        """Override konfigurasi dengan environment variables."""
        for env_name, env_value in os.environ.items():
            # Hanya proses var dengan prefix yang sesuai
            if not env_name.startswith(self.env_prefix):
                continue
                
            # Convert format ENV_VAR ke nested dict
            # contoh: SMARTCASH_MODEL_IMG_SIZE_WIDTH=640 -> config['model']['img_size']['width']=640
            config_path = env_name[len(self.env_prefix):].lower().split('_')
            
            # Traverse & update config dict
            current = self.config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set nilai (dengan auto type conversion)
            key = config_path[-1]
            current[key] = self._parse_env_value(env_value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse nilai environment variable ke tipe yang sesuai."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
            
        # Numbers
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
            
        # Lists (comma-separated values)
        if ',' in value:
            return [self._parse_env_value(item.strip()) for item in value.split(',')]
            
        # Default: string
        return value
    
    def get(self, key: str, default=None) -> Any:
        """
        Ambil nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key dengan dot notation (e.g., 'model.img_size.width')
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key dengan dot notation (e.g., 'model.img_size.width')
            value: Nilai yang akan di-set
        """
        parts = key.split('.')
        current = self.config
        
        # Traverse sampai level terakhir
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set nilai
        current[parts[-1]] = value
    
    def merge_config(self, config: Union[Dict, str]) -> Dict[str, Any]:
        """
        Merge konfigurasi dari dict atau file.
        
        Args:
            config: Dictionary konfigurasi atau path file
            
        Returns:
            Konfigurasi setelah merge
        """
        # Load dari file jika string
        if isinstance(config, str):
            config_path = self._resolve_config_path(config)
            
            if config_path.suffix.lower() in ('.yml', '.yaml'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Format file tidak didukung: {config_path.suffix}")
        
        # Deep merge config
        self._deep_merge(self.config, config)
        return self.config
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """
        Deep merge dua dictionary.
        
        Args:
            target: Dictionary target
            source: Dictionary source
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Rekursif untuk nested dict
                self._deep_merge(target[key], value)
            else:
                # Override atau tambahkan key baru
                target[key] = value
    
    def save_config(self, config_file: str, create_dirs: bool = True) -> None:
        """
        Simpan konfigurasi ke file.
        
        Args:
            config_file: Path file untuk menyimpan konfigurasi
            create_dirs: Flag untuk create direktori jika belum ada
        """
        config_path = Path(config_file)
        
        # Buat direktori jika diperlukan
        if create_dirs and not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan berdasarkan ekstensi
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        else:
            # Default ke YAML
            with open(f"{config_path}.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.set(key, value)

# Singleton instance
_config_manager = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance ConfigManager (singleton).
    
    Args:
        base_dir: Direktori root project
        config_file: Path file konfigurasi utama
        env_prefix: Prefix untuk environment variables
        
    Returns:
        Instance ConfigManager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(base_dir, config_file, env_prefix)
    return _config_manager
```

### Logger

`logger.py` menyediakan sistem logging dengan dukungan untuk emoji, warna, dan berbagai formatter:

```python
# smartcash/common/logger.py
"""
File: smartcash/common/logger.py
Deskripsi: Sistem logging terpusat dengan dukungan emoji, warna, dan callback
"""

import logging
import sys
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union, Callable, List
from enum import Enum, auto

class LogLevel(Enum):
    """Level log dengan emoji."""
    DEBUG = auto()
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class SmartCashLogger:
    """
    Logger untuk SmartCash dengan fitur:
    - Emoji untuk level log
    - Format teks berwarna
    - Output ke file dan console
    - Support callback untuk UI
    """
    
    # Emoji untuk log level
    EMOJIS = {
        LogLevel.DEBUG: '🐞',
        LogLevel.INFO: 'ℹ️',
        LogLevel.SUCCESS: '✅',
        LogLevel.WARNING: '⚠️',
        LogLevel.ERROR: '❌',
        LogLevel.CRITICAL: '🔥'
    }
    
    # Colors (ANSI color codes)
    COLORS = {
        LogLevel.DEBUG: '\033[90m',  # Gray
        LogLevel.INFO: '\033[0m',    # Default
        LogLevel.SUCCESS: '\033[92m', # Green
        LogLevel.WARNING: '\033[93m', # Yellow
        LogLevel.ERROR: '\033[91m',   # Red
        LogLevel.CRITICAL: '\033[91;1m' # Bold Red
    }
    
    # Reset ANSI color
    RESET_COLOR = '\033[0m'
    
    # Mapping ke logging level standar
    LEVEL_MAPPING = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.SUCCESS: logging.INFO,  # Custom level, map ke INFO
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL
    }
    
    def __init__(self, 
                name: str, 
                level: LogLevel = LogLevel.INFO,
                log_file: Optional[str] = None,
                use_colors: bool = True,
                use_emojis: bool = True,
                log_dir: str = 'logs'):
        """
        Inisialisasi SmartCashLogger.
        
        Args:
            name: Nama logger
            level: Level minimum log
            log_file: Path file log (auto-generated jika None)
            use_colors: Flag untuk menggunakan warna
            use_emojis: Flag untuk menggunakan emoji
            log_dir: Direktori untuk file log
        """
        self.name = name
        self.level = level
        self.use_colors = use_colors
        self.use_emojis = use_emojis
        self._callbacks = []
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LEVEL_MAPPING[level])
        
        # Hapus handler yang sudah ada
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Tambahkan console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.LEVEL_MAPPING[level])
        self.logger.addHandler(console_handler)
        
        # Tambahkan file handler jika diperlukan
        if log_file or log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Otomatis generate nama file jika tidak ada
            if not log_file:
                log_file = f"{self.log_dir}/{name}_{time.strftime('%Y%m%d')}.log"
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(self.LEVEL_MAPPING[level])
            
            # Format sederhana untuk file
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format pesan log dengan emoji dan warna."""
        formatted = ""
        
        # Tambahkan emoji jika diperlukan
        if self.use_emojis and level in self.EMOJIS:
            formatted += f"{self.EMOJIS[level]} "
            
        # Tambahkan pesan
        formatted += message
        
        # Tambahkan warna jika di console dan diizinkan
        if self.use_colors and level in self.COLORS:
            colored = f"{self.COLORS[level]}{formatted}{self.RESET_COLOR}"
            # Return versi berwarna untuk console, plain untuk file
            return colored, formatted
            
        # Tidak ada warna
        return formatted, formatted
    
    def log(self, level: LogLevel, message: str) -> None:
        """
        Log pesan dengan level tertentu.
        
        Args:
            level: Level log
            message: Pesan yang akan di-log
        """
        # Format pesan
        console_msg, file_msg = self._format_message(level, message)
        
        # Map ke level logging standar
        std_level = self.LEVEL_MAPPING[level]
        
        # Log via Python logger
        self.logger.log(std_level, file_msg)
        
        # Panggil callbacks jika ada
        for callback in self._callbacks:
            callback(level, message)
    
    def add_callback(self, callback: Callable[[LogLevel, str], None]) -> None:
        """
        Tambahkan callback untuk event log.
        
        Args:
            callback: Fungsi yang dipanggil saat log
        """
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """
        Hapus callback.
        
        Args:
            callback: Callback yang akan dihapus
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    # Convenience methods
    def debug(self, message: str) -> None:
        """Log pesan debug."""
        self.log(LogLevel.DEBUG, message)
    
    def info(self, message: str) -> None:
        """Log pesan info."""
        self.log(LogLevel.INFO, message)
    
    def success(self, message: str) -> None:
        """Log pesan sukses."""
        self.log(LogLevel.SUCCESS, message)
    
    def warning(self, message: str) -> None:
        """Log pesan warning."""
        self.log(LogLevel.WARNING, message)
    
    def error(self, message: str) -> None:
        """Log pesan error."""
        self.log(LogLevel.ERROR, message)
    
    def critical(self, message: str) -> None:
        """Log pesan critical."""
        self.log(LogLevel.CRITICAL, message)
    
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
            return tqdm(iterable, desc=desc, **kwargs)
        except ImportError:
            self.warning("tqdm tidak ditemukan, progress tracking tidak aktif")
            return iterable

# Fungsi helper untuk mendapatkan logger
def get_logger(name: str, 
              level: LogLevel = LogLevel.INFO, 
              log_file: Optional[str] = None,
              use_colors: bool = True,
              use_emojis: bool = True,
              log_dir: str = 'logs') -> SmartCashLogger:
    """
    Dapatkan instance SmartCashLogger.
    
    Args:
        name: Nama logger
        level: Level minimum log
        log_file: Path file log (auto-generated jika None)
        use_colors: Flag untuk menggunakan warna
        use_emojis: Flag untuk menggunakan emoji
        log_dir: Direktori untuk file log
        
    Returns:
        Instance SmartCashLogger
    """
    return SmartCashLogger(name, level, log_file, use_colors, use_emojis, log_dir)
```

### Constants

`constants.py` mendefinisikan konstanta yang digunakan di seluruh aplikasi:

```python
# smartcash/common/constants.py
"""
File: smartcash/common/constants.py
Deskripsi: Konstanta global yang digunakan di seluruh project
"""

from enum import Enum, auto
from pathlib import Path
import os

# Versi aplikasi
VERSION = "0.1.0"
APP_NAME = "SmartCash"

# Paths
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODEL_DIR = "models"
DEFAULT_LOGS_DIR = "logs"

# Google Drive paths (for Colab)
DRIVE_BASE_PATH = "/content/drive/MyDrive/SmartCash"

# Layer detection
class DetectionLayer(Enum):
    """Layer untuk deteksi objek."""
    BANKNOTE = "banknote"  # Deteksi uang kertas utuh
    NOMINAL = "nominal"    # Deteksi area nominal
    SECURITY = "security"  # Deteksi fitur keamanan

# Format Input/Output
class ModelFormat(Enum):
    """Format model yang didukung."""
    PYTORCH = auto()
    ONNX = auto()
    TORCHSCRIPT = auto()
    TENSORRT = auto()
    TFLITE = auto()

# File extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
MODEL_EXTENSIONS = {
    ModelFormat.PYTORCH: '.pt',
    ModelFormat.ONNX: '.onnx',
    ModelFormat.TORCHSCRIPT: '.pt',
    ModelFormat.TENSORRT: '.engine',
    ModelFormat.TFLITE: '.tflite'
}

# Environment variables
ENV_CONFIG_PATH = os.environ.get("SMARTCASH_CONFIG_PATH", "")
ENV_MODEL_PATH = os.environ.get("SMARTCASH_MODEL_PATH", "")
ENV_DATA_PATH = os.environ.get("SMARTCASH_DATA_PATH", "")

# Default values
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = (640, 640)

# Limits
MAX_BATCH_SIZE = 64
MAX_IMAGE_SIZE = 1280

# API settings (if applicable)
API_PORT = 8000
API_HOST = "0.0.0.0"
```

### Exceptions

`exceptions.py` mendefinisikan custom exceptions untuk SmartCash:

```python
# smartcash/common/exceptions.py
"""
File: smartcash/common/exceptions.py
Deskripsi: Custom exceptions untuk SmartCash
"""

class SmartCashError(Exception):
    """Exception dasar untuk semua error SmartCash."""
    pass

class ConfigError(SmartCashError):
    """Exception untuk error konfigurasi."""
    pass

class DatasetError(SmartCashError):
    """Exception untuk error terkait dataset."""
    pass

class ModelError(SmartCashError):
    """Exception untuk error terkait model."""
    pass

class DetectionError(SmartCashError):
    """Exception untuk error terkait proses deteksi."""
    pass

class FileError(SmartCashError):
    """Exception untuk error file I/O."""
    pass

class APIError(SmartCashError):
    """Exception untuk error API."""
    pass

class ValidationError(SmartCashError):
    """Exception untuk error validasi input."""
    pass

class NotSupportedError(SmartCashError):
    """Exception untuk fitur yang tidak didukung."""
    pass
```

### Types

`types.py` mendefinisikan type definitions yang digunakan di seluruh aplikasi:

```python
# smartcash/common/types.py
"""
File: smartcash/common/types.py
Deskripsi: Type definitions untuk SmartCash
"""

from typing import Dict, List, Tuple, Union, Optional, Any, TypedDict, NewType, Callable
import numpy as np
import torch
from enum import Enum, auto

# Type aliases
ImageType = Union[np.ndarray, str, bytes]
PathType = Union[str, 'Path']
TensorType = Union[torch.Tensor, np.ndarray]
ConfigType = Dict[str, Any]

# Callback types
ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str, str], None]

# Typed dictionaries
class BoundingBox(TypedDict):
    """Bounding box dengan format [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    
class Detection(TypedDict):
    """Hasil deteksi objek."""
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    layer: str

class ModelInfo(TypedDict):
    """Informasi model."""
    name: str
    version: str
    format: str
    input_size: Tuple[int, int]
    layers: List[str]
    classes: Dict[str, List[str]]
    
class DatasetStats(TypedDict):
    """Statistik dataset."""
    total_images: int
    total_annotations: int
    class_distribution: Dict[str, int]
    split_info: Dict[str, Dict[str, int]]
```

### Utils

`utils.py` menyediakan fungsi utilitas umum yang digunakan di seluruh aplikasi:

```python
# smartcash/common/utils.py
"""
File: smartcash/common/utils.py
Deskripsi: Fungsi utilitas umum untuk SmartCash
"""

import os
import sys
import shutil
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import uuid
import platform
import datetime

def is_colab() -> bool:
    """Cek apakah berjalan di Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_notebook() -> bool:
    """Cek apakah berjalan di Jupyter Notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except ImportError:
        return False

def get_system_info() -> Dict[str, str]:
    """Dapatkan informasi tentang sistem."""
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'system': platform.system(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0],
        'memory': 'unknown',
        'gpu': 'unknown'
    }
    
    # Dapatkan info RAM jika memungkinkan
    try:
        import psutil
        mem_info = psutil.virtual_memory()
        info['memory'] = f"{mem_info.total / (1024**3):.2f} GB"
    except ImportError:
        pass
    
    # Dapatkan info GPU jika memungkinkan
    try:
        import torch
        info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
        info['cuda_available'] = str(torch.cuda.is_available())
        info['cuda_version'] = torch.version.cuda or 'unknown'
    except ImportError:
        pass
    
    return info

def generate_unique_id() -> str:
    """Generate ID unik untuk eksperimen atau operasi."""
    return str(uuid.uuid4())

def format_time(seconds: float) -> str:
    """Format waktu dalam detik ke format yang lebih mudah dibaca."""
    if seconds < 60:
        return f"{seconds:.2f} detik"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)} menit {int(seconds)} detik"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)} jam {int(minutes)} menit {int(seconds)} detik"

def get_timestamp() -> str:
    """Dapatkan timestamp string format untuk nama file."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: Union[str, Path]) -> Path:
    """Pastikan direktori ada, jika tidak buat."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def copy_file(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> bool:
    """
    Copy file dari src ke dst.
    
    Args:
        src: Path sumber
        dst: Path tujuan
        overwrite: Flag untuk overwrite jika file tujuan sudah ada
        
    Returns:
        True jika berhasil, False jika gagal
    """
    src, dst = Path(src), Path(dst)
    
    # Cek jika file sumber ada
    if not src.exists():
        return False
        
    # Cek jika file tujuan sudah ada dan overwrite = False
    if dst.exists() and not overwrite:
        return False
        
    # Buat direktori tujuan jika belum ada
    if not dst.parent.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        
    # Copy file
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False

def file_exists(path: Union[str, Path]) -> bool:
    """Cek apakah file ada."""
    return Path(path).exists()

def file_size(path: Union[str, Path]) -> int:
    """Dapatkan ukuran file dalam bytes."""
    return Path(path).stat().st_size

def format_size(size_bytes: int) -> str:
    """Format ukuran dalam bytes ke format yang lebih mudah dibaca."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1048576:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1073741824:
        return f"{size_bytes/1048576:.2f} MB"
    else:
        return f"{size_bytes/1073741824:.2f} GB"

def load_json(path: Union[str, Path]) -> Dict:
    """Load data dari file JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict, path: Union[str, Path], pretty: bool = True) -> None:
    """Simpan data ke file JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(data, f, indent=2)
        else:
            json.dump(data, f)

def load_yaml(path: Union[str, Path]) -> Dict:
    """Load data dari file YAML."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict, path: Union[str, Path]) -> None:
    """Simpan data ke file YAML."""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)

def get_project_root() -> Path:
    """Dapatkan root direktori project."""
    script_path = Path(__file__).resolve()
    
    # Traverse up hingga menemukan root (ada file setup.py atau .git)
    current = script_path.parent
    while current != current.parent:  # Selama belum di root filesystem
        if (current / 'setup.py').exists() or (current / '.git').exists():
            return current
        current = current.parent
    
    # Fallback ke parent dari direktori common
    return script_path.parent.parent
