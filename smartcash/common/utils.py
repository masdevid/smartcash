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