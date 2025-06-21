
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

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Menggabungkan dua dictionary secara rekursif.
    
    Args:
        dict1: Dictionary pertama
        dict2: Dictionary kedua yang akan digabungkan ke dict1
        
    Returns:
        Dictionary hasil penggabungan
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Jika kedua nilai adalah dictionary, gabungkan secara rekursif
            result[key] = deep_merge(result[key], value)
        else:
            # Jika tidak, ganti nilai dengan nilai dari dict2
            result[key] = value
    
    return result