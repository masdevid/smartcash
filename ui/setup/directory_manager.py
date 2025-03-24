"""
File: smartcash/ui/setup/directory_manager.py
Deskripsi: Manager direktori proyek untuk membuat struktur folder yang diperlukan
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

def setup_project_directories(ui_components: Dict[str, Any], custom_path: Optional[str] = None) -> Tuple[bool, Path]:
    """
    Buat struktur direktori proyek yang diperlukan.
    
    Args:
        ui_components: Dictionary komponen UI
        custom_path: Path kustom untuk direktori (opsional)
        
    Returns:
        Tuple (success, base_dir)
    """
    logger = ui_components.get('logger')
    
    try:
        # Tentukan base dir
        base_dir = _determine_base_dir(custom_path)
        
        # Log informasi
        _log_to_ui(ui_components, f"Menyiapkan direktori di {base_dir}", "info", "📂")
        
        # Update progress jika tersedia
        _update_progress(ui_components, 1, f"Base dir: {base_dir}")
        
        # Buat struktur direktori dengan list comprehension
        _create_directory_structure(base_dir)
        
        # Log sukses
        _log_to_ui(ui_components, f"Struktur direktori berhasil dibuat di {base_dir}", "success", "✅")
        
        return True, base_dir
    except Exception as e:
        _log_to_ui(ui_components, f"Error saat setup direktori: {str(e)}", "error", "❌")
        return False, Path.cwd()

def _determine_base_dir(custom_path: Optional[str] = None) -> Path:
    """
    Tentukan direktori dasar untuk proyek.
    
    Args:
        custom_path: Path kustom yang ditentukan pengguna
        
    Returns:
        Path direktori dasar
    """
    # Jika custom path tersedia, gunakan
    if custom_path:
        return Path(custom_path)
    
    # Cek jika di Colab
    try:
        from smartcash.common.utils import is_colab
        if is_colab():
            return Path("/content/SmartCash")
    except ImportError:
        # Fallback: cek manual
        import sys
        if 'google.colab' in sys.modules:
            return Path("/content/SmartCash")
    
    # Default: gunakan current directory
    return Path.cwd()

def _create_directory_structure(base_dir: Path) -> None:
    """
    Buat struktur direktori proyek.
    
    Args:
        base_dir: Direktori dasar proyek
    """
    # Daftar direktori yang perlu dibuat - gunakan list comprehension untuk membuat
    [(base_dir / dir_name).mkdir(parents=True, exist_ok=True) for dir_name in [
        'configs',
        'data', 'data/train', 'data/train/images', 'data/train/labels',
        'data/valid', 'data/valid/images', 'data/valid/labels',
        'data/test', 'data/test/images', 'data/test/labels',
        'data/preprocessed', 'data/preprocessed/train', 'data/preprocessed/valid', 'data/preprocessed/test',
        'runs', 'runs/train', 'runs/train/weights',
        'logs', 'checkpoints'
    ]]

def create_project_structure_async(ui_components: Dict[str, Any], custom_path: Optional[str] = None) -> None:
    """
    Asinkron membuat struktur direktori proyek.
    
    Args:
        ui_components: Dictionary komponen UI
        custom_path: Path kustom untuk direktori (opsional)
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(setup_project_directories, ui_components, custom_path)
        try:
            success, base_dir = future.result()
        except Exception as e:
            _log_to_ui(ui_components, f"Error saat membuat struktur direktori: {str(e)}", "error", "❌")

def _update_progress(ui_components: Dict[str, Any], value: int, message: str) -> None:
    """Update progress bar jika tersedia."""
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = value
        ui_components['progress_message'].value = message

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", emoji: str = "") -> None:
    """Log pesan ke UI dan logger."""
    logger = ui_components.get('logger')
    
    # Log ke UI
    from smartcash.ui.utils.ui_logger import log_to_ui
    log_to_ui(ui_components, message, level, emoji)
    
    # Log ke logger jika tersedia
    if logger:
        if level == "error": logger.error(message)
        elif level == "warning": logger.warning(message)
        elif level == "success": logger.success(message)
        else: logger.info(message)