"""
File: smartcash/ui/setup/directory_handler.py
Deskripsi: Handler untuk setup direktori lokal SmartCash dengan integrasi UI logger
"""

import os, time, shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from IPython.display import display, HTML, clear_output
from concurrent.futures import ThreadPoolExecutor

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.ui_logger import log_to_ui
from smartcash.common.utils import is_colab

def handle_directory_setup(ui_components: Dict[str, Any], custom_path: Optional[str] = None, silent: bool = False):
    """
    Setup direktori lokal SmartCash.
    
    Args:
        ui_components: Dictionary komponen UI
        custom_path: Path kustom untuk direktori (opsional)
        silent: Jika True, tidak menampilkan output UI
    """
    logger = ui_components.get('logger')
    
    # Update progress tracking
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Menyiapkan direktori..."
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
    
    # Log ke UI langsung
    log_to_ui(ui_components, "Memulai setup direktori lokal...", "info", ICONS.get('processing', '🔄'))
    
    try:
        # Jalankan setup dalam thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            setup_future = executor.submit(setup_project_directories, ui_components, custom_path, silent)
            base_dir = setup_future.result()
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 2
            ui_components['progress_message'].value = "Menyiapkan symlinks..."
        
        # Siapkan symlinks jika di Colab (untuk lokal tidak perlu)
        if is_colab():
            # Coba setup symlinks jika di Colab
            try:
                from smartcash.ui.setup.drive_handler import create_symlinks
                log_to_ui(ui_components, "Menyiapkan symlinks di Colab...", "info", "🔗")
                
                # Cari drive path
                from smartcash.ui.utils.drive_utils import detect_drive_mount
                is_mounted, drive_path = detect_drive_mount()
                
                if is_mounted:
                    smartcash_dir = Path(drive_path) / 'SmartCash'
                    
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        symlink_future = executor.submit(create_symlinks, smartcash_dir, ui_components, silent)
                        symlink_future.result()
            except Exception as e:
                log_to_ui(ui_components, f"Error saat menyiapkan symlinks: {str(e)}", "warning", "⚠️")
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 4
            ui_components['progress_message'].value = "Menyelesaikan setup..."
        
        # Log ke UI
        log_to_ui(ui_components, f"Setup direktori berhasil di {base_dir}", "success", "✅")
        
        # Reset progress setelah selesai
        if hasattr(ui_components, 'reset_progress') and callable(ui_components['reset_progress']):
            ui_components['reset_progress']()
            
    except Exception as e:
        # Update progress pada error
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 5
            ui_components['progress_message'].value = f"Error: {str(e)[:30]}..."
        
        # Log error ke UI
        log_to_ui(ui_components, f"Error saat setup direktori: {str(e)}", "error", "❌")
        
        # Reset progress setelah error
        if hasattr(ui_components, 'reset_progress') and callable(ui_components['reset_progress']):
            ui_components['reset_progress']()

def setup_project_directories(ui_components: Dict[str, Any], custom_path: Optional[str] = None, silent: bool = False) -> Path:
    """
    Buat struktur direktori proyek yang diperlukan.
    
    Args:
        ui_components: Dictionary komponen UI
        custom_path: Path kustom untuk direktori (opsional)
        silent: Jika True, tidak menampilkan output UI
        
    Returns:
        Path direktori dasar
    """
    logger = ui_components.get('logger')
    
    # Tentukan base dir
    if custom_path:
        base_dir = Path(custom_path)
    else:
        # Cek jika di Colab
        if is_colab():
            base_dir = Path("/content/SmartCash")
        else:
            base_dir = Path.cwd()
    
    log_to_ui(ui_components, f"Menyiapkan direktori di {base_dir}", "info")
        
    # Update progress
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = 1
        ui_components['progress_message'].value = f"Base dir: {base_dir}"
    
    # Direktori utama yang diperlukan - konsolisasi one-liner dengan list comprehension
    [(base_dir / dir_name).mkdir(parents=True, exist_ok=True) for dir_name in [
        'configs',
        'data', 'data/train', 'data/train/images', 'data/train/labels',
        'data/valid', 'data/valid/images', 'data/valid/labels',
        'data/test', 'data/test/images', 'data/test/labels',
        'runs', 'runs/train', 'runs/train/weights',
        'logs', 'checkpoints'
    ]]
    
    # Log ke UI
    log_to_ui(ui_components, f"Struktur direktori berhasil dibuat di {base_dir}", "success", "✅")
        
    return base_dir
