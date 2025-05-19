"""
File: smartcash/ui/setup/env_config/handlers/directory_button_handler.py
Deskripsi: Handler untuk tombol setup direktori
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Tuple
import os
from pathlib import Path

from smartcash.ui.setup.env_config.utils.environment_detector import detect_environment
from smartcash.ui.setup.env_config.utils.ui_helpers import disable_ui_during_processing, cleanup_ui
from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.common.environment import get_environment_manager
from smartcash.ui.utils.ui_logger import log_to_ui

def setup_directory_button_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk tombol setup direktori
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Register handler
    if 'directory_button' in ui_components:
        ui_components['directory_button'].on_click(
            lambda b: on_directory_button_click(b, ui_components)
        )

def on_directory_button_click(b: widgets.Button, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol setup direktori
    
    Args:
        b: Button widget
        ui_components: Dictionary berisi komponen UI
    """
    # Simpan status tombol Drive
    drive_button_disabled = ui_components['drive_button'].disabled if 'drive_button' in ui_components else False
    drive_button_description = ui_components['drive_button'].description if 'drive_button' in ui_components else "Hubungkan Drive"
    drive_button_tooltip = ui_components['drive_button'].tooltip if 'drive_button' in ui_components else "Hubungkan ke Google Drive"
    drive_button_icon = ui_components['drive_button'].icon if 'drive_button' in ui_components else "cloud"
    
    # Nonaktifkan UI selama proses
    disable_ui_during_processing(ui_components, True)
    
    # Update status panel
    ui_components['status_panel'].value = create_info_box(
        "Menyiapkan Direktori", 
        "Sedang menyiapkan struktur direktori...",
        style="info"
    ).value
    
    # Log info
    log_to_ui(ui_components, "Menyiapkan struktur direktori...", "info", "🔄")
    
    try:
        # Dapatkan environment manager
        env_manager = get_environment_manager()
        
        # Buat struktur direktori
        success, message, created_dirs = setup_directory_structure(env_manager)
        
        if success:
            # Log success
            log_to_ui(ui_components, message, "success", "✅")
            
            # Log direktori yang dibuat
            for dir_path in created_dirs:
                log_to_ui(ui_components, f"Dibuat: {dir_path}", "info", "📁")
            
            # Update status panel
            ui_components['status_panel'].value = create_info_box(
                "Direktori Siap", 
                f"Struktur direktori berhasil dibuat. {len(created_dirs)} direktori dibuat.",
                style="success"
            ).value
            
            # Detect environment lagi
            detect_environment(ui_components, env_manager)
        else:
            # Log error
            log_to_ui(ui_components, message, "error", "❌")
            
            # Update status panel
            ui_components['status_panel'].value = create_info_box(
                "Error", 
                f"Gagal menyiapkan struktur direktori: {message}",
                style="error"
            ).value
    except Exception as e:
        # Log error
        log_to_ui(ui_components, f"Error: {str(e)}", "error", "❌")
        
        # Update status panel
        ui_components['status_panel'].value = create_info_box(
            "Error", 
            f"Terjadi kesalahan: {str(e)}",
            style="error"
        ).value
    finally:
        # Cleanup UI dengan mempertahankan status tombol Drive
        cleanup_ui(ui_components)
        
        # Kembalikan status tombol Drive jika sebelumnya disabled
        if drive_button_disabled and 'drive_button' in ui_components:
            ui_components['drive_button'].disabled = True
            ui_components['drive_button'].description = drive_button_description
            ui_components['drive_button'].tooltip = drive_button_tooltip
            ui_components['drive_button'].icon = drive_button_icon

def setup_directory_structure(env_manager) -> Tuple[bool, str, List[str]]:
    """
    Setup struktur direktori untuk aplikasi
    
    Args:
        env_manager: Environment manager
    
    Returns:
        Tuple (success, message, created_dirs)
    """
    # Direktori yang perlu dibuat
    dirs_to_create = [
        "configs",
        "data",
        "data/raw",
        "data/processed",
        "models",
        "models/checkpoints",
        "models/weights",
        "output",
        "logs"
    ]
    
    # Direktori yang berhasil dibuat
    created_dirs = []
    
    try:
        # Buat direktori di base_dir
        base_dir = env_manager.base_dir
        for dir_name in dirs_to_create:
            dir_path = Path(base_dir) / dir_name
            if not dir_path.exists():
                os.makedirs(dir_path, exist_ok=True)
                created_dirs.append(str(dir_path))
        
        # Buat direktori di drive jika terhubung
        if env_manager.is_drive_mounted:
            drive_dir = env_manager.drive_path
            for dir_name in dirs_to_create:
                dir_path = drive_dir / dir_name
                if not dir_path.exists():
                    os.makedirs(dir_path, exist_ok=True)
                    created_dirs.append(str(dir_path))
        
        return True, f"Berhasil membuat {len(created_dirs)} direktori", created_dirs
    except Exception as e:
        return False, f"Error: {str(e)}", created_dirs
