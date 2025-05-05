"""
File: smartcash/ui/setup/environment_symlink_helper.py
Deskripsi: Helper untuk membuat symlinks dari Colab ke Google Drive
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

def create_drive_symlinks(drive_path: Path, ui_components: Dict[str, Any] = None) -> Dict[str, int]:
    """
    Buat symlink dari direktori lokal ke direktori Google Drive.
    
    Args:
        drive_path: Path Google Drive
        ui_components: Dictionary UI components untuk logging
        
    Returns:
        Statistik pembuatan symlink
    """
    logger = ui_components.get('logger') if ui_components else None
    
    # Mapping symlink (perbaikan: langsung ke /content)
    symlinks = {
        'data': drive_path / 'data',
        'configs': drive_path / 'configs', 
        'runs': drive_path / 'runs',
        'logs': drive_path / 'logs',
        'checkpoints': drive_path / 'checkpoints'
    }
    
    stats = {'created': 0, 'existing': 0, 'error': 0}
    total_symlinks = len(symlinks)
    
    # Progress tracking
    def update_progress(idx: int, total: int, message: str):
        if ui_components and 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = idx
            ui_components['progress_bar'].max = total
            ui_components['progress_message'].value = message
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_message'].layout.visibility = 'visible'
    
    # Log ke UI
    def log_to_ui(message: str, level: str = "info"):
        if ui_components and 'status' in ui_components:
            from smartcash.ui.utils.ui_logger import log_to_ui
            icon = "✅" if level == "success" else "⚠️" if level == "warning" else "❌" if level == "error" else "ℹ️"
            log_to_ui(ui_components, message, level, icon)
    
    for idx, (local_name, target_path) in enumerate(symlinks.items()):
        update_progress(idx + 1, total_symlinks, f"Membuat symlink: {local_name} -> {target_path}")
            
        try:
            # Pastikan direktori target ada
            os.makedirs(target_path, exist_ok=True)
            
            # FIX: Gunakan path langsung ke /content
            local_path = Path('/content') / local_name
            
            # Cek jika path lokal ada dan bukan symlink
            if local_path.exists() and not local_path.is_symlink():
                backup_path = local_path.with_name(f"{local_name}_backup")
                log_to_ui(f"Memindahkan direktori lokal ke backup: {local_path} -> {backup_path}", "info")
                
                # Hapus backup yang sudah ada
                if backup_path.exists(): shutil.rmtree(backup_path)
                
                # Pindahkan direktori lokal ke backup
                local_path.rename(backup_path)
            
            # Buat symlink jika belum ada
            if not local_path.exists():
                local_path.symlink_to(target_path)
                stats['created'] += 1
                log_to_ui(f"Symlink berhasil dibuat: {local_name} -> {target_path}", "success")
                if logger: logger.info(f"🔗 Symlink berhasil dibuat: {local_name} -> {target_path}")
            else:
                stats['existing'] += 1
        except Exception as e:
            stats['error'] += 1
            log_to_ui(f"Error pembuatan symlink: {local_name} - {str(e)}", "error")
            if logger: logger.warning(f"⚠️ Error pembuatan symlink: {local_name} - {str(e)}")
    
    return stats