"""
File: smartcash/ui/setup/directory_handler.py
Deskripsi: Handler untuk setup struktur direktori proyek dengan integrasi UI utils dan progress tracking
"""

import os
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

def handle_directory_setup(ui_components: Dict[str, Any]):
    """
    Setup struktur direktori proyek dengan integrasi UI utils.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    # Update progress tracking
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Mempersiapkan setup direktori..."
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
    
    with ui_components['status']:
        clear_output()
        display(create_info_alert(f"Membuat struktur direktori...", "info", ICONS['processing']))
    
    try:
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 1
            ui_components['progress_message'].value = "Mendeteksi direktori yang ada..."
        
        setup_directory_structure(ui_components)
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 3
            ui_components['progress_message'].value = "Setup direktori selesai"
    except Exception as e:
        # Update progress pada error
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].value = 3
            ui_components['progress_message'].value = f"Error: {str(e)[:30]}..."
            
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator('error', f"Error saat membuat struktur direktori: {str(e)}"))
        
        # Log error jika logger tersedia
        if logger:
            logger.error(f"❌ Error setup direktori: {str(e)}")

def setup_directory_structure(ui_components: Dict[str, Any]):
    """
    Setup struktur direktori standar untuk proyek.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    # Struktur direktori standar
    directories = [
        "data/train/images", "data/train/labels",
        "data/valid/images", "data/valid/labels", 
        "data/test/images", "data/test/labels",
        "data/preprocessed/train", "data/preprocessed/valid", "data/preprocessed/test",
        "configs", "runs/train/weights", 
        "logs", "exports",
        "checkpoints"
    ]
    
    stats = {'created': 0, 'existing': 0}
    created_dirs = []
    
    with ui_components['status']:
        clear_output()
        total_dirs = len(directories)
        
        # Update progress
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].max = total_dirs
        
        for i, dir_path in enumerate(directories):
            # Update progress
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].value = i
                ui_components['progress_message'].value = f"Membuat: {dir_path}"
            
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                stats['created'] += 1
                created_dirs.append(dir_path)
                display(create_status_indicator('success', f"Direktori dibuat: {dir_path}"))
            else:
                stats['existing'] += 1
        
        # Tampilkan ringkasan menggunakan create_info_alert
        display(create_info_alert(
            f"Struktur direktori selesai dibuat: <strong>{stats['created']} direktori baru</strong>, <strong>{stats['existing']} sudah ada</strong>",
            "success", ICONS['success']
        ))
        
        # Tampilkan struktur direktori
        display_directory_tree(ui_components)
        
    # Log hasil
    if logger:
        logger.success(f"✅ Setup direktori selesai: {stats['created']} direktori baru, {stats['existing']} sudah ada")
        if created_dirs:
            logger.info(f"📁 Direktori yang dibuat: {', '.join(created_dirs[:5])}" + 
                      (f" dan {len(created_dirs)-5} lainnya" if len(created_dirs) > 5 else ""))

def display_directory_tree(ui_components: Dict[str, Any]):
    """
    Tampilkan struktur direktori dalam format tree yang hanya menampilkan folder utama project.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    with ui_components['status']:
        display(create_status_indicator('info', f'📂 Struktur Direktori Project'))
        
        try:
            # Dapatkan direktori project
            project_path = Path.cwd()
            tree_html = create_project_tree(project_path)
            display(HTML(tree_html))
        except Exception as e:
            display(create_status_indicator('warning', f"Tidak dapat menampilkan struktur direktori: {str(e)}"))

def create_project_tree(project_path: Path) -> str:
    """
    Buat struktur direktori yang hanya menampilkan folder utama project.
    
    Args:
        project_path: Path direktori project
        
    Returns:
        HTML string berisi tree direktori
    """
    tree = f"""<pre style="margin:0; padding:10px; background:{COLORS['light']}; 
                        font-family:monospace; color:{COLORS['dark']}; 
                        border-radius:4px; overflow:auto; border:1px solid {COLORS['border']}">
"""
    tree += f"<span style='color:{COLORS['primary']}; font-weight:bold;'>{project_path.name}/</span>\n"
    
    # Dapatkan folder-folder yang relevan
    important_dirs = ["data", "configs", "runs", "logs", "exports", "checkpoints"]
    
    # Fungsi untuk mendapatkan simbol dan warna berdasarkan jenis direktori
    def get_dir_symbol(dir_name):
        if dir_name == "data":
            return "📊", "#e91e63"  # Pink
        elif dir_name == "configs":
            return "⚙️", "#9c27b0"   # Purple
        elif dir_name == "runs":
            return "🏃", "#2196f3"   # Blue
        elif dir_name == "logs":
            return "📝", "#4caf50"   # Green
        elif dir_name == "exports":
            return "📦", "#ff9800"   # Orange
        elif dir_name == "checkpoints":
            return "💾", "#795548"   # Brown
        else:
            return "📁", COLORS['primary']
    
    # Tampilkan direktori penting terlebih dahulu
    for dir_name in important_dirs:
        dir_path = project_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            symbol, color = get_dir_symbol(dir_name)
            tree += f"├─ <span style='color:{color}; font-weight:bold;'>{symbol} {dir_name}/</span>\n"
    
    # Tampilkan direktori lainnya
    other_dirs = [d for d in project_path.iterdir() 
                  if d.is_dir() and not d.name.startswith('.') and d.name not in important_dirs]
    
    for i, dir_path in enumerate(sorted(other_dirs)):
        symbol, color = get_dir_symbol(dir_path.name)
        connector = "└─" if i == len(other_dirs) - 1 else "├─"
        tree += f"{connector} <span style='color:{color};'>{symbol} {dir_path.name}/</span>\n"
    
    tree += "</pre>"
    return tree