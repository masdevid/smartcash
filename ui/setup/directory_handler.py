"""
File: smartcash/ui/setup/directory_handler.py
Deskripsi: Handler untuk setup struktur direktori proyek dengan UI yang ditingkatkan
"""

import os
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS

def handle_directory_setup(ui_components: Dict[str, Any]):
    """
    Setup struktur direktori proyek.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    with ui_components['status']:
        clear_output()
        display(HTML(f"""
            <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                      color:{COLORS['alert_info_text']}; 
                      border-radius:4px; margin:5px 0;
                      border-left:4px solid {COLORS['alert_info_text']};">
                <p style="margin:5px 0">{ICONS['processing']} Membuat struktur direktori...</p>
            </div>
        """))
    
    try:
        setup_directory_structure(ui_components)
    except Exception as e:
        with ui_components['status']:
            clear_output()
            display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                          color:{COLORS['alert_danger_text']}; 
                          border-radius:4px; margin:5px 0;
                          border-left:4px solid {COLORS['alert_danger_text']};">
                    <p style="margin:5px 0">{ICONS['error']} Error saat membuat struktur direktori: {str(e)}</p>
                </div>
            """))

def setup_directory_structure(ui_components: Dict[str, Any]):
    """
    Setup struktur direktori standar untuk proyek.
    
    Args:
        ui_components: Dictionary komponen UI
    """
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
    
    with ui_components['status']:
        clear_output()
        for dir_path in directories:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                stats['created'] += 1
                display(HTML(f"""
                    <div style="padding:4px; color:{COLORS['alert_success_text']}">
                        <p style="margin:5px 0">{ICONS['success']} Direktori dibuat: {dir_path}</p>
                    </div>
                """))
            else:
                stats['existing'] += 1
        
        # Tampilkan ringkasan
        color = COLORS['alert_success_text']
        bg_color = COLORS['alert_success_bg']
        display(HTML(f"""
            <div style="padding:10px; background-color:{bg_color}; 
                      color:{color}; border-radius:4px; margin:5px 0;
                      border-left:4px solid {color};">
                <p style="margin:5px 0">{ICONS['success']} Struktur direktori selesai dibuat: 
                   <strong>{stats['created']} direktori baru</strong>, 
                   <strong>{stats['existing']} sudah ada</strong>
                </p>
            </div>
        """))
        
        # Tampilkan struktur direktori
        display_directory_tree(ui_components)

def display_directory_tree(ui_components: Dict[str, Any]):
    """
    Tampilkan struktur direktori dalam format tree.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    with ui_components['status']:
        display(HTML(f"""
            <div style="margin-top:15px">
                <h3 style="color:{COLORS['secondary']}; margin:5px 0">📂 Struktur Direktori Project</h3>
            </div>
        """))
        
        try:
            # Dapatkan direktori project
            project_path = Path.cwd()
            tree_html = create_project_tree(project_path)
            display(HTML(tree_html))
        except Exception as e:
            display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                          color:{COLORS['alert_warning_text']}; 
                          border-radius:4px; margin:5px 0;
                          border-left:4px solid {COLORS['alert_warning_text']};">
                    <p style="margin:5px 0">{ICONS['warning']} Tidak dapat menampilkan struktur direktori: {str(e)}</p>
                </div>
            """))

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
    
    for dir_path in sorted(other_dirs):
        symbol, color = get_dir_symbol(dir_path.name)
        tree += f"├─ <span style='color:{color};'>{symbol} {dir_path.name}/</span>\n"
    
    tree += "</pre>"
    return tree