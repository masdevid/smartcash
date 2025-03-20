"""
File: smartcash/ui/setup/directory_handler.py
Deskripsi: Handler untuk setup struktur direktori proyek dengan integrasi UI utils
"""

import os
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.ui.utils.file_utils import directory_tree, format_file_size

def handle_directory_setup(ui_components: Dict[str, Any]):
    """
    Setup struktur direktori proyek dengan integrasi UI utils.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    with ui_components['status']:
        clear_output()
        display(create_info_alert(f"Membuat struktur direktori...", "info", ICONS['processing']))
    
    try:
        setup_directory_structure(ui_components)
    except Exception as e:
        with ui_components['status']:
            clear_output()
            display(create_status_indicator('error', f"Error saat membuat struktur direktori: {str(e)}"))
        
        # Log error jika logger tersedia
        if 'logger' in ui_components and ui_components['logger']:
            ui_components['logger'].error(f"❌ Error setup direktori: {str(e)}")

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

def display_directory_tree(ui_components: Dict[str, Any]):
    """
    Tampilkan struktur direktori dalam format tree dengan file_utils.
    
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
            # Gunakan utility directory_tree dari file_utils untuk menampilkan tree
            project_path = Path.cwd()
            tree_html = directory_tree(
                project_path, 
                max_depth=2,
                exclude_patterns=[r'\.git', r'\.vscode', r'__pycache__', r'\.ipynb_checkpoints'],
                include_only=None
            )
            display(HTML(tree_html))
        except Exception as e:
            display(create_status_indicator('warning', f"Tidak dapat menampilkan struktur direktori: {str(e)}"))