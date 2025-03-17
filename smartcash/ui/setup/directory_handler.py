"""
File: smartcash/ui/setup/directory_handler.py
Deskripsi: Handler untuk setup struktur direktori proyek
"""

import os
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, HTML, clear_output

def handle_directory_setup(ui_components: Dict[str, Any]):
    """
    Setup struktur direktori proyek.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    with ui_components['status']:
        clear_output()
        display(HTML("""
            <div style="padding:10px; background-color:#d1ecf1; color:#0c5460; border-radius:4px; margin:5px 0">
                <p style="margin:5px 0">🔄 Membuat struktur direktori...</p>
            </div>
        """))
    
    try:
        setup_directory_structure(ui_components)
    except Exception as e:
        with ui_components['status']:
            clear_output()
            display(HTML(f"""
                <div style="padding:10px; background-color:#f8d7da; color:#721c24; border-radius:4px; margin:5px 0">
                    <p style="margin:5px 0">❌ Error saat membuat struktur direktori: {str(e)}</p>
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
                    <div style="padding:4px; color:#155724">
                        <p style="margin:5px 0">✅ Direktori dibuat: {dir_path}</p>
                    </div>
                """))
            else:
                stats['existing'] += 1
        
        display(HTML(f"""
            <div style="padding:10px; background-color:#d4edda; color:#155724; border-radius:4px; margin:5px 0">
                <p style="margin:5px 0">✅ Struktur direktori selesai dibuat: {stats['created']} direktori baru, {stats['existing']} sudah ada</p>
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
        display(HTML("""
            <div style="margin-top:10px">
                <h3 style="color:inherit; margin:5px 0">📂 Struktur Direktori Project</h3>
            </div>
        """))
        
        try:
            # Dapatkan direktori project
            project_path = Path.cwd()
            tree_html = create_project_tree(project_path)
            display(HTML(tree_html))
        except Exception as e:
            display(HTML(f"""
                <div style="padding:10px; background-color:#fff3cd; color:#856404; border-radius:4px; margin:5px 0">
                    <p style="margin:5px 0">⚠️ Tidak dapat menampilkan struktur direktori: {str(e)}</p>
                </div>
            """))

def create_project_tree(project_path: Path) -> str:
    """
    Buat struktur direktori yang hanya menampilkan folder project.
    
    Args:
        project_path: Path direktori project
        
    Returns:
        HTML string berisi tree direktori
    """
    tree = "<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n"
    tree += f"<span style='color:#0366d6; font-weight:bold;'>{project_path.name}/</span>\n"
    
    # Dapatkan folder-folder di root project
    for item in sorted(project_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Hanya tampilkan folder (tanpa subfolder)
            tree += f"├─ <span style='color:#0366d6; font-weight:bold;'>{item.name}/</span>\n"
    
    tree += "</pre>"
    return tree