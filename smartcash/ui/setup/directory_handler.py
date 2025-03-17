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
            <div style="padding:8px; background-color:#d1ecf1; color:#0c5460; border-radius:4px">
                <p style="margin:5px 0">🔄 Membuat struktur direktori...</p>
            </div>
        """))
    
    try:
        setup_directory_structure(ui_components)
    except Exception as e:
        with ui_components['status']:
            clear_output()
            display(HTML(f"""
                <div style="padding:8px; background-color:#f8d7da; color:#721c24; border-radius:4px">
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
            <div style="padding:8px; background-color:#d4edda; color:#155724; border-radius:4px; margin-top:10px">
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
            <div style="margin-top:15px">
                <h3 style="color:#155724">📂 Struktur Direktori Project</h3>
            </div>
        """))
        
        try:
            # Coba gunakan environment manager
            from smartcash.common.environment import get_environment_manager
            env = get_environment_manager()
            tree_html = env.get_directory_tree(Path.cwd(), max_depth=2)
            display(HTML(tree_html))
        except Exception:
            # Fallback ke implementasi sederhana
            directory_tree_fallback(ui_components)

def directory_tree_fallback(ui_components: Dict[str, Any], max_depth: int = 2):
    """
    Implementasi fallback untuk directory tree jika environment manager tidak tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        max_depth: Kedalaman maksimum directory tree
    """
    try:
        # Implementasi sederhana untuk directory tree
        tree = "<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n"
        cwd = Path.cwd()
        tree += f"<span style='color:#0366d6; font-weight:bold;'>{cwd.name}/</span>\n"
        
        # Buat tree rekursif
        def _add_tree(path, prefix="", depth=0):
            if depth >= max_depth:
                return ""
                
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            last_idx = len(items) - 1
            result = ""
            
            for i, item in enumerate(items):
                is_last = i == last_idx
                connector = "└─ " if is_last else "├─ "
                next_prefix = prefix + ("   " if is_last else "│  ")
                
                if item.is_dir() and not item.name.startswith('.'):
                    result += f"{prefix}{connector}<span style='color:#0366d6; font-weight:bold;'>{item.name}/</span>\n"
                    if depth < max_depth - 1:
                        result += _add_tree(item, next_prefix, depth + 1)
                elif not item.name.startswith('.'):
                    result += f"{prefix}{connector}{item.name}\n"
                    
            return result
        
        # Tambahkan tree direktori
        tree += _add_tree(cwd)
        tree += "</pre>"
        
        with ui_components['status']:
            display(HTML(tree))
    except Exception as e:
        with ui_components['status']:
            display(HTML(f"""
                <div style="padding:8px; background-color:#fff3cd; color:#856404; border-radius:4px">
                    <p style="margin:5px 0">⚠️ Tidak dapat menampilkan struktur direktori: {str(e)}</p>
                </div>
            """))