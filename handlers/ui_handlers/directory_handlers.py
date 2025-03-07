"""
File: smartcash/handlers/ui_handlers/directory_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen directory management, menangani setup direktori dan integrasi dengan Google Drive.
          Updated to use centralized EnvironmentManager.
"""

import os
import pickle
from pathlib import Path
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
import sys

# Import EnvironmentManager for centralized environment detection
from smartcash.utils.environment_manager import EnvironmentManager


def is_colab():
    """
    Cek apakah notebook berjalan di Google Colab.
    
    Returns:
        Boolean yang menunjukkan apakah berjalan di Colab
    """
    try:
        from google.colab import drive
        return True
    except ImportError:
        return False

def setup_google_drive(drive_path):
    """
    Setup Google Drive untuk penyimpanan.
    
    Args:
        drive_path: Path ke direktori SmartCash di Google Drive
        
    Returns:
        Tuple (success, message) menunjukkan keberhasilan dan pesan
    """
    try:
        from google.colab import drive
        
        # Cek apakah Google Drive sudah di-mount
        if not Path('/content/drive').exists() or not Path('/content/drive/MyDrive').exists():
            drive.mount('/content/drive')
        
        # Membuat direktori SmartCash di Drive jika belum ada
        os.makedirs(drive_path, exist_ok=True)
        
        return True, f"✅ Google Drive berhasil di-mount ke {drive_path}"
    except Exception as e:
        return False, f"❌ Error saat setup Google Drive: {str(e)}"

def create_directory_structure(base_dir):
    """
    Buat struktur direktori project.
    
    Args:
        base_dir: Path direktori dasar
        
    Returns:
        Dictionary berisi statistik direktori yang dibuat
    """
    directory_list = [
        "data/train/images",
        "data/train/labels",
        "data/valid/images",
        "data/valid/labels",
        "data/test/images",
        "data/test/labels",
        "configs",
        "runs/train/weights",
        "logs"
    ]
    
    stats = {
        'created': 0,
        'existing': 0,
        'error': 0
    }
    
    for d in directory_list:
        try:
            dir_path = Path(f"{base_dir}/{d}")
            if not dir_path.exists():
                os.makedirs(dir_path, exist_ok=True)
                stats['created'] += 1
            else:
                stats['existing'] += 1
        except Exception:
            stats['error'] += 1
    
    return stats

def create_symlinks(drive_path):
    """
    Buat symlink ke direktori Google Drive untuk akses yang lebih mudah.
    
    Args:
        drive_path: Path ke direktori SmartCash di Google Drive
        
    Returns:
        Dictionary berisi statistik symlink yang dibuat
    """
    stats = {
        'created': 0,
        'existing': 0,
        'error': 0
    }
    
    symlinks = {
        'data': f"{drive_path}/data",
        'configs': f"{drive_path}/configs",
        'runs': f"{drive_path}/runs",
        'logs': f"{drive_path}/logs"
    }
    
    for name, target in symlinks.items():
        try:
            if not os.path.exists(name):
                os.symlink(target, name)
                stats['created'] += 1
            else:
                stats['existing'] += 1
        except Exception:
            stats['error'] += 1
    
    return stats

def get_directory_tree(base_dir, max_depth=2):
    """
    Menghasilkan tree view dari direktori project dalam HTML.
    
    Args:
        base_dir: Path direktori dasar
        max_depth: Kedalaman maksimal directory tree
        
    Returns:
        String HTML yang menampilkan struktur direktori
    """
    def _get_tree(directory, prefix='', is_last=True, depth=0):
        if depth > max_depth:
            return ""
            
        base_name = os.path.basename(directory)
        result = ""
        
        if depth > 0:
            # Add connector line and directory name
            result += f"{prefix}{'└── ' if is_last else '├── '}<span style='color: #3498db;'>{base_name}</span><br/>"
        else:
            # Root directory
            result += f"<span style='color: #2980b9; font-weight: bold;'>{base_name}</span><br/>"
            
        # Update prefix for children
        if depth > 0:
            prefix += '    ' if is_last else '│   '
            
        # List directory contents
        try:
            items = list(sorted([x for x in Path(directory).iterdir()]))
            
            # Only show a subset of items if there are too many
            if len(items) > 10 and depth > 0:
                items = items[:10]
                show_ellipsis = True
            else:
                show_ellipsis = False
                
            for i, item in enumerate(items):
                if item.is_dir():
                    # Recursively process subdirectories
                    result += _get_tree(str(item), prefix, i == len(items) - 1 and not show_ellipsis, depth + 1)
                elif depth < max_depth:
                    # Add file name
                    result += f"{prefix}{'└── ' if i == len(items) - 1 and not show_ellipsis else '├── '}{item.name}<br/>"
                    
            if show_ellipsis:
                result += f"{prefix}└── <i>... dan item lainnya</i><br/>"
        except Exception:
            result += f"{prefix}└── <i>Error saat membaca direktori</i><br/>"
            
        return result
    
    html = "<div style='font-family: monospace;'>"
    html += _get_tree(base_dir)
    html += "</div>"
    
    return html

def on_setup_button_clicked(b, ui_components):
    """
    Handler untuk tombol setup direktori.
    
    Args:
        b: Button instance
        ui_components: Dictionary berisi komponen UI
    """
    # Disable tombol selama proses
    ui_components['setup_button'].disabled = True
    
    # Clear output area
    with ui_components['output_area']:
        clear_output()
        
        try:
            use_drive = ui_components['drive_checkbox'].value
            
            # Setup direktori berdasarkan pilihan
            if use_drive and is_colab():
                drive_path = ui_components['drive_path_text'].value
                success, message = setup_google_drive(drive_path)
                
                if success:
                    print(message)
                    base_dir = drive_path
                    
                    # Buat struktur direktori di Google Drive
                    dir_stats = create_directory_structure(base_dir)
                    print(f"📁 Struktur direktori dibuat di Google Drive:")
                    print(f"  • {dir_stats['created']} direktori baru dibuat")
                    print(f"  • {dir_stats['existing']} direktori sudah ada")
                    
                    # Buat symlink
                    link_stats = create_symlinks(base_dir)
                    print(f"🔗 Symlink dibuat untuk akses yang lebih mudah:")
                    print(f"  • {link_stats['created']} symlink baru dibuat")
                    print(f"  • {link_stats['existing']} symlink sudah ada")
                    
                    # Update status indicator
                    ui_components['status_indicator'].value = f"<p>Status: <span style='color: green'>✅ Setup selesai di Google Drive</span></p>"
                else:
                    print(message)
                    print("🔄 Beralih ke direktori lokal sebagai fallback...")
                    base_dir = os.getcwd()
                    dir_stats = create_directory_structure(base_dir)
                    
                    # Update status indicator
                    ui_components['status_indicator'].value = f"<p>Status: <span style='color: orange'>⚠️ Menggunakan direktori lokal</span></p>"
            else:
                # Gunakan direktori lokal
                base_dir = os.getcwd()
                dir_stats = create_directory_structure(base_dir)
                print(f"📁 Struktur direktori dibuat di lokal:")
                print(f"  • {dir_stats['created']} direktori baru dibuat")
                print(f"  • {dir_stats['existing']} direktori sudah ada")
                
                # Update status indicator
                ui_components['status_indicator'].value = f"<p>Status: <span style='color: blue'>✓ Setup selesai di direktori lokal</span></p>"
            
            # Simpan path base direktori untuk digunakan di cell lain
            with open('base_dir.pkl', 'wb') as f:
                pickle.dump(base_dir, f)
                
            print(f"🔍 Base direktori yang digunakan: {base_dir}")
            
            # Update directory tree
            ui_components['directory_tree'].value = get_directory_tree(base_dir)
            
        except Exception as e:
            print(f"❌ Error saat setup direktori: {str(e)}")
            ui_components['status_indicator'].value = f"<p>Status: <span style='color: red'>❌ Error: {str(e)}</span></p>"
        
        # Re-enable tombol
        ui_components['setup_button'].disabled = False

def on_drive_checkbox_changed(change, ui_components):
    """
    Handler untuk perubahan checkbox Google Drive.
    
    Args:
        change: Change event
        ui_components: Dictionary berisi komponen UI
    """
    if change['new']:  # Jika checkbox dinyalakan
        # Cek apakah berjalan di Colab
        if is_colab():
            ui_components['drive_path_text'].disabled = False
        else:
            ui_components['drive_checkbox'].value = False
            with ui_components['output_area']:
                clear_output()
                print("⚠️ Google Drive hanya tersedia di Google Colab")
    else:
        ui_components['drive_path_text'].disabled = True

def setup_directory_handlers(ui_components):
    """
    Setup handler untuk UI komponen directory management.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary berisi komponen UI yang telah disetup handler-nya
    """
    # Setup initial state
    is_running_in_colab = is_colab()
    
    if not is_running_in_colab:
        ui_components['drive_checkbox'].value = False
        ui_components['drive_checkbox'].disabled = True
        ui_components['drive_path_text'].disabled = True
        ui_components['status_indicator'].value = "<p>Status: <span style='color: orange'>⚠️ Google Drive tidak tersedia (tidak di Colab)</span></p>"
    
    # Setup event handlers
    ui_components['drive_checkbox'].observe(
        lambda change: on_drive_checkbox_changed(change, ui_components),
        names='value'
    )
    
    ui_components['setup_button'].on_click(
        lambda b: on_setup_button_clicked(b, ui_components)
    )
    
    return ui_components