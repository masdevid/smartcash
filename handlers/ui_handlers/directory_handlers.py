"""
File: smartcash/handlers/ui_handlers/directory_handlers.py
Author: Alfrida Sabar
Deskripsi: Handler untuk UI komponen setup dan manajemen direktori project, 
           menangani integrasi Google Drive dan struktur direktori.
"""

import os
import sys
import shutil
import subprocess
from IPython.display import clear_output, HTML, display
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

def setup_google_drive(mount_point: str = '/content/drive') -> Tuple[bool, str]:
    """
    Mount Google Drive dan verifikasi akses.
    
    Args:
        mount_point: Path untuk mount Google Drive
        
    Returns:
        Tuple berisi (success, message)
    """
    try:
        from google.colab import drive
        
        # Check if already mounted
        if os.path.exists(f"{mount_point}/MyDrive"):
            return True, f"✅ Google Drive sudah terpasang di {mount_point}"
        
        # Mount drive
        drive.mount(mount_point)
        
        # Verify mount
        if os.path.exists(f"{mount_point}/MyDrive"):
            return True, f"✅ Google Drive berhasil dipasang di {mount_point}"
        else:
            return False, "❌ Gagal memverifikasi pemasangan Google Drive"
    except ImportError:
        return False, "⚠️ Google Drive hanya dapat dipasang di Google Colab"
    except Exception as e:
        return False, f"❌ Error saat memasang Google Drive: {str(e)}"

def create_directory_structure(base_dir: str, drive_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Buat struktur direktori project di direktori target.
    
    Args:
        base_dir: Path direktori dasar project
        drive_path: Optional path ke Google Drive
        
    Returns:
        Dictionary dengan statistik direktori yang dibuat
    """
    # Define target directory (either local or Drive)
    target_dir = drive_path if drive_path else base_dir
    
    # Define directories to create
    dirs = [
        "data/train/images",
        "data/train/labels",
        "data/valid/images",
        "data/valid/labels",
        "data/test/images",
        "data/test/labels",
        "runs/train/weights",
        "runs/evaluation",
        "configs",
        "pretrained",
        "logs"
    ]
    
    # Create directories
    results = {
        "created": 0,
        "exists": 0,
        "error": 0,
        "directories": []
    }
    
    for d in dirs:
        full_path = os.path.join(target_dir, d)
        try:
            if not os.path.exists(full_path):
                os.makedirs(full_path, exist_ok=True)
                results["created"] += 1
                results["directories"].append({"path": full_path, "status": "created"})
            else:
                results["exists"] += 1
                results["directories"].append({"path": full_path, "status": "exists"})
        except Exception as e:
            results["error"] += 1
            results["directories"].append({"path": full_path, "status": f"error: {str(e)}"})
    
    return results

def create_symlinks(base_dir: str, drive_path: str) -> Dict[str, Any]:
    """
    Buat symlinks dari direktori lokal ke direktori Google Drive.
    
    Args:
        base_dir: Path direktori lokal
        drive_path: Path direktori di Google Drive
        
    Returns:
        Dictionary dengan statistik symlinks yang dibuat
    """
    # Define directories to symlink
    symlinks = [
        {"name": "data", "src": os.path.join(base_dir, "data"), "dst": os.path.join(drive_path, "data")},
        {"name": "runs", "src": os.path.join(base_dir, "runs"), "dst": os.path.join(drive_path, "runs")},
        {"name": "configs", "src": os.path.join(base_dir, "configs"), "dst": os.path.join(drive_path, "configs")},
        {"name": "pretrained", "src": os.path.join(base_dir, "pretrained"), "dst": os.path.join(drive_path, "pretrained")}
    ]
    
    results = {
        "created": 0,
        "exists": 0,
        "error": 0,
        "symlinks": []
    }
    
    for link in symlinks:
        try:
            # Ensure target directory exists
            if not os.path.exists(link["dst"]):
                os.makedirs(link["dst"], exist_ok=True)
            
            # Remove source if it exists but is not a symlink
            if os.path.exists(link["src"]) and not os.path.islink(link["src"]):
                if os.path.isdir(link["src"]):
                    shutil.rmtree(link["src"])
                else:
                    os.remove(link["src"])
            
            # Create symlink if it doesn't exist
            if not os.path.exists(link["src"]):
                os.symlink(link["dst"], link["src"])
                results["created"] += 1
                results["symlinks"].append({"name": link["name"], "status": "created"})
            else:
                results["exists"] += 1
                results["symlinks"].append({"name": link["name"], "status": "exists"})
        except Exception as e:
            results["error"] += 1
            results["symlinks"].append({"name": link["name"], "status": f"error: {str(e)}"})
    
    return results

def get_directory_tree(base_dir: str, max_depth: int = 3) -> str:
    """
    Generate HTML representation of directory tree.
    
    Args:
        base_dir: Path ke direktori dasar
        max_depth: Kedalaman maksimum untuk ditampilkan
        
    Returns:
        HTML string dari directory tree
    """
    def _tree_html(path, prefix="", is_last=True, depth=0):
        if depth > max_depth:
            return ""
            
        name = os.path.basename(path)
        html = ""
        
        # Add directory name
        if depth > 0:
            html += f"{prefix}{'└── ' if is_last else '├── '}<span style='color: #3498db;'>{name}</span><br/>"
            prefix += '    ' if is_last else '│   '
        else:
            html += f"<span style='color: #2980b9; font-weight: bold;'>{name}</span><br/>"
        
        # List contents
        try:
            entries = sorted([e for e in os.listdir(path) if not e.startswith('.')])
            
            # Limit number of entries shown
            if len(entries) > 10 and depth > 1:
                entries = entries[:10]
                has_more = True
            else:
                has_more = False
                
            for i, entry in enumerate(entries):
                entry_path = os.path.join(path, entry)
                is_entry_last = i == len(entries) - 1 and not has_more
                
                if os.path.isdir(entry_path):
                    html += _tree_html(entry_path, prefix, is_entry_last, depth + 1)
                elif depth < max_depth:
                    html += f"{prefix}{'└── ' if is_entry_last else '├── '}{entry}<br/>"
            
            if has_more:
                html += f"{prefix}└── <i>... dan item lainnya</i><br/>"
        except Exception as e:
            html += f"{prefix}└── <i>Error saat membaca direktori: {str(e)}</i><br/>"
        
        return html
    
    html = "<div style='font-family: monospace; line-height: 1.5;'>"
    html += _tree_html(base_dir)
    html += "</div>"
    
    return html

def on_setup_button_clicked(ui_components: Dict[str, Any], logger: Optional[Any] = None) -> None:
    """
    Handler untuk tombol setup direktori.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_directory_ui()
        logger: Optional logger untuk mencatat aktivitas
    """
    # Validate required components
    required_components = ['setup_button', 'output_area', 'drive_checkbox', 
                         'drive_path_text', 'status_indicator', 'directory_tree']
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    if missing_components:
        if logger:
            logger.error(f"❌ Missing UI components: {', '.join(missing_components)}")
        return
    
    # Disable button during setup
    ui_components['setup_button'].disabled = True
    ui_components['setup_button'].description = "Setting up..."
    
    with ui_components['output_area']:
        clear_output()
        
        try:
            # Get parameters
            use_drive = ui_components['drive_checkbox'].value
            drive_path = ui_components['drive_path_text'].value if use_drive else None
            base_dir = os.getcwd()
            
            # Mount Google Drive if needed
            if use_drive:
                print("🔄 Mencoba memasang Google Drive...")
                success, message = setup_google_drive()
                print(message)
                
                if not success:
                    ui_components['status_indicator'].value = "<p>Status: <span style='color: red'>Setup Gagal - Google Drive tidak tersedia</span></p>"
                    ui_components['setup_button'].disabled = False
                    ui_components['setup_button'].description = "Setup Direktori"
                    return
            
            # Create directory structure
            print(f"\n🔄 Membuat struktur direktori di {'Google Drive' if use_drive else 'direktori lokal'}...")
            dir_results = create_directory_structure(base_dir, drive_path if use_drive else None)
            
            print(f"✅ {dir_results['created']} direktori dibuat")
            print(f"ℹ️ {dir_results['exists']} direktori sudah ada")
            if dir_results['error'] > 0:
                print(f"⚠️ {dir_results['error']} error saat membuat direktori")
            
            # Create symlinks if using Drive
            if use_drive:
                print("\n🔄 Membuat symlinks ke Google Drive...")
                symlink_results = create_symlinks(base_dir, drive_path)
                
                print(f"✅ {symlink_results['created']} symlinks dibuat")
                print(f"ℹ️ {symlink_results['exists']} symlinks sudah ada")
                if symlink_results['error'] > 0:
                    print(f"⚠️ {symlink_results['error']} error saat membuat symlinks")
            
            # Update status and directory tree
            ui_components['status_indicator'].value = "<p>Status: <span style='color: green'>Setup Berhasil</span></p>"
            ui_components['directory_tree'].value = get_directory_tree(base_dir)
            
            print("\n✅ Setup direktori selesai!")
            if use_drive:
                print("📂 Data akan disimpan di Google Drive untuk persistensi antar sesi")
            
        except Exception as e:
            print(f"❌ Error saat setup direktori: {str(e)}")
            ui_components['status_indicator'].value = f"<p>Status: <span style='color: red'>Error - {str(e)}</span></p>"
            import traceback
            traceback.print_exc()
        
        finally:
            # Re-enable button
            ui_components['setup_button'].disabled = False
            ui_components['setup_button'].description = "Setup Direktori"

def on_drive_checkbox_changed(change: Dict[str, Any], ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk perubahan checkbox penggunaan Google Drive.
    
    Args:
        change: Change event dari observe
        ui_components: Dictionary berisi komponen UI dari create_directory_ui()
    """
    if 'drive_path_text' not in ui_components:
        return
        
    # Enable/disable drive path text based on checkbox
    ui_components['drive_path_text'].disabled = not change['new']

def setup_directory_handlers(ui_components: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    Setup semua event handler untuk UI direktori.
    
    Args:
        ui_components: Dictionary berisi komponen UI dari create_directory_ui()
        logger: Optional logger untuk mencatat aktivitas
        
    Returns:
        Dictionary updated UI components dengan handlers yang sudah di-attach
    """
    # Validate required components
    required_components = ['setup_button', 'drive_checkbox', 'directory_tree']
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        if logger:
            logger.error(f"❌ Missing required UI components: {', '.join(missing_components)}")
        return ui_components
    
    # Setup handler untuk tombol setup
    ui_components['setup_button'].on_click(
        lambda b: on_setup_button_clicked(ui_components, logger)
    )
    
    # Setup handler untuk checkbox Drive
    ui_components['drive_checkbox'].observe(
        lambda change: on_drive_checkbox_changed(change, ui_components),
        names='value'
    )
    
    # Initialize directory tree
    base_dir = os.getcwd()
    if os.path.exists(base_dir):
        ui_components['directory_tree'].value = get_directory_tree(base_dir)
    
    # Check if in Google Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    # Disable drive checkbox if not in Colab
    if not is_colab:
        ui_components['drive_checkbox'].disabled = True
        ui_components['drive_checkbox'].value = False
        ui_components['drive_path_text'].disabled = True
        if 'status_indicator' in ui_components:
            ui_components['status_indicator'].value = "<p>Status: <span style='color: orange'>Google Drive hanya tersedia di Colab</span></p>"
    
    return ui_components