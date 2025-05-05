"""
File: smartcash/ui/utils/file_utils.py
Deskripsi: Utilitas file untuk komponen UI dengan operasi general dan visualisasi, tanpa ThreadPool
"""

import ipywidgets as widgets
from IPython.display import display, HTML
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from datetime import datetime
import numpy as np

from smartcash.common.io import format_size
from smartcash.ui.utils.constants import COLORS, ICONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def display_file_info(file_path: str, description: Optional[str] = None) -> widgets.HTML:
    """
    Tampilkan informasi file dalam box informatif.
    
    Args:
        file_path: Path ke file
        description: Deskripsi opsional
        
    Returns:
        Widget HTML berisi info file
    """
    # Ambil info file
    path = Path(file_path)
    if path.exists():
        file_size = path.stat().st_size
        file_time = path.stat().st_mtime
        
        # Format ukuran file
        size_str = format_size(file_size)
        
        # Format waktu
        time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # Buat HTML
        html = f"""
        <div style="padding: 10px; background-color: {COLORS['light']}; border-radius: 5px; margin: 10px 0;">
            <p><strong>{ICONS['file']} File:</strong> {path.name}</p>
            <p><strong>{ICONS['folder']} Path:</strong> {path.parent}</p>
            <p><strong>📏 Size:</strong> {size_str}</p>
            <p><strong>{ICONS['time']} Modified:</strong> {time_str}</p>
        """
        
        if description:
            html += f"<p><strong>📝 Description:</strong> {description}</p>"
        
        html += "</div>"
        
        return widgets.HTML(value=html)
    else:
        return widgets.HTML(value=f"<p>{ICONS['warning']} File tidak ditemukan: {file_path}</p>")


def shorten_filename(filename: str, max_length: int = 15) -> str:
    """
    Persingkat nama file dengan ellipsis untuk tampilan yang lebih baik.
    
    Args:
        filename: Nama file yang akan dipersingkat
        max_length: Panjang maksimum nama file
        
    Returns:
        Nama file yang telah dipersingkat
    """
    if len(filename) <= max_length:
        return filename
    
    # Potong nama file dengan format "awal...akhir"
    prefix_len = max_length // 2 - 1
    suffix_len = max_length - prefix_len - 3  # 3 untuk "..."
    
    return f"{filename[:prefix_len]}...{filename[-suffix_len:]}"


def directory_tree(
    root_dir: Union[str, Path], 
    max_depth: int = 2, 
    exclude_patterns: List[str] = None,
    include_only: List[str] = None
) -> str:
    """
    Buat representasi tree directory dalam HTML.
    
    Args:
        root_dir: Direktori root
        max_depth: Kedalaman maksimum
        exclude_patterns: Pattern file/direktori yang diabaikan
        include_only: Hanya tampilkan file dengan pattern ini
        
    Returns:
        String HTML berisi tree direktori
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return f"<span style='color:{COLORS['danger']}'>❌ Directory not found: {root_dir}</span>"
    
    import re
    
    def match_pattern(path: str, patterns: List[str]) -> bool:
        """Check if path matches any of the patterns."""
        for pattern in patterns:
            if re.search(pattern, path):
                return True
        return False
    
    result = "<pre style='margin:0;padding:5px;background:#f8f9fa;font-family:monospace;color:#333'>\n"
    result += f"<span style='color:{COLORS['primary']};font-weight:bold'>{root_dir.name}/</span>\n"
    
    def traverse_dir(path: Path, prefix: str = "", depth: int = 0) -> str:
        if depth > max_depth: 
            return ""
            
        items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name))
        tree = ""
        
        for i, item in enumerate(items):
            # Check exclusion patterns
            if exclude_patterns and match_pattern(str(item), exclude_patterns):
                continue
                
            # Check inclusion patterns
            if include_only and not item.is_dir() and not match_pattern(str(item), include_only):
                continue
                
            is_last = i == len(items) - 1
            connector = "└─ " if is_last else "├─ "
            
            if item.is_dir():
                tree += f"{prefix}{connector}<span style='color:{COLORS['primary']};font-weight:bold'>{item.name}/</span>\n"
                next_prefix = prefix + ("   " if is_last else "│  ")
                if depth < max_depth:
                    tree += traverse_dir(item, next_prefix, depth + 1)
            else:
                tree += f"{prefix}{connector}{item.name}\n"
        
        return tree
    
    result += traverse_dir(root_dir)
    result += "</pre>"
    return result

def create_file_upload_widget(
    accept: str = '.zip,.csv,.txt',
    multiple: bool = False,
    description: str = "Upload File:",
    callback: Optional[Callable] = None
) -> widgets.FileUpload:
    """
    Buat widget upload file dengan callback.
    
    Args:
        accept: MIME types yang diterima
        multiple: Izinkan upload multiple files
        description: Deskripsi widget
        callback: Callback yang dipanggil setelah upload
        
    Returns:
        Widget FileUpload
    """
    upload = widgets.FileUpload(
        accept=accept,
        multiple=multiple,
        description=description
    )
    
    if callback:
        upload.observe(callback, names='value')
    
    return upload

def save_uploaded_file(
    file_content: bytes, 
    file_name: str, 
    target_dir: str = "uploads",
    create_dirs: bool = True
) -> Tuple[bool, str]:
    """
    Simpan file yang diupload dari FileUpload widget.
    
    Args:
        file_content: Konten file dalam bytes
        file_name: Nama file
        target_dir: Direktori tujuan
        create_dirs: Buat direktori jika belum ada
        
    Returns:
        Tuple (success, message)
    """
    # Create directory if needed
    if create_dirs:
        os.makedirs(target_dir, exist_ok=True)
        
    target_path = os.path.join(target_dir, file_name)
    
    try:
        with open(target_path, 'wb') as f:
            f.write(file_content)
        return True, f"File berhasil disimpan ke {target_path}"
    except Exception as e:
        return False, f"Error saat menyimpan file: {str(e)}"

def create_file_browser(
    root_dir: str = ".", 
    file_extensions: List[str] = None,
    on_select: Optional[Callable] = None
) -> widgets.VBox:
    """
    Buat file browser sederhana dengan sistem direktori.
    
    Args:
        root_dir: Direktori root
        file_extensions: Filter ekstensi file
        on_select: Callback saat file dipilih
        
    Returns:
        Widget VBox berisi file browser
    """
    # Convert extensions to lowercase
    if file_extensions:
        file_extensions = [ext.lower().lstrip('.') for ext in file_extensions]
    
    # Current path state
    current_path = [os.path.abspath(root_dir)]
    
    # Create title
    title = widgets.HTML(f"<h3>{ICONS['folder']} File Browser</h3>")
    
    # Create path display
    path_display = widgets.Text(
        value=current_path[0],
        description='Path:',
        disabled=True,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Create file list output
    file_list = widgets.Select(
        options=[],
        description='Files:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='200px', margin='5px 0')
    )
    
    # Create navigation buttons
    back_button = widgets.Button(
        description='Up',
        icon='arrow-up',
        disabled=False,
        layout=widgets.Layout(width='auto')
    )
    
    refresh_button = widgets.Button(
        description='Refresh',
        icon='refresh',
        layout=widgets.Layout(width='auto')
    )
    
    # Status display
    status = widgets.HTML("")
    
    # Update file list
    def update_file_list():
        current = current_path[0]
        try:
            # Get directories and files
            items = []
            
            for item in sorted(os.listdir(current)):
                full_path = os.path.join(current, item)
                if os.path.isdir(full_path):
                    items.append(f"📁 {item}")
                else:
                    # Check extensions if specified
                    if file_extensions:
                        ext = os.path.splitext(item)[1].lstrip('.').lower()
                        if ext in file_extensions:
                            items.append(f"📄 {item}")
                    else:
                        items.append(f"📄 {item}")
            
            file_list.options = items
            path_display.value = current
            status.value = f"<span style='color:{COLORS['success']}'>✅ Loaded {len(items)} items</span>"
        except Exception as e:
            status.value = f"<span style='color:{COLORS['danger']}'>❌ Error: {str(e)}</span>"
    
    # Handle back button
    def on_back_click(b):
        parent = os.path.dirname(current_path[0])
        if parent != current_path[0]:  # Not at root
            current_path[0] = parent
            update_file_list()
    
    # Handle refresh button
    def on_refresh_click(b):
        update_file_list()
    
    # Handle file selection
    def on_file_select(change):
        if change['type'] == 'change' and change['name'] == 'value':
            selected = change['new']
            if selected:
                if selected.startswith('📁 '):  # It's a directory
                    dir_name = selected[2:]  # Remove icon
                    new_path = os.path.join(current_path[0], dir_name)
                    current_path[0] = new_path
                    update_file_list()
                elif on_select and selected.startswith('📄 '):  # It's a file
                    file_name = selected[2:]  # Remove icon
                    file_path = os.path.join(current_path[0], file_name)
                    on_select(file_path)
    
    # Register event handlers
    back_button.on_click(on_back_click)
    refresh_button.on_click(on_refresh_click)
    file_list.observe(on_file_select)
    
    # Create browser layout
    browser = widgets.VBox([
        title,
        path_display,
        widgets.HBox([back_button, refresh_button]),
        file_list,
        status
    ])
    
    # Initialize file list
    update_file_list()
    
    return browser

def backup_file(
    file_path: str, 
    backup_dir: Optional[str] = None,
    timestamp: bool = True
) -> Tuple[bool, str]:
    """
    Buat backup file dengan timestamp.
    
    Args:
        file_path: Path ke file yang akan dibackup
        backup_dir: Direktori backup (default: <file_dir>/backup)
        timestamp: Tambahkan timestamp ke nama file
        
    Returns:
        Tuple (success, backup_path or error message)
    """
    try:
        source_path = Path(file_path)
        if not source_path.exists():
            return False, f"File tidak ditemukan: {file_path}"
        
        # Create backup dir
        if backup_dir:
            backup_path = Path(backup_dir)
        else:
            backup_path = source_path.parent / "backup"
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create destination filename
        if timestamp:
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{source_path.stem}_{ts}{source_path.suffix}"
        else:
            backup_filename = source_path.name
        
        # Create full backup path
        backup_file = backup_path / backup_filename
        
        # Copy file
        shutil.copy2(source_path, backup_file)
        
        return True, str(backup_file)
    except Exception as e:
        return False, f"Error saat backup file: {str(e)}"

def list_files(
    directory: str,
    pattern: Optional[str] = None,
    recursive: bool = False,
    include_dirs: bool = False,
    sort_by: str = 'name'  # 'name', 'date', 'size'
) -> List[str]:
    """
    Daftar file dalam direktori dengan berbagai opsi filter.
    
    Args:
        directory: Direktori yang akan dicari
        pattern: Pattern nama file (glob pattern)
        recursive: Cari di subdirektori
        include_dirs: Sertakan direktori dalam hasil
        sort_by: Kriteria pengurutan
        
    Returns:
        List path file
    """
    import glob
    
    path = Path(directory)
    
    # Build glob pattern
    if pattern:
        glob_pattern = pattern
    else:
        glob_pattern = '*'
        
    if recursive:
        glob_pattern = f"**/{glob_pattern}"
    
    # Get file list
    file_paths = []
    for item in path.glob(glob_pattern):
        if item.is_file() or (include_dirs and item.is_dir()):
            file_paths.append(str(item))
    
    # Sort results
    if sort_by == 'date':
        file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    elif sort_by == 'size':
        file_paths.sort(key=lambda x: os.path.getsize(x), reverse=True)
    else:  # Default: sort by name
        file_paths.sort()
    
    return file_paths

def is_image_file(file_path: str) -> bool:
    """
    Cek apakah file adalah gambar berdasarkan ekstensi.
    
    Args:
        file_path: Path ke file
        
    Returns:
        Boolean menunjukkan apakah file adalah gambar
    """
    ext = os.path.splitext(file_path)[1].lstrip('.').lower()
    return ext in IMAGE_EXTENSIONS

def is_video_file(file_path: str) -> bool:
    """
    Cek apakah file adalah video berdasarkan ekstensi.
    
    Args:
        file_path: Path ke file
        
    Returns:
        Boolean menunjukkan apakah file adalah video
    """
    ext = os.path.splitext(file_path)[1].lstrip('.').lower()
    return ext in VIDEO_EXTENSIONS

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Dapatkan informasi lengkap tentang file.
    
    Args:
        file_path: Path ke file
        
    Returns:
        Dictionary berisi informasi file
    """
    path = Path(file_path)
    if not path.exists():
        return {'error': f"File tidak ditemukan: {file_path}"}
    
    try:
        stat = path.stat()
        file_size = stat.st_size
        
        info = {
            'name': path.name,
            'path': str(path),
            'parent': str(path.parent),
            'stem': path.stem,
            'suffix': path.suffix,
            'size': file_size,
            'size_formatted': format_size(file_size),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'is_image': is_image_file(file_path),
            'is_video': is_video_file(file_path),
        }
        
        return info
    except Exception as e:
        return {'error': f"Error mendapatkan info file: {str(e)}"}

def create_file_download_link(file_path: str, link_text: Optional[str] = None) -> widgets.HTML:
    """
    Buat link download untuk file.
    
    Args:
        file_path: Path ke file
        link_text: Teks untuk link (default: nama file)
        
    Returns:
        Widget HTML berisi link download
    """
    try:
        import base64
        
        path = Path(file_path)
        if not path.exists():
            return widgets.HTML(f"<p style='color:{COLORS['danger']}'>❌ File tidak ditemukan: {file_path}</p>")
        
        # Read file
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Encode to base64
        b64 = base64.b64encode(content).decode()
        
        # Determine MIME type
        import mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        # Create download link
        text = link_text or path.name
        href = f"data:{mime_type};base64,{b64}"
        link = f"""
        <a href="{href}" download="{path.name}" 
           style="padding: 8px 12px; background-color: {COLORS['primary']}; color: white; 
                 text-decoration: none; border-radius: 4px; display: inline-block;">
           {ICONS['download']} {text}
        </a>
        """
        
        return widgets.HTML(link)
    except Exception as e:
        return widgets.HTML(f"<p style='color:{COLORS['danger']}'>❌ Error membuat link download: {str(e)}</p>")


def find_label_path(img_path: Path) -> Optional[Path]:
    """
    Fungsi helper untuk mencari label path dari image path.
    
    Args:
        img_path: Path gambar
        
    Returns:
        Path file label atau None jika tidak ditemukan
    """
    # Cek apakah ada file label di folder paralel
    parent_dir = img_path.parent.parent
    label_path = parent_dir / 'labels' / f"{img_path.stem}.txt"
    
    if label_path.exists():
        return label_path
    
    # Cek apakah ada file label di folder sibling
    sibling_label_dir = img_path.parent.parent / 'labels'
    if sibling_label_dir.exists():
        sibling_label_path = sibling_label_dir / f"{img_path.stem}.txt"
        if sibling_label_path.exists():
            return sibling_label_path
    
    return None

def load_image(img_path: Path) -> np.ndarray:
    import cv2

    """Fungsi helper untuk loading gambar dengan berbagai format."""
    if str(img_path).endswith('.npy'):
        # Handle numpy array
        img = np.load(str(img_path))
        # Denormalisasi jika perlu
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
    else:
        # Handle gambar biasa
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_matching_pairs(
    augmented_files: List[Path], 
    original_files: List[Path], 
    orig_prefix: str = "rp",
    aug_prefix: str = "aug"
) -> List[Tuple[Path, Path]]:
    """
    Temukan pasangan file augmentasi dan original berdasarkan uuid dalam nama file.
    
    Args:
        augmented_files: List path file augmentasi
        original_files: List path file original
        orig_prefix: Prefix untuk file original
        aug_prefix: Prefix untuk file augmentasi
        
    Returns:
        List tuple (original_path, augmented_path)
    """
    # Siapkan mapping original filename -> path
    original_map = {f.name: f for f in original_files}
    
    # Cari pasangan yang cocok
    matched_pairs = []
    
    for aug_file in augmented_files:
        # Dapatkan nama file original yang sesuai
        orig_filename = get_original_from_augmented(aug_file.name, orig_prefix)
        
        if orig_filename and orig_filename in original_map:
            orig_file = original_map[orig_filename]
            matched_pairs.append((orig_file, aug_file))
    
    return matched_pairs


def get_original_from_augmented(aug_filename: str, orig_prefix: str = "rp") -> Optional[str]:
    """
    Ekstrak nama file original dari nama file augmentasi.
    Format nama file augmentasi: [augmented_prefix]_[source_prefix]_[class_name]_[uuid]_var[n]
    
    Args:
        aug_filename: Nama file augmentasi
        orig_prefix: Prefix untuk file original
        
    Returns:
        Nama file original yang sesuai atau None jika tidak ditemukan
    """
    # Pattern untuk mendeteksi bagian unik dari nama file
    # Format: aug_rp_class_uuid_var1.jpg -> rp_class_uuid.jpg
    pattern = r'(?:[^_]+)_(' + re.escape(orig_prefix) + r'_[^_]+_[^_]+)_var\d+'
    match = re.search(pattern, aug_filename)
    
    if match:
        # Ekstrak bagian yang cocok dengan original
        original_part = match.group(1)
        return f"{original_part}.jpg"
    
    return None