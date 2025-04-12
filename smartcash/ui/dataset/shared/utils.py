"""
File: smartcash/ui/dataset/shared/utils.py
Deskripsi: Utilitas bersama untuk modul dataset preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, Callable, Tuple, List
import os
import shutil
import time
import re
from pathlib import Path
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.constants import ICONS, COLORS

def get_file_count(directory: str, pattern: str = "*.jpg") -> int:
    """
    Hitung jumlah file dengan pattern tertentu dalam direktori.
    
    Args:
        directory: Path direktori
        pattern: Pattern file untuk dihitung
        
    Returns:
        Jumlah file yang ditemukan
    """
    path = Path(directory)
    if not path.exists():
        return 0
        
    return len(list(path.glob(pattern)))

def collect_dataset_stats(directory: str, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Kumpulkan statistik dataset dari direktori.
    
    Args:
        directory: Path direktori dataset
        prefix: Prefix file yang dicari (opsional)
        
    Returns:
        Dictionary berisi statistik dataset
    """
    stats = {
        "total_images": 0,
        "total_labels": 0,
        "splits": {},
        "classes": {}
    }
    
    # Validasi direktori
    path = Path(directory)
    if not path.exists():
        return stats
    
    # Daftar split yang mungkin
    splits = [d.name for d in path.iterdir() if d.is_dir() and d.name in ['train', 'valid', 'test']]
    
    # Hitung file per split
    for split in splits:
        split_dir = path / split
        
        # Cek direktori images
        images_dir = split_dir / 'images'
        if images_dir.exists():
            # Hitung semua file gambar atau dengan prefix tertentu
            if prefix:
                images = list(images_dir.glob(f"{prefix}*.jpg")) + list(images_dir.glob(f"{prefix}*.png"))
            else:
                images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            # Cek direktori labels
            labels_dir = split_dir / 'labels'
            if labels_dir.exists():
                # Hitung semua file label atau dengan prefix tertentu
                if prefix:
                    labels = list(labels_dir.glob(f"{prefix}*.txt"))
                else:
                    labels = list(labels_dir.glob("*.txt"))
                
                # Update statistik
                stats["splits"][split] = {
                    "images": len(images),
                    "labels": len(labels)
                }
                stats["total_images"] += len(images)
                stats["total_labels"] += len(labels)
                
                # Analisis distribusi kelas
                if not labels:
                    continue
                    
                # Analisis 100 label secara acak untuk efisiensi
                import random
                sample_labels = random.sample(labels, min(100, len(labels)))
                
                for label_file in sample_labels:
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                parts = line.strip().split()
                                if parts:
                                    class_id = parts[0]
                                    if class_id in stats["classes"]:
                                        stats["classes"][class_id] += 1
                                    else:
                                        stats["classes"][class_id] = 1
                    except Exception:
                        continue
    
    return stats

def format_time(seconds: float) -> str:
    """
    Format waktu dari detik ke format yang lebih readable.
    
    Args:
        seconds: Jumlah detik
        
    Returns:
        String waktu terformat
    """
    if seconds < 60:
        return f"{seconds:.2f} detik"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} menit {secs:.2f} detik"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} jam {minutes} menit {secs:.2f} detik"

def ensure_directory(directory: str) -> bool:
    """
    Pastikan direktori ada, buat jika belum ada.
    
    Args:
        directory: Path direktori
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception:
        return False

def find_matching_pairs(original_dir: str, augmented_dir: str, 
                      orig_prefix: str = "rp", aug_prefix: str = "aug") -> List[Tuple[Path, Path]]:
    """
    Temukan pasangan file original dan augmented berdasarkan UUID.
    
    Args:
        original_dir: Direktori file original
        augmented_dir: Direktori file augmented
        orig_prefix: Prefix untuk file original
        aug_prefix: Prefix untuk file augmented
        
    Returns:
        List tuple (original_path, augmented_path)
    """
    # Path untuk direktori
    orig_path = Path(original_dir)
    aug_path = Path(augmented_dir)
    
    if not orig_path.exists() or not aug_path.exists():
        return []
    
    # Pattern untuk ekstrak UUID
    # Format: aug_rp_class_uuid_var1.jpg
    uuid_pattern = re.compile(f"{aug_prefix}_{orig_prefix}_([^_]+)_([^_]+)_var\\d+")
    
    # Dapatkan semua file augmentasi
    aug_files = list(aug_path.glob(f"{aug_prefix}_*.jpg")) + list(aug_path.glob(f"{aug_prefix}_*.png"))
    
    # Dapatkan semua file original
    orig_files = list(orig_path.glob(f"{orig_prefix}_*.jpg")) + list(orig_path.glob(f"{orig_prefix}_*.png"))
    
    # Map file original berdasarkan nama
    orig_map = {f.name: f for f in orig_files}
    
    # Temukan pasangan
    pairs = []
    
    for aug_file in aug_files:
        # Ekstrak UUID
        match = uuid_pattern.match(aug_file.name)
        if match:
            class_name, uuid = match.groups()
            # Cari file original yang sesuai
            orig_name = f"{orig_prefix}_{class_name}_{uuid}.jpg"
            if orig_name in orig_map:
                pairs.append((orig_map[orig_name], aug_file))
    
    return pairs

def create_summary_widget(summary: Dict[str, Any], module_type: str = "preprocessing") -> widgets.HTML:
    """
    Buat widget summary untuk hasil processing.
    
    Args:
        summary: Dictionary berisi ringkasan hasil
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Widget HTML berisi ringkasan
    """
    # Tentukan judul berdasarkan module_type
    title = "Ringkasan Preprocessing" if module_type == "preprocessing" else "Ringkasan Augmentasi"
    
    # Tentukan konten berdasarkan module_type
    if module_type == "preprocessing":
        # Format untuk preprocessing
        content = f"""
        <div style="padding: 15px; background-color: {COLORS['light']}; border-radius: 5px;">
            <h3 style="color: {COLORS['dark']};">{ICONS.get('stats', '📊')} {title}</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 15px;">
                <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                    <div style="color: {COLORS['muted']};">Total Gambar</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['primary']};">{summary.get('total_images', 0)}</div>
                </div>
                <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                    <div style="color: {COLORS['muted']};">Total Label</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['primary']};">{summary.get('total_labels', 0)}</div>
                </div>
        """
        
        # Tambahkan info resolusi jika tersedia
        if 'image_size' in summary:
            img_size = summary['image_size']
            if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
                content += f"""
                <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                    <div style="color: {COLORS['muted']};">Resolusi</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['primary']};">{img_size[0]}x{img_size[1]}</div>
                </div>
                """
        
        # Tambahkan waktu proses jika tersedia
        if 'processing_time' in summary:
            content += f"""
            <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                <div style="color: {COLORS['muted']};">Waktu Proses</div>
                <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['primary']};">{format_time(summary['processing_time'])}</div>
            </div>
            """
            
        # Tutup div wrapper
        content += """
            </div>
        """
        
        # Tambahkan info split jika tersedia
        if 'split_stats' in summary:
            content += """
            <h4 style="color: #495057;">Statistik per Split</h4>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 15px;">
                <thead>
                    <tr style="background-color: #f8f9fa;">
                        <th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Split</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Gambar</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Label</th>
                        <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Status</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for split, stats in summary['split_stats'].items():
                # Tentukan status
                status_icon = ICONS['success'] if stats.get('images', 0) > 0 and stats.get('labels', 0) > 0 else ICONS['warning']
                status_color = COLORS['success'] if stats.get('images', 0) > 0 and stats.get('labels', 0) > 0 else COLORS['warning']
                status_text = "Lengkap" if stats.get('images', 0) > 0 and stats.get('labels', 0) > 0 else "Tidak Lengkap"
                
                content += f"""
                <tr>
                    <td style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">{split.capitalize()}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{stats.get('images', 0)}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">{stats.get('labels', 0)}</td>
                    <td style="padding: 8px; text-align: center; border: 1px solid #dee2e6; color: {status_color};">
                        {status_icon} {status_text}
                    </td>
                </tr>
                """
            
            content += """
                </tbody>
            </table>
            """
        
        # Tambahkan path output jika tersedia
        if 'output_dir' in summary:
            content += f"""
            <div style="margin-top: 10px; padding: 10px; background-color: white; border-radius: 5px;">
                <strong>Path Output:</strong> {summary['output_dir']}
            </div>
            """
        
        # Tutup div utama
        content += """
        </div>
        """
    else:
        # Format untuk augmentasi
        content = f"""
        <div style="padding: 15px; background-color: {COLORS['light']}; border-radius: 5px;">
            <h3 style="color: {COLORS['dark']};">{ICONS.get('stats', '📊')} {title}</h3>
            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 15px;">
                <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                    <div style="color: {COLORS['muted']};">File Original</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['primary']};">{summary.get('original', 0)}</div>
                </div>
                <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                    <div style="color: {COLORS['muted']};">File Augmentasi</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['success']};">{summary.get('generated', 0)}</div>
                </div>
                <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                    <div style="color: {COLORS['muted']};">Total File</div>
                    <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['primary']};">{summary.get('total_files', 0)}</div>
                </div>
        """
        
        # Tambahkan waktu proses jika tersedia
        if 'duration' in summary:
            content += f"""
            <div style="background-color: white; padding: 10px; border-radius: 5px; min-width: 150px; text-align: center;">
                <div style="color: {COLORS['muted']};">Waktu Proses</div>
                <div style="font-size: 1.5em; font-weight: bold; color: {COLORS['primary']};">{format_time(summary['duration'])}</div>
            </div>
            """
            
        # Tutup div wrapper
        content += """
            </div>
        """
        
        # Tambahkan jenis augmentasi jika tersedia
        if 'augmentation_types' in summary and summary['augmentation_types']:
            aug_types = summary['augmentation_types']
            
            # Map ke nama yang lebih deskriptif
            type_map = {
                'combined': 'Kombinasi', 
                'position': 'Variasi Posisi', 
                'lighting': 'Variasi Pencahayaan', 
                'extreme_rotation': 'Rotasi Ekstrim'
            }
            
            readable_types = [type_map.get(t, t) for t in aug_types]
            
            content += f"""
            <div style="margin-top: 10px; padding: 10px; background-color: white; border-radius: 5px;">
                <strong>Jenis Augmentasi:</strong> {', '.join(readable_types)}
            </div>
            """
        
        # Tambahkan path output jika tersedia
        if 'output_dir' in summary:
            content += f"""
            <div style="margin-top: 10px; padding: 10px; background-color: white; border-radius: 5px;">
                <strong>Path Output:</strong> {summary['output_dir']}
            </div>
            """
        
        # Tutup div utama
        content += """
        </div>
        """
    
    # Buat widget HTML
    return widgets.HTML(content)

def create_progress_widget(description: str = "Progress") -> Tuple[widgets.HBox, Callable]:
    """
    Buat widget progress bar dengan throttled callback.
    
    Args:
        description: Deskripsi progress bar
        
    Returns:
        Tuple (widget, update_function)
    """
    # Progress bar untuk overall progress
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description=description,
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(width='70%')
    )
    
    # Label untuk progress
    progress_label = widgets.Label("0%")
    
    # Container untuk progress bar dan label
    progress_container = widgets.HBox([progress_bar, progress_label])
    
    # Timestamp update terakhir (untuk throttling)
    last_update = {'time': 0}
    
    # Update function dengan throttling
    def update_progress(value, total, message=None):
        # Throttling untuk mencegah update terlalu sering
        current_time = time.time()
        if current_time - last_update['time'] < 0.1:  # Update maksimal 10x/detik
            return
            
        # Normalisasi progress
        progress_value = min(value, total)
        percentage = int(progress_value / total * 100) if total > 0 else 0
        
        # Update progress bar
        progress_bar.max = total
        progress_bar.value = progress_value
        
        # Update label
        progress_label.value = f"{percentage}%"
        
        # Update description jika ada pesan
        if message:
            progress_bar.description = message
            
        # Update timestamp
        last_update['time'] = current_time
    
    return progress_container, update_progress

def run_with_progress(func, args=None, kwargs=None, 
                    progress_widget=None, progress_callback=None,
                    on_complete=None, on_error=None):
    """
    Jalankan fungsi dengan progress tracking.
    
    Args:
        func: Fungsi yang akan dijalankan
        args: Argumen positional untuk fungsi
        kwargs: Argumen keyword untuk fungsi
        progress_widget: Widget progress untuk update
        progress_callback: Callback untuk update progress
        on_complete: Callback saat selesai
        on_error: Callback saat terjadi error
        
    Returns:
        Hasil dari fungsi
    """
    args = args or []
    kwargs = kwargs or {}
    
    def update_progress(value, total, message=None):
        # Update progress melalui widget atau callback
        if progress_widget and hasattr(progress_widget, 'children') and len(progress_widget.children) > 0:
            progress_bar = progress_widget.children[0]
            if hasattr(progress_bar, 'max') and hasattr(progress_bar, 'value'):
                progress_bar.max = total
                progress_bar.value = value
                percentage = int(value / total * 100) if total > 0 else 0
                progress_bar.description = f"{percentage}%"
                
                # Update label jika ada
                if len(progress_widget.children) > 1 and hasattr(progress_widget.children[1], 'value'):
                    progress_widget.children[1].value = f"{percentage}%"
        
        # Update progress melalui callback jika tersedia
        if progress_callback and callable(progress_callback):
            progress_callback(value, total, message)
    
    # Tambahkan progress callback ke kwargs
    kwargs['progress_callback'] = update_progress
    
    # Jalankan fungsi dalam thread terpisah
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        
        try:
            result = future.result()
            
            # Panggil callback complete jika tersedia
            if on_complete and callable(on_complete):
                on_complete(result)
                
            return result
        except Exception as e:
            # Panggil callback error jika tersedia
            if on_error and callable(on_error):
                on_error(e)
                
            raise e

def get_class_distribution(directory: str, split: str = 'train', prefix: Optional[str] = None) -> Dict[str, int]:
    """
    Dapatkan distribusi kelas dari dataset.
    
    Args:
        directory: Path direktori dataset
        split: Split dataset yang dianalisis
        prefix: Prefix file yang dicari (opsional)
        
    Returns:
        Dictionary {class_id: count}
    """
    class_counts = {}
    
    # Path labels
    labels_dir = Path(directory) / split / 'labels'
    if not labels_dir.exists():
        return class_counts
    
    # Semua file label dengan prefix tertentu
    if prefix:
        label_files = list(labels_dir.glob(f"{prefix}*.txt"))
    else:
        label_files = list(labels_dir.glob("*.txt"))
    
    # Analisis semua file label
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = parts[0]
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except Exception:
            continue
    
    return class_counts

def compare_class_distributions(directory: str, split: str = 'train', 
                             orig_prefix: str = 'rp', aug_prefix: str = 'aug') -> Dict[str, Dict[str, int]]:
    """
    Bandingkan distribusi kelas antara dataset original dan augmented.
    
    Args:
        directory: Path direktori dataset
        split: Split yang dianalisis
        orig_prefix: Prefix untuk file original
        aug_prefix: Prefix untuk file augmented
        
    Returns:
        Dictionary {category: {class_id: count}}
    """
    # Dapatkan distribusi kelas untuk masing-masing kategori
    orig_distribution = get_class_distribution(directory, split, orig_prefix)
    aug_distribution = get_class_distribution(directory, split, aug_prefix)
    
    # Gabungkan semua kelas yang unik
    all_classes = set(list(orig_distribution.keys()) + list(aug_distribution.keys()))
    
    # Buat combined distribution dengan struktur lebih mudah untuk visualisasi
    combined = {}
    for cls in all_classes:
        combined[cls] = {
            'original': orig_distribution.get(cls, 0),
            'augmented': aug_distribution.get(cls, 0),
            'total': orig_distribution.get(cls, 0) + aug_distribution.get(cls, 0)
        }
    
    # Tambahkan summary untuk keseluruhan
    result = {
        'by_class': combined,
        'summary': {
            'original': sum(orig_distribution.values()),
            'augmented': sum(aug_distribution.values()),
            'total': sum(orig_distribution.values()) + sum(aug_distribution.values())
        }
    }
    
    return result

def sync_to_drive(src_dir: str, dst_dir: str, logger=None) -> bool:
    """
    Sinkronisasi direktori ke Google Drive.
    
    Args:
        src_dir: Direktori sumber
        dst_dir: Direktori tujuan di Drive
        logger: Logger instance (opsional)
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    try:
        # Deteksi Google Drive
        from smartcash.ui.utils.drive_utils import detect_drive_mount
        is_mounted, drive_path = detect_drive_mount()
        
        if not is_mounted:
            if logger: logger.warning(f"{ICONS['warning']} Google Drive tidak ter-mount")
            return False
        
        # Pastikan direktori tujuan ada
        dst_path = Path(dst_dir)
        dst_path.mkdir(parents=True, exist_ok=True)
        
        # Gunakan sync_directory dari drive_utils jika tersedia
        try:
            from smartcash.ui.utils.drive_utils import sync_directory
            result = sync_directory(src_dir, dst_dir, logger=logger)
            
            if result.get('copied', 0) > 0:
                if logger: logger.success(f"{ICONS['success']} Berhasil menyalin {result['copied']} file ke Drive")
            else:
                if logger: logger.info(f"{ICONS['info']} Tidak ada file yang perlu disalin ke Drive")
                
            return result.get('errors', 0) == 0
        except ImportError:
            # Fallback: salin manual
            import shutil
            from glob import glob
            
            files_copied = 0
            
            # Salin semua file
            for file in glob(os.path.join(src_dir, '**'), recursive=True):
                if os.path.isfile(file):
                    # Tentukan path relatif
                    rel_path = os.path.relpath(file, src_dir)
                    dest_file = os.path.join(dst_dir, rel_path)
                    
                    # Pastikan direktori ada
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    
                    # Salin file
                    shutil.copy2(file, dest_file)
                    files_copied += 1
            
            if logger: logger.success(f"{ICONS['success']} Berhasil menyalin {files_copied} file ke Drive")
            return True
    except Exception as e:
        if logger: logger.error(f"{ICONS['error']} Error saat sinkronisasi ke Drive: {str(e)}")
        return False