"""
File: smartcash/ui/dataset/handlers/zip_handler.py
Deskripsi: Handler untuk ekstraksi dataset ZIP atau arsip lainnya
"""

import os
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

def extract_dataset(file_path: Path, output_path: Path, ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Ekstrak dataset dari file ZIP atau arsip lainnya.
    
    Args:
        file_path: Path file arsip
        output_path: Path direktori output
        ui_components: Dictionary komponen UI
        
    Returns:
        Tuple (success, message)
    """
    logger = ui_components.get('logger')
    
    try:
        # Update progress
        _update_progress(ui_components, 80, f"Mengekstrak dataset dari {file_path.name}...")
        
        # Ekstrak file arsip
        extract_dir = extract_zip_dataset(file_path, ui_components)
        
        if not extract_dir:
            return False, "Gagal mengekstrak file dataset"
        
        # Deteksi format dataset
        _update_progress(ui_components, 90, "Mendeteksi format dataset...")
        dataset_format = _detect_dataset_format(extract_dir)
        
        # Cek apakah perlu reorganisasi file (jika tidak mengikuti struktur standar)
        if dataset_format == "unknown":
            _update_progress(ui_components, 95, "Format dataset tidak dikenal, mencoba reorganisasi...")
            _reorganize_dataset(extract_dir, output_path)
        
        # Hitung statistik
        stats = _count_dataset_files(extract_dir)
        
        # Update progress
        _update_progress(ui_components, 100, "Ekstraksi selesai")
        
        # Format pesan hasil
        result_msg = f"Dataset berhasil diekstrak ke {extract_dir}: {stats['images']} gambar, {stats['labels']} label"
        return True, result_msg
    
    except Exception as e:
        if logger: logger.error(f"❌ Error saat ekstraksi dataset: {str(e)}")
        return False, f"Error saat ekstraksi dataset: {str(e)}"

def extract_zip_dataset(file_path: Path, ui_components: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """
    Ekstrak file ZIP atau arsip dan return path outputnya.
    
    Args:
        file_path: Path file arsip
        ui_components: Dictionary komponen UI (opsional)
        
    Returns:
        Path direktori hasil ekstraksi atau None jika gagal
    """
    try:
        # Determine extract path
        extract_dir = file_path.parent / file_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract berdasarkan tipe file
        if file_path.name.lower().endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Get total files untuk progress tracking
                total_files = len(zip_ref.namelist())
                
                # Ekstrak dengan progress tracking
                if ui_components:
                    _update_progress(ui_components, 80, f"Mengekstrak {total_files} file...")
                    
                    # Ekstrak satu per satu dengan progress
                    for i, filename in enumerate(zip_ref.namelist()):
                        zip_ref.extract(filename, extract_dir)
                        
                        # Update progress setiap 10% file terekstrack
                        if i % max(1, total_files // 10) == 0:
                            progress = 80 + int(10 * i / total_files)
                            _update_progress(ui_components, progress, f"Mengekstrak file {i+1}/{total_files}...")
                else:
                    # Ekstrak semua file langsung
                    zip_ref.extractall(extract_dir)
                    
        elif any(file_path.name.lower().endswith(ext) for ext in ['.tar', '.gz', '.tgz']):
            with tarfile.open(file_path) as tar_ref:
                # Get total files untuk progress tracking
                members = tar_ref.getmembers()
                total_files = len(members)
                
                # Ekstrak dengan progress tracking
                if ui_components:
                    _update_progress(ui_components, 80, f"Mengekstrak {total_files} file...")
                    
                    # Ekstrak satu per satu dengan progress
                    for i, member in enumerate(members):
                        tar_ref.extract(member, extract_dir)
                        
                        # Update progress setiap 10% file terekstrack
                        if i % max(1, total_files // 10) == 0:
                            progress = 80 + int(10 * i / total_files)
                            _update_progress(ui_components, progress, f"Mengekstrak file {i+1}/{total_files}...")
                else:
                    # Ekstrak semua file langsung
                    tar_ref.extractall(extract_dir)
        else:
            # Tipe file tidak didukung
            return None
        
        return extract_dir
    
    except Exception as e:
        # Log error
        logger = ui_components.get('logger') if ui_components else None
        if logger: logger.error(f"❌ Error saat ekstraksi file: {str(e)}")
        return None

def _detect_dataset_format(directory: Path) -> str:
    """
    Deteksi format dataset berdasarkan struktur direktori.
    
    Args:
        directory: Path direktori dataset
        
    Returns:
        String format dataset ("yolo", "coco", "voc", "unknown")
    """
    # Cek struktur YOLO
    yolo_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
    if any((directory / d).exists() for d in yolo_dirs):
        return "yolo"
    
    # Cek struktur COCO
    if (directory / "annotations").exists() and any((directory / split).exists() for split in ["train", "val", "test"]):
        return "coco"
    
    # Cek struktur VOC
    voc_dirs = ["ImageSets", "JPEGImages", "Annotations"]
    if all((directory / d).exists() for d in voc_dirs):
        return "voc"
    
    # Format tidak dikenal
    return "unknown"

def _reorganize_dataset(source_dir: Path, output_dir: Path) -> None:
    """
    Reorganisasi dataset ke struktur YOLO standar.
    
    Args:
        source_dir: Path direktori sumber
        output_dir: Path direktori output
    """
    # Buat struktur direktori YOLO standar
    for split in ['train', 'valid', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(output_dir / split / subdir, exist_ok=True)
    
    # Cari semua file gambar dan label
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    label_ext = '.txt'
    
    # Dapatkan semua file gambar
    image_files = []
    for ext in image_exts:
        image_files.extend(list(source_dir.glob(f"**/*{ext}")))
    
    # Bagi file gambar ke train, valid, test (80/10/10)
    import random
    random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(0.8 * n_total)
    n_valid = int(0.1 * n_total)
    
    train_images = image_files[:n_train]
    valid_images = image_files[n_train:n_train+n_valid]
    test_images = image_files[n_train+n_valid:]
    
    # Salin file ke direktori yang sesuai
    splits = {
        'train': train_images,
        'valid': valid_images,
        'test': test_images
    }
    
    for split, images in splits.items():
        for img_path in images:
            # Salin gambar
            dst_img = output_dir / split / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Cari dan salin label jika ada
            label_path = img_path.parent / f"{img_path.stem}{label_ext}"
            if label_path.exists():
                dst_lbl = output_dir / split / 'labels' / f"{img_path.stem}{label_ext}"
                shutil.copy2(label_path, dst_lbl)

def _count_dataset_files(directory: Path) -> Dict[str, int]:
    """
    Hitung jumlah file dalam dataset.
    
    Args:
        directory: Path direktori dataset
        
    Returns:
        Dictionary berisi jumlah file
    """
    stats = {'images': 0, 'labels': 0, 'other': 0}
    
    # Hitung file dalam direktori
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    label_exts = ['.txt', '.xml', '.json']
    
    for file_path in directory.glob('**/*.*'):
        ext = file_path.suffix.lower()
        if ext in image_exts:
            stats['images'] += 1
        elif ext in label_exts:
            stats['labels'] += 1
        else:
            stats['other'] += 1
    
    return stats

def _update_progress(ui_components: Dict[str, Any], value: int, message: str) -> None:
    """
    Update progress bar dan progress tracker.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress
    """
    # Update progress bar
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    if progress_bar:
        progress_bar.value = value
    
    if progress_message:
        progress_message.value = message
    
    # Update progress tracker
    tracker_key = 'dataset_downloader_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)
        
    # Log ke logger
    logger = ui_components.get('logger')
    if logger and value % 20 == 0:  # Log setiap 20%
        logger.info(f"🔄 {message} ({value}%)")