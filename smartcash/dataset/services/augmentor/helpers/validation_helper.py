"""
File: smartcash/dataset/services/augmentor/helpers/validation_helper.py
Deskripsi: Helper untuk validasi input file augmentasi dan metadata
"""

import os
import glob
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging

def validate_input_files(images_dir: str, file_prefix: str, logger = None) -> Tuple[List[str], Dict[str, Any]]:
    """
    Validasi file input untuk augmentasi dengan file pattern detection.
    Jika tidak ada file dengan prefix tertentu, akan mencari semua file gambar.
    
    Args:
        images_dir: Path direktori gambar input
        file_prefix: Prefix file untuk dicari
        logger: Logger untuk logging
        
    Returns:
        Tuple (list file yang ditemukan, dict hasil validasi)
    """
    # Konversi ke Path untuk operasi path yang lebih robust
    images_path = Path(images_dir)
    
    # Cek apakah direktori ada
    if not images_path.exists():
        if logger:
            logger.warning(f"⚠️ Direktori {images_dir} tidak ditemukan, mencoba membuat direktori...")
        try:
            # Buat direktori output jika belum ada
            os.makedirs(images_dir, exist_ok=True)
        except Exception as e:
            if logger:
                logger.error(f"❌ Gagal membuat direktori {images_dir}: {str(e)}")
    
    # Inisialisasi list untuk menyimpan file gambar
    image_files = []
    
    # Dapatkan file dengan pattern spesifik
    pattern = f"{file_prefix}_*.jpg"
    if images_path.exists():
        pattern_files = list(images_path.glob(pattern))
        image_files.extend([str(f) for f in pattern_files])
    
    # Jika tidak ada file dengan pattern spesifik, coba cari semua file gambar
    if not image_files and images_path.exists():
        if logger:
            logger.info(f"⚠️ Tidak ada file dengan pola {pattern}, mencari semua file gambar...")
        
        # Cari semua file gambar umum
        all_patterns = ["*.jpg", "*.jpeg", "*.png"]
        for img_pattern in all_patterns:
            pattern_files = list(images_path.glob(img_pattern))
            image_files.extend([str(f) for f in pattern_files])
    
    # Daftar lokasi alternatif untuk mencari gambar
    alternative_locations = [
        # Subdirektori images
        images_path.parent / "images",
        # Direktori data/images
        Path("data/images"),
        # Direktori data/raw/images
        Path("data/raw/images"),
        # Direktori data/preprocessed/images
        Path("data/preprocessed/images"),
        # Direktori data/preprocessed/train/images
        Path("data/preprocessed/train/images"),
        # Direktori data/preprocessed/valid/images
        Path("data/preprocessed/valid/images"),
        # Direktori data/preprocessed/test/images
        Path("data/preprocessed/test/images")
    ]
    
    # Buat context logger khusus untuk augmentasi yang tidak mempengaruhi modul lain
    aug_logger = None
    if logger:
        # Cek apakah logger memiliki metode bind
        if hasattr(logger, 'bind'):
            aug_logger = logger.bind(context="augmentation_only")
        else:
            aug_logger = logger
    
    # Jika tidak ada file gambar ditemukan, coba cari di lokasi alternatif
    if not image_files:
        if aug_logger:
            aug_logger.debug(f"🔍 Tidak ada gambar di direktori {images_dir}, mencoba lokasi alternatif...")
        
        for alt_path in alternative_locations:
            alt_dir = images_path / alt_path
            if alt_dir.exists():
                if aug_logger:
                    aug_logger.debug(f"🔍 Mencari gambar di {alt_dir}...")
                
                for img_pattern in all_patterns:
                    pattern_files = list(alt_dir.glob(img_pattern))
                    if pattern_files:
                        image_files.extend([str(f) for f in pattern_files])
                        if aug_logger:
                            aug_logger.debug(f"✅ Menemukan {len(pattern_files)} gambar di {alt_dir}")
                        break
    
    # Jika masih tidak ada file gambar, coba cari di direktori parent
    if not image_files and images_path.parent.exists() and images_path.parent != images_path:
        parent_dir = images_path.parent
        if aug_logger:
            aug_logger.debug(f"🔍 Mencari gambar di direktori parent {parent_dir}...")
        
        # Cari di direktori parent/images jika ada
        parent_images_dir = parent_dir / 'images'
        if parent_images_dir.exists():
            for img_pattern in all_patterns:
                pattern_files = list(parent_images_dir.glob(img_pattern))
                if pattern_files:
                    image_files.extend([str(f) for f in pattern_files])
                    if aug_logger:
                        aug_logger.debug(f"✅ Menemukan {len(pattern_files)} gambar di {parent_images_dir}")
                    break
    
    # Jika masih tidak ada file gambar, cari di direktori lain yang umum
    if not image_files:
        common_dirs = [
            Path('data/raw/images'),
            Path('data/raw'),
            Path('data/preprocessed/images'),
            Path('data/preprocessed'),
            Path('data/images'),
            Path('data')
        ]
        
        for common_dir in common_dirs:
            if common_dir.exists():
                if aug_logger:
                    aug_logger.debug(f"🔍 Mencari gambar di direktori umum {common_dir}...")
                
                for img_pattern in all_patterns:
                    pattern_files = list(common_dir.glob(img_pattern))
                    if pattern_files:
                        image_files.extend([str(f) for f in pattern_files])
                        if aug_logger:
                            aug_logger.debug(f"✅ Menemukan {len(pattern_files)} gambar di {common_dir}")
                        break
    
    # Jika masih tidak ada file gambar, berikan pesan error dengan konteks augmentasi
    if not image_files:
        error_message = f"❌ Tidak ada gambar di direktori input: {images_dir}"
        if aug_logger:
            aug_logger.debug(error_message)
        
        return [], {"success": False, "error": error_message, "context": "augmentation_only"}
    
    # Log jumlah file yang ditemukan
    if logger:
        logger.info(f"🔍 Ditemukan {len(image_files)} file gambar di {images_dir}")
    
    # Return hasil sukses
    result = {
        "success": True,
        "message": f"Ditemukan {len(image_files)} file gambar",
        "count": len(image_files),
        "directory": images_dir
    }
    
    return image_files, result

def validate_augmentation_parameters(
    augmentation_types: List[str], 
    num_variations: int,
    process_bboxes: bool,
    target_balance: bool
) -> Dict[str, Any]:
    """
    Validasi parameter augmentasi.
    
    Args:
        augmentation_types: List tipe augmentasi
        num_variations: Jumlah variasi
        process_bboxes: Process bounding box
        target_balance: Target balancing
        
    Returns:
        Dictionary hasil validasi
    """
    # Validasi parameter dengan one-liner
    errors = []
    
    # Check augmentation_types
    if not augmentation_types:
        errors.append("Tipe augmentasi tidak boleh kosong")
    
    # Check num_variations
    if num_variations <= 0:
        errors.append("Jumlah variasi harus lebih dari 0")
    
    # Return hasil validasi
    if errors:
        return {
            "success": False,
            "message": "Validasi parameter gagal",
            "errors": errors
        }
    
    return {
        "success": True,
        "message": "Parameter valid"
    }

def check_output_file_exists(output_dir: str, file_id: str, output_prefix: str) -> bool:
    """
    Cek apakah file output sudah ada.
    
    Args:
        output_dir: Direktori output
        file_id: ID file
        output_prefix: Prefix output
        
    Returns:
        Boolean apakah file output sudah ada
    """
    # Format nama file output
    pattern = f"{output_prefix}_{file_id}_var*.jpg"
    existing_files = glob.glob(os.path.join(output_dir, pattern))
    
    return len(existing_files) > 0

def validate_class_metadata(
    class_data: Dict[str, Any], 
    target_count: int,
    logger = None
) -> Dict[str, Any]:
    """
    Validasi metadata kelas untuk augmentasi.
    
    Args:
        class_data: Data kelas hasil _prepare_balancing
        target_count: Target jumlah instance per kelas
        logger: Logger untuk logging
        
    Returns:
        Dictionary hasil validasi
    """
    # Cek apakah ada data kelas
    if not class_data or 'class_counts' not in class_data:
        return {
            "success": False,
            "message": "Data kelas tidak valid"
        }
    
    # Cek apakah ada selected files
    if 'selected_files' not in class_data or not class_data['selected_files']:
        return {
            "success": False,
            "message": "Tidak ada file yang terpilih untuk augmentasi"
        }
    
    # Cek apakah ada class yang perlu di-augmentasi
    augmentation_needs = class_data.get('augmentation_needs', {})
    classes_to_augment = [cls for cls, need in augmentation_needs.items() if need > 0]
    
    if not classes_to_augment and logger:
        logger.info("ℹ️ Tidak ada kelas yang perlu diaugmentasi (semua sudah memenuhi target)")
    
    total_files = len(class_data.get('selected_files', []))
    if logger:
        logger.info(f"📊 Total: {len(classes_to_augment)} kelas perlu augmentasi, {total_files} file terpilih")
    
    # Return hasil validasi
    return {
        "success": True,
        "message": f"Data kelas valid: {len(classes_to_augment)} kelas perlu augmentasi",
        "classes_to_augment": len(classes_to_augment),
        "selected_files": total_files
    }