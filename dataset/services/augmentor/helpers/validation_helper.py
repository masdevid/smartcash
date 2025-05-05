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
    
    Args:
        images_dir: Path direktori gambar input
        file_prefix: Prefix file untuk dicari
        logger: Logger untuk logging
        
    Returns:
        Tuple (list file yang ditemukan, dict hasil validasi)
    """
    # Buat direktori output jika belum ada
    os.makedirs(images_dir, exist_ok=True)
    
    # Dapatkan file dengan pattern
    pattern = f"{file_prefix}_*.jpg"
    image_files = glob.glob(os.path.join(images_dir, pattern))
    
    # Validasi input file
    if not image_files:
        result = {
            "success": False,
            "message": f"Tidak ada file gambar ditemukan dengan pola {pattern} di direktori {images_dir}"
        }
        return [], result
    
    # Log jumlah file yang ditemukan
    if logger:
        logger.info(f"🔍 Ditemukan {len(image_files)} file dengan pattern {pattern} di {images_dir}")
    
    # Return hasil sukses
    result = {
        "success": True,
        "message": f"Ditemukan {len(image_files)} file gambar",
        "count": len(image_files)
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