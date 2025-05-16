"""
File: smartcash/dataset/services/augmentor/helpers/ensure_output.py
Deskripsi: Helper untuk memastikan direktori output ada dan file augmentasi berhasil disimpan
"""

import os
import glob
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path

def ensure_output_directories_exist(paths: Dict[str, str], logger=None) -> bool:
    """
    Memastikan semua direktori output ada dan siap digunakan.
    
    Args:
        paths: Dictionary path hasil setup_paths
        logger: Logger untuk mencatat proses
        
    Returns:
        Boolean keberhasilan
    """
    try:
        # Buat direktori output yang diperlukan
        for key in ['images_output_dir', 'labels_output_dir']:
            if key in paths:
                os.makedirs(paths[key], exist_ok=True)
                if logger:
                    logger.info(f"📁 Memastikan direktori {paths[key]} ada")
        
        # Periksa apakah direktori berhasil dibuat
        for key in ['images_output_dir', 'labels_output_dir']:
            if key in paths and not os.path.exists(paths[key]):
                if logger:
                    logger.error(f"❌ Gagal membuat direktori {paths[key]}")
                return False
        
        return True
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat membuat direktori output: {str(e)}")
        return False

def verify_augmentation_files(paths: Dict[str, str], output_prefix: str, logger=None) -> Dict[str, Any]:
    """
    Verifikasi file hasil augmentasi ada di direktori output.
    
    Args:
        paths: Dictionary path hasil setup_paths
        output_prefix: Prefix untuk file hasil augmentasi
        logger: Logger untuk mencatat proses
        
    Returns:
        Dictionary hasil verifikasi
    """
    result = {
        "status": "success",
        "found_images": 0,
        "found_labels": 0,
        "image_examples": [],
        "message": ""
    }
    
    try:
        # Cari semua file gambar hasil augmentasi
        image_pattern = os.path.join(paths['images_output_dir'], f"{output_prefix}_*.jpg")
        image_files = glob.glob(image_pattern)
        result["found_images"] = len(image_files)
        
        # Cari semua file label hasil augmentasi
        label_pattern = os.path.join(paths['labels_output_dir'], f"{output_prefix}_*.txt")
        label_files = glob.glob(label_pattern)
        result["found_labels"] = len(label_files)
        
        # Simpan contoh file yang ditemukan
        if image_files:
            result["image_examples"] = [os.path.basename(f) for f in image_files[:3]]
        
        # Log hasil verifikasi (hanya jumlah total tanpa detail)
        if logger:
            logger.info(f"🔍 Verifikasi file augmentasi: {result['found_images']} gambar, {result['found_labels']} label")
        
        # Tentukan status berdasarkan jumlah file yang ditemukan
        if result["found_images"] == 0:
            result["status"] = "warning"
            result["message"] = "Tidak ada file gambar hasil augmentasi yang ditemukan"
        elif result["found_labels"] == 0 and paths.get('labels_output_dir'):
            result["status"] = "warning"
            result["message"] = "Tidak ada file label hasil augmentasi yang ditemukan"
        
        return result
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat verifikasi file augmentasi: {str(e)}")
        return {
            "status": "error",
            "found_images": 0,
            "found_labels": 0,
            "image_examples": [],
            "message": f"Error saat verifikasi: {str(e)}"
        }

def copy_augmentation_files(
    paths: Dict[str, str], 
    output_prefix: str, 
    target_dir: str, 
    split: str, 
    logger=None
) -> Dict[str, Any]:
    """
    Salin file hasil augmentasi ke direktori target.
    
    Args:
        paths: Dictionary path hasil setup_paths
        output_prefix: Prefix untuk file hasil augmentasi
        target_dir: Direktori target untuk menyalin file
        split: Nama split dataset ('train', 'valid', 'test')
        logger: Logger untuk mencatat proses
        
    Returns:
        Dictionary hasil penyalinan
    """
    result = {
        "status": "success",
        "copied_images": 0,
        "copied_labels": 0,
        "message": ""
    }
    
    try:
        # Pastikan direktori target ada
        target_images_dir = os.path.join(target_dir, split, 'images')
        target_labels_dir = os.path.join(target_dir, split, 'labels')
        os.makedirs(target_images_dir, exist_ok=True)
        os.makedirs(target_labels_dir, exist_ok=True)
        
        # Cari semua file gambar hasil augmentasi
        image_pattern = os.path.join(paths['images_output_dir'], f"{output_prefix}_*.jpg")
        image_files = glob.glob(image_pattern)
        
        # Salin file gambar ke direktori target
        for img_file in image_files:
            img_name = os.path.basename(img_file)
            target_img_path = os.path.join(target_images_dir, img_name)
            shutil.copy2(img_file, target_img_path)
            result["copied_images"] += 1
            
            # Cari file label yang sesuai
            label_name = f"{os.path.splitext(img_name)[0]}.txt"
            label_file = os.path.join(paths['labels_output_dir'], label_name)
            
            # Salin file label ke direktori target jika ada
            if os.path.exists(label_file):
                target_label_path = os.path.join(target_labels_dir, label_name)
                shutil.copy2(label_file, target_label_path)
                result["copied_labels"] += 1
        
        # Log hasil penyalinan (hanya jumlah total)
        if logger:
            logger.info(f"✅ Selesai memproses {result['copied_images']} file")
        
        # Tentukan status berdasarkan jumlah file yang disalin
        if result["copied_images"] == 0:
            result["status"] = "warning"
            result["message"] = "Tidak ada file gambar yang disalin"
        
        return result
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat menyalin file augmentasi: {str(e)}")
        return {
            "status": "error",
            "copied_images": 0,
            "copied_labels": 0,
            "message": f"Error saat menyalin: {str(e)}"
        }
