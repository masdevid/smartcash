"""
File: smartcash/dataset/services/augmentor/augmentation_worker.py
Deskripsi: Worker untuk augmentasi file individu dengan dukungan metadata kelas
"""

import os
import cv2
import shutil
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

def get_class_from_label(label_path: str) -> Optional[str]:
    """
    Ekstrak ID kelas utama dari file label YOLOv5.
    
    Args:
        label_path: Path file label
        
    Returns:
        ID kelas utama atau None jika tidak ada
    """
    try:
        if not os.path.exists(label_path):
            return None
            
        # Baca file label
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Cari class ID
        class_ids = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Format YOLOv5: class_id x y width height
                class_ids.append(parts[0])
                
        # Jika ada class ID, ambil yang terkecil (prioritas banknote)
        if class_ids:
            return min(class_ids)
        
        return None
    except Exception:
        return None

def process_single_file(
    image_path: str,
    pipeline,
    num_variations: int = 2,
    output_prefix: str = 'aug',
    process_bboxes: bool = True,
    validate_results: bool = True,
    bbox_augmentor = None,
    labels_input_dir: str = None,
    images_output_dir: str = None,
    labels_output_dir: str = None,
    class_id: Optional[str] = None  # Parameter opsional untuk metadata kelas
) -> Dict[str, Any]:
    """
    Proses augmentasi untuk satu file gambar dengan metadata kelas.
    
    Args:
        image_path: Path file gambar
        pipeline: Pipeline augmentasi
        num_variations: Jumlah variasi yang akan dibuat
        output_prefix: Prefix untuk file output
        process_bboxes: Proses bounding box juga
        validate_results: Validasi hasil augmentasi
        bbox_augmentor: Augmentor untuk bounding box
        labels_input_dir: Direktori input label
        images_output_dir: Direktori output gambar
        labels_output_dir: Direktori output label
        class_id: ID kelas untuk metadata (opsional)
        
    Returns:
        Dictionary hasil augmentasi dengan metadata
    """
    try:
        # Konversi path ke Path object
        img_path = Path(image_path)
        
        # Jika path tidak valid, return status error
        if not img_path.exists():
            return {
                "status": "error",
                "message": f"File tidak ditemukan: {img_path}",
                "generated": 0
            }
        
        # Baca gambar
        image = cv2.imread(str(img_path))
        if image is None:
            return {
                "status": "error",
                "message": f"Tidak dapat membaca gambar: {img_path}",
                "generated": 0
            }
        
        # Dapatkan path label jika perlu memproses bbox
        label_path = None
        if process_bboxes and labels_input_dir:
            label_name = f"{img_path.stem}.txt"
            label_path = os.path.join(labels_input_dir, label_name)
            
            # Jika file label tidak ada, skip processing bbox
            if not os.path.exists(label_path):
                process_bboxes = False
        
        # Dapatkan class ID jika belum ditentukan
        if class_id is None and label_path:
            class_id = get_class_from_label(label_path)
        
        # Baca data bbox jika perlu
        bboxes = []
        if process_bboxes and label_path and os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    bboxes = [line.strip() for line in f.readlines()]
            except Exception:
                # Jika gagal membaca bbox, nonaktifkan processing
                process_bboxes = False
        
        # Siapkan metadata
        img_id = img_path.stem
        original_h, original_w = image.shape[:2]
        
        # Proses augmentasi untuk setiap variasi
        results = []
        for var_idx in range(num_variations):
            try:
                # Augmentasi gambar
                augmented = pipeline(image=image)
                augmented_image = augmented['image']
                
                # Dapatkan bounding box baru jika perlu
                augmented_bboxes = []
                if process_bboxes and bbox_augmentor and bboxes:
                    # Gunakan bbox augmentor jika tersedia
                    augmented_bboxes = bbox_augmentor.transform_bboxes(
                        image, augmented_image, bboxes, augmented.get('transforms', [])
                    )
                elif bboxes:
                    # Gunakan bbox original jika tidak perlu transform
                    augmented_bboxes = bboxes
                
                # Validasi hasil jika diminta
                if validate_results:
                    # Cek dimensi gambar
                    if augmented_image.shape[0] <= 0 or augmented_image.shape[1] <= 0:
                        continue
                    
                    # Cek integritas bbox jika ada
                    if process_bboxes and not augmented_bboxes:
                        continue
                
                # Generate nama file output
                output_basename = f"{output_prefix}_{img_id}_var{var_idx+1}"
                output_image_path = os.path.join(images_output_dir, f"{output_basename}.jpg")
                
                # Simpan hasil augmentasi gambar
                cv2.imwrite(output_image_path, augmented_image)
                
                # Simpan hasil augmentasi bbox jika perlu
                if process_bboxes and labels_output_dir and augmented_bboxes:
                    output_label_path = os.path.join(labels_output_dir, f"{output_basename}.txt")
                    with open(output_label_path, 'w') as f:
                        for bbox in augmented_bboxes:
                            f.write(f"{bbox}\n")
                
                # Catat hasil untuk variasi ini
                results.append({
                    "variant": var_idx + 1,
                    "image_path": output_image_path,
                    "original_size": (original_w, original_h),
                    "augmented_size": (augmented_image.shape[1], augmented_image.shape[0]),
                    "has_bboxes": len(augmented_bboxes) > 0 if process_bboxes else False
                })
                
            except Exception as e:
                # Skip variasi yang error
                continue
        
        # Return hasil proses
        return {
            "status": "success",
            "image_id": img_id,
            "class_id": class_id,  # Tambahkan informasi kelas
            "original_path": str(img_path),
            "generated": len(results),
            "variations": results
        }
    
    except Exception as e:
        # Handle error
        return {
            "status": "error",
            "message": f"Error saat augmentasi {str(image_path)}: {str(e)}",
            "generated": 0
        }