"""
File: smartcash/dataset/services/augmentor/augmentation_worker.py
Deskripsi: Worker untuk augmentasi gambar dan label dataset dengan teknik SRP
"""

import os
import cv2
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import albumentations as A
from pathlib import Path

def process_single_file(
    image_path: str,
    pipeline: A.Compose,
    num_variations: int = 2,
    output_prefix: str = 'aug',
    process_bboxes: bool = True,
    validate_results: bool = True,
    bbox_augmentor = None,
    labels_input_dir: str = '',
    images_output_dir: str = '',
    labels_output_dir: str = ''
) -> Dict[str, Any]:
    """
    Proses augmentasi satu file dengan multiple variasi.
    
    Args:
        image_path: Path file gambar input
        pipeline: Pipeline augmentasi Albumentations
        num_variations: Jumlah variasi augmentasi
        output_prefix: Prefix nama file output
        process_bboxes: Apakah label bbox juga diproses
        validate_results: Validasi hasil augmentasi
        bbox_augmentor: Instance dari BBoxAugmentor
        labels_input_dir: Path direktori label input
        images_output_dir: Path direktori output gambar
        labels_output_dir: Path direktori output label
        
    Returns:
        Dictionary berisi hasil augmentasi file
    """
    try:
        # Extract filename
        filename = os.path.basename(image_path)
        filename_stem = os.path.splitext(filename)[0]
        
        # Cek apakah ada file label
        label_path = os.path.join(labels_input_dir, f"{filename_stem}.txt")
        has_label = os.path.exists(label_path)
        
        # Jika proses bbox tapi tidak ada label, skip
        if process_bboxes and not has_label:
            return {
                "status": "skipped", 
                "message": f"File label tidak ditemukan: {label_path}", 
                "generated": 0
            }
        
        # Load gambar
        image = cv2.imread(image_path)
        if image is None:
            return {
                "status": "error", 
                "message": f"Gagal membaca gambar: {image_path}", 
                "generated": 0
            }
        
        # Konversi ke RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Dapatkan bboxes dari file label jika ada
        bboxes = []
        class_ids = []
        
        if has_label and process_bboxes:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: class_id, x_center, y_center, width, height
                        bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                        bboxes.append(bbox)
                        class_ids.append(class_id)
        
        # Lakukan augmentasi untuk setiap variasi
        augmented_count = 0
        
        for i in range(num_variations):
            # Ekstrak komponen nama file untuk penggunaan dalam augmented file
            original_parts = filename_stem.split('_', 1)
            # Format: [prefix]_[class+uuid]
            if len(original_parts) > 1:
                file_class_section = original_parts[1]  # Bagian kelas dan uuid
            else:
                # Fallback jika format tidak sesuai
                file_class_section = f"unknown_{uuid.uuid4().hex[:8]}"
                
            # Simpan original filename di augmented filename untuk memudahkan matching
            # Format: [augmented_prefix]_[source_prefix]_[class+uuid]_var[n]
            source_prefix = original_parts[0] if len(original_parts) > 0 else "unknown"
            
            # Generate nama file yang baru dengan menyimpan informasi sumber
            # Ini memudahkan untuk visualisasi perbandingan karena menyimpan uuid yang sama
            augmented_filename = f"{output_prefix}_{source_prefix}_{file_class_section}_var{i+1}"
            augmented_image_path = os.path.join(images_output_dir, f"{augmented_filename}.jpg")
            augmented_label_path = os.path.join(labels_output_dir, f"{augmented_filename}.txt")
            
            # Proses augmentasi
            try:
                # Transformasi dengan pipeline albumentations
                if process_bboxes and bboxes:
                    # Dengan bboxes
                    transformed = pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_ids
                    )
                    
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']
                    transformed_labels = transformed['class_labels']
                else:
                    # Tanpa bboxes
                    transformed = pipeline(image=image)
                    transformed_image = transformed['image']
                    transformed_bboxes = []
                    transformed_labels = []
                
                # Validasi hasil jika diminta
                if validate_results:
                    # Cek apakah gambar valid
                    if transformed_image is None or transformed_image.size == 0:
                        continue
                    
                    # Validasi bboxes
                    if process_bboxes and transformed_bboxes:
                        valid_bboxes = []
                        valid_labels = []
                        
                        for j, (bbox, class_id) in enumerate(zip(transformed_bboxes, transformed_labels)):
                            # Validasi: bbox valid jika ukuran > 0 dan koordinat dalam range [0,1]
                            x, y, w, h = bbox
                            if (0 <= x <= 1 and 0 <= y <= 1 and 
                                0 < w <= 1 and 0 < h <= 1 and
                                x - w/2 >= 0 and x + w/2 <= 1 and
                                y - h/2 >= 0 and y + h/2 <= 1):
                                valid_bboxes.append(bbox)
                                valid_labels.append(class_id)
                        
                        # Update bboxes dan labels dengan yang valid
                        transformed_bboxes = valid_bboxes
                        transformed_labels = valid_labels
                
                # Simpan gambar
                # Konversi kembali ke BGR untuk OpenCV
                output_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(augmented_image_path, output_image)
                
                # Simpan label jika ada
                if process_bboxes and transformed_bboxes:
                    with open(augmented_label_path, 'w') as f:
                        for j, (bbox, class_id) in enumerate(zip(transformed_bboxes, transformed_labels)):
                            # YOLO format: class_id, x_center, y_center, width, height
                            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
                
                augmented_count += 1
                
            except Exception as e:
                continue
        
        # Return results
        return {
            "status": "success",
            "generated": augmented_count,
            "original_file": filename,
            "message": f"Berhasil augmentasi {filename}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error saat proses file {os.path.basename(image_path)}: {str(e)}",
            "generated": 0
        }