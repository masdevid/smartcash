"""
File: smartcash/dataset/services/augmentor/augmentation_worker.py
Deskripsi: Worker untuk augmentasi file individu dengan tracking multi-class dan optimasi one-liner
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from collections import defaultdict

def get_classes_from_label(label_path: str) -> Tuple[Optional[str], Set[str], Dict[str, int]]:
    """
    Ekstrak ID kelas utama, semua kelas, dan distribusi kelas dari file label YOLOv5.
    
    Args:
        label_path: Path file label
        
    Returns:
        Tuple (ID kelas utama, set semua kelas, distribusi kelas)
    """
    try:
        if not os.path.exists(label_path): return None, set(), {}
        
        # Baca file dan ekstrak semua class_ids dengan one-liner
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Initialize dengan one-liner
        classes_counter = defaultdict(int)
        
        # Parse dengan one-liner where possible
        [classes_counter.update({parts[0]: classes_counter[parts[0]] + 1}) 
         for line in lines if len(parts := line.strip().split()) >= 5 
         and parts[0] != "-1" and parts[0].lower() != "unknown"]
        
        # Dapatkan main class dan semua kelas dengan one-liner
        main_class = min(classes_counter.keys()) if classes_counter else None
        all_classes = set(classes_counter.keys())
        
        return main_class, all_classes, dict(classes_counter)
    except Exception:
        return None, set(), {}

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
    class_id: Optional[str] = None,
    track_multi_class: bool = False
) -> Dict[str, Any]:
    """
    Proses augmentasi untuk satu file gambar dengan tracking multi-class.
    
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
        track_multi_class: Aktifkan tracking multi-class distributions
        
    Returns:
        Dictionary hasil augmentasi dengan metadata
    """
    try:
        # Validasi input dengan one-liner
        if not os.path.exists(image_path): 
            return {"status": "error", "message": f"File tidak ditemukan: {image_path}", "generated": 0}
        
        # Baca gambar dengan one-liner
        image = cv2.imread(str(image_path))
        if image is None: 
            return {"status": "error", "message": f"Tidak dapat membaca gambar: {image_path}", "generated": 0}
        
        # Dapatkan path label dengan one-liner
        img_name = Path(image_path).stem
        label_path = os.path.join(labels_input_dir, f"{img_name}.txt") if labels_input_dir else None
        
        # Dapatkan class ID, semua kelas, dan distribusi kelas
        main_class_id, all_classes, class_distribution = (None, set(), {})
        if label_path and os.path.exists(label_path):
            main_class_id, all_classes, class_distribution = get_classes_from_label(label_path)
            
        # Gunakan class_id dari parameter atau dari label
        class_id = class_id or main_class_id
        
        # Baca data bbox dengan one-liner
        bboxes = []
        if process_bboxes and label_path and os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    bboxes = [line.strip() for line in f.readlines()]
            except Exception:
                process_bboxes = False
        
        # Siapkan metadata
        img_id = Path(image_path).stem
        original_h, original_w = image.shape[:2]
        
        # Proses augmentasi untuk setiap variasi
        results, class_updates = [], defaultdict(int)
        for var_idx in range(num_variations):
            try:
                # Augmentasi gambar dengan pipeline
                augmented = pipeline(image=image)
                augmented_image = augmented['image']
                
                # Proses bounding box dengan conditional one-liner
                augmented_bboxes = (
                    bbox_augmentor.transform_bboxes(image, augmented_image, bboxes, augmented.get('transforms', []))
                    if process_bboxes and bbox_augmentor and bboxes else bboxes
                )
                
                # Validasi hasil jika diminta
                if validate_results:
                    # Skip jika dimensi tidak valid atau bbox gagal diproses
                    if augmented_image.shape[0] <= 0 or augmented_image.shape[1] <= 0 or (process_bboxes and not augmented_bboxes):
                        continue
                
                # Generate nama file dengan one-liner
                output_basename = f"{output_prefix}_{img_id}_var{var_idx+1}"
                output_paths = {
                    'image': os.path.join(images_output_dir, f"{output_basename}.jpg"),
                    'label': os.path.join(labels_output_dir, f"{output_basename}.txt") if process_bboxes and labels_output_dir else None
                }
                
                # Simpan hasil augmentasi gambar
                cv2.imwrite(output_paths['image'], augmented_image)
                
                # Simpan hasil augmentasi bbox dan track class update
                if process_bboxes and output_paths['label'] and augmented_bboxes:
                    with open(output_paths['label'], 'w') as f:
                        for bbox in augmented_bboxes:
                            f.write(f"{bbox}\n")
                            
                            # Track perubahan distribusi kelas jika diaktifkan
                            if track_multi_class:
                                try:
                                    # Extract class ID dari bbox dengan one-liner
                                    bbox_class = bbox.strip().split()[0]
                                    if bbox_class != "-1" and bbox_class.lower() != "unknown":
                                        class_updates[bbox_class] += 1
                                except IndexError:
                                    pass
                
                # Catat hasil untuk variasi ini dengan one-liner
                results.append({
                    "variant": var_idx + 1,
                    "image_path": output_paths['image'],
                    "label_path": output_paths['label'],
                    "original_size": (original_w, original_h),
                    "augmented_size": (augmented_image.shape[1], augmented_image.shape[0]),
                    "has_bboxes": len(augmented_bboxes) > 0 if process_bboxes else False
                })
                
            except Exception as e:
                # Skip variasi yang error
                continue
        
        # Buat output dengan one-liner update
        output = {
            "status": "success",
            "image_id": img_id,
            "class_id": class_id,
            "all_classes": list(all_classes) if all_classes else [class_id] if class_id else [],
            "original_path": str(image_path),
            "generated": len(results),
            "variations": results
        }
        
        # Tambahkan informasi multi-class jika tracking diaktifkan
        if track_multi_class:
            output.update({
                "class_distribution": class_distribution,
                "multi_class_update": dict(class_updates)
            })
        
        return output
    
    except Exception as e:
        # Handle error dengan one-liner
        return {"status": "error", "message": f"Error saat augmentasi {image_path}: {str(e)}", "generated": 0}