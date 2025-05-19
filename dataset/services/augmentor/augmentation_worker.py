"""
File: smartcash/dataset/services/augmentor/augmentation_worker.py
Deskripsi: Worker untuk augmentasi file individu dengan dukungan denominasi mata uang Rupiah
"""

import os
import cv2
import uuid
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import utils untuk denominasi dan label dari modul terkonsolidasi
from smartcash.dataset.utils.denomination_utils import DENOMINATION_CLASS_MAP, get_denomination_label, extract_info_from_filename
from smartcash.dataset.utils.label_utils import get_classes_from_label

class AugmentationWorker:
    """
    Worker untuk menangani proses augmentasi dataset secara paralel
    """
    
    def __init__(
        self,
        pipeline,
        num_variations: int = 2,
        output_prefix: str = 'aug',
        process_bboxes: bool = True,
        validate_results: bool = True,
        bbox_augmentor = None,
        max_workers: int = None
    ):
        """
        Inisialisasi worker augmentasi
        
        Args:
            pipeline: Pipeline augmentasi
            num_variations: Jumlah variasi yang akan dibuat
            output_prefix: Prefix untuk file output
            process_bboxes: Proses bounding box juga
            validate_results: Validasi hasil augmentasi
            bbox_augmentor: Augmentor untuk bounding box
            max_workers: Jumlah worker maksimum (default: None = CPU count)
        """
        self.pipeline = pipeline
        self.num_variations = num_variations
        self.output_prefix = output_prefix
        self.process_bboxes = process_bboxes
        self.validate_results = validate_results
        self.bbox_augmentor = bbox_augmentor
        self.max_workers = max_workers
    
    def process_batch(
        self,
        image_paths: List[str],
        labels_input_dir: str = None,
        images_output_dir: str = None,
        labels_output_dir: str = None,
        track_multi_class: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Proses batch file gambar secara paralel
        
        Args:
            image_paths: List path file gambar
            labels_input_dir: Direktori input label
            images_output_dir: Direktori output gambar
            labels_output_dir: Direktori output label
            track_multi_class: Aktifkan tracking untuk multi-class
            
        Returns:
            List hasil augmentasi untuk setiap file
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit semua task
            future_to_path = {
                executor.submit(
                    process_single_file,
                    image_path,
                    self.pipeline,
                    self.num_variations,
                    self.output_prefix,
                    self.process_bboxes,
                    self.validate_results,
                    self.bbox_augmentor,
                    labels_input_dir,
                    images_output_dir,
                    labels_output_dir,
                    track_multi_class=track_multi_class
                ): image_path for image_path in image_paths
            }
            
            # Proses hasil
            for future in as_completed(future_to_path):
                image_path = future_to_path[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "status": "error",
                        "message": f"Error saat augmentasi {image_path}: {str(e)}",
                        "generated": 0
                    })
        
        return results

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
    class_id: Optional[str] = None,  # Parameter opsional untuk metadata kelas
    track_multi_class: bool = False  # Tracking untuk multi-class
) -> Dict[str, Any]:
    """
    Proses augmentasi untuk satu file gambar dengan format penamaan denominasi.
    
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
        track_multi_class: Aktifkan tracking untuk multi-class
        
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
        if labels_input_dir:
            label_name = f"{img_path.stem}.txt"
            label_path = os.path.join(labels_input_dir, label_name)
        
        # Dapatkan class ID dan semua kelas jika belum ditentukan
        main_class_id, all_classes = None, set()
        if class_id is None and label_path and os.path.exists(label_path):
            main_class_id, all_classes = get_classes_from_label(label_path)
            class_id = main_class_id
        
        # Baca data bbox jika perlu dan label ada
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
        
        # Extract info dari nama file original untuk uuid dll
        original_filename_info = extract_info_from_filename(img_path.stem)
        
        # Tentukan format penamaan file berdasarkan hasil ekstraksi
        if original_filename_info['is_valid']:
            # Gunakan format yang sama dengan file asli
            file_uuid = original_filename_info['uuid']
            augmentation_prefix = original_filename_info.get('prefix', 'rp')
        else:
            # Fallback ke format default
            file_uuid = img_path.stem  # Gunakan stem sebagai uuid jika format tak dikenali
            augmentation_prefix = 'rp'
        
        # Tracking untuk multi-class jika diaktifkan
        multi_class_update = {}
        
        # Proses augmentasi untuk setiap variasi
        results = []
        for var_idx in range(max(1, num_variations)):  # Pastikan minimal 1 variasi
            try:
                # Augmentasi gambar dengan error handling yang lebih baik
                try:
                    # Tambahkan class_labels sebagai parameter kosong untuk memenuhi kebutuhan label_fields
                    augmented = pipeline(image=image, class_labels=[])
                    augmented_image = augmented['image']
                except Exception as aug_error:
                    print(f"Error saat augmentasi {os.path.basename(image_path)}: {str(aug_error)}")
                    # Gunakan gambar asli sebagai fallback jika augmentasi gagal
                    augmented_image = image.copy()
                    augmented = {'transforms': []}
                
                # Debug info - simpan ke variabel untuk menghindari overhead log
                debug_info = f"Debug: File: {os.path.basename(image_path)}, Shape: {augmented_image.shape}, Original: {image.shape}"
                
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
                
                # Debug info untuk dimensi gambar yang sangat kecil, tapi tidak melakukan validasi
                if augmented_image.shape[0] <= 10 or augmented_image.shape[1] <= 10:
                    print(f"{debug_info} - Peringatan: dimensi gambar sangat kecil tetapi tetap diproses")
                
                # Debug info untuk bbox yang hilang, tapi tidak melakukan validasi
                if process_bboxes and bboxes and not augmented_bboxes:
                    print(f"{debug_info} - Peringatan: tidak ada bbox yang valid tetapi tetap diproses")
                
                # Generate nama file output dengan format denominasi
                if class_id in DENOMINATION_CLASS_MAP:
                    # Format: aug_rp_100k_uuid_var_1.jpg
                    output_basename = f"aug_{augmentation_prefix}_{get_denomination_label(class_id)}_{file_uuid}_var_{var_idx+1}"
                else:
                    # Format fallback: aug_rp_unknown_uuid_var_1.jpg
                    output_basename = f"aug_{augmentation_prefix}_unknown_{file_uuid}_var_{var_idx+1}"
                
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
                
                # Update tracking multi-class jika diaktifkan
                if track_multi_class:
                    for cls in all_classes:
                        if cls not in multi_class_update:
                            multi_class_update[cls] = 0
                        multi_class_update[cls] += 1
                
            except Exception as e:
                # Skip variasi yang error
                continue
        
        # Return hasil proses
        result = {
            "status": "success",
            "image_id": img_id,
            "class_id": class_id,  # Kelas utama
            "denomination": get_denomination_label(class_id) if class_id else "unknown",
            "all_classes": list(all_classes) if all_classes else [class_id] if class_id else [],  # Semua kelas
            "original_path": str(img_path),
            "generated": len(results),
            "variations": results
        }
        
        # Tambahkan tracking multi-class jika diaktifkan
        if track_multi_class and multi_class_update:
            result["multi_class_update"] = multi_class_update
        
        return result
    
    except Exception as e:
        # Handle error
        return {
            "status": "error",
            "message": f"Error saat augmentasi {str(image_path)}: {str(e)}",
            "generated": 0
        }