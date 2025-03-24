"""
File: smartcash/dataset/utils/preprocessing_image_utils.py
Deskripsi: Utilitas untuk pemrosesan gambar selama preprocessing
"""

import os
import uuid
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.dataset.utils.label_utils import extract_class_from_label
from smartcash.dataset.utils.denomination_utils import get_denomination_label
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE, DENOMINATION_CLASS_MAP

def preprocess_single_image(
    img_path: Path, 
    labels_dir: Path, 
    target_images_dir: Path, 
    target_labels_dir: Path,
    pipeline, 
    storage, 
    file_prefix: str,
    preprocessing_options: Dict[str, Any] = None,
    logger=None
) -> bool:
    """
    Preprocess satu gambar dan label terkait dengan penamaan yang dioptimalkan.
    
    Args:
        img_path: Path gambar input
        labels_dir: Path direktori label input
        target_images_dir: Path direktori gambar output
        target_labels_dir: Path direktori label output
        pipeline: Pipeline preprocessing
        storage: Storage manager
        file_prefix: Prefix untuk file output
        preprocessing_options: Opsi preprocessing
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    # Generate nama file output berdasarkan nama file input
    img_id, label_path = img_path.stem, labels_dir / f"{img_path.stem}.txt"
    
    # Lewati jika tidak ada file label yang sesuai
    if not label_path.exists():
        if logger: logger.debug(f"⚠️ File label tidak ditemukan untuk {img_id}, dilewati")
        return False
        
    try:
        # Baca dan proses gambar
        if (image := cv2.imread(str(img_path))) is None:
            if logger: logger.warning(f"⚠️ Tidak dapat membaca gambar: {img_path}")
            return False
            
        # Setup pipeline dengan options dari parameter atau default
        options = preprocessing_options or {}
        pipeline.set_options(
            img_size=options.get('img_size', DEFAULT_IMG_SIZE), 
            normalize=options.get('normalize', True), 
            preserve_aspect_ratio=options.get('preserve_aspect_ratio', True)
        )
        
        # Preprocess gambar
        processed_image = pipeline.process(image)
        
        # Ekstrak kelas dan generate nama file dengan format denominasi
        banknote_class = extract_class_from_label(label_path, pipeline.config, logger)
        
        # Dapatkan class ID dari file label untuk denominasi
        class_id = None
        try:
            with open(label_path, 'r') as f:
                label_lines = f.readlines()
                if label_lines:
                    # Ekstrak semua class ID dari label
                    class_ids = []
                    for line in label_lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_ids.append(parts[0])
                    
                    # Prioritaskan kelas yang ada di mapping denominasi
                    valid_denomination_ids = [cls for cls in class_ids if cls in DENOMINATION_CLASS_MAP]
                    
                    # Jika ada kelas denominasi yang valid, ambil terkecil
                    if valid_denomination_ids:
                        class_id = min(valid_denomination_ids)
                    else:
                        # Jika tidak ada, ambil class ID terkecil
                        class_id = min(class_ids) if class_ids else None
        except Exception as e:
            if logger: logger.debug(f"⚠️ Gagal ekstrak class ID dari {label_path}: {str(e)}")
        
        # Generate nama file dengan format denominasi
        unique_id = str(uuid.uuid4())[:8]
        
        # Gunakan format denominasi jika class ID valid
        if class_id and class_id in DENOMINATION_CLASS_MAP:
            denomination = get_denomination_label(class_id)
            new_filename = f"{file_prefix}_{denomination}_{unique_id}"
        else:
            # Fallback jika kelas tidak valid atau tidak ada di mapping
            new_filename = f"{file_prefix}_unknown_{unique_id}"
        
        # Simpan hasil gambar dengan normalisasi jika perlu
        output_path = target_images_dir / f"{new_filename}.jpg"
        save_image = (processed_image * 255).astype(np.uint8) if preprocessing_options.get('normalize', True) and processed_image.dtype == np.float32 else processed_image
        cv2.imwrite(str(output_path), save_image)
        
        # Salin file label dengan nama baru dengan one-liner
        with open(label_path, 'r') as src_file, open(target_labels_dir / f"{new_filename}.txt", 'w') as dst_file:
            dst_file.write(src_file.read())
        
        # Simpan metadata dengan error handling untuk backward compatibility
        try:
            storage.save_metadata(
                split=Path(target_images_dir).parent.name, 
                image_id=new_filename, 
                metadata={
                    'original_path': str(img_path),
                    'original_id': img_id,
                    'original_size': (image.shape[1], image.shape[0]),
                    'processed_size': (processed_image.shape[1], processed_image.shape[0]),
                    'preprocessing_timestamp': time.time(),
                    'banknote_class': banknote_class,
                    'class_id': class_id,
                    'denomination': get_denomination_label(class_id) if class_id else 'unknown',
                    'new_filename': new_filename
                }
            )
        except Exception as e:
            if logger: logger.debug(f"⚠️ Storage tidak mendukung save_metadata: {str(e)}")
        
        return True
        
    except Exception as e:
        if logger: logger.error(f"❌ Error saat preprocessing {img_id[:5]}{'...' + img_id[-5:] if len(img_id) > 10 else ''}: {str(e)}")
        return False