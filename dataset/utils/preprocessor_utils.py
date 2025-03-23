"""
File: smartcash/dataset/utils/preprocessor_utils.py
Deskripsi: Fungsi helper untuk preprocessing dataset yang digunakan oleh dataset_preprocessor.py
"""

import os
import uuid
import time
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

def extract_class_from_label(label_path: Path, config: Dict = None, logger=None) -> Optional[str]:
    """
    Ekstrak kelas dari file label YOLO dengan memilih class ID terkecil (prioritas banknote).
    
    Args:
        label_path: Path ke file label
        config: Konfigurasi aplikasi
        logger: Logger untuk logging (opsional)
        
    Returns:
        Nama kelas atau None jika tidak ditemukan
    """
    if not label_path.exists():
        return None
        
    try:
        # Baca semua baris dari file label
        with open(label_path, 'r') as f:
            label_lines = f.readlines()
            
        # Cek apakah ada isi label
        if not label_lines:
            return None
        
        # Ekstrak semua class ID dari semua baris
        class_ids = []
        for line in label_lines:
            parts = line.strip().split()
            if parts:
                # Class ID ada di posisi pertama format YOLO
                class_ids.append(int(parts[0]))
        
        # Jika tidak ada class ID valid
        if not class_ids:
            return None
            
        # Ambil class ID terkecil (prioritas banknote)
        min_class_id = min(class_ids)
        
        # Map class ID ke nama kelas jika tersedia
        class_names = config.get('data', {}).get('class_names', {}) if config else {}
        if class_names and str(min_class_id) in class_names:
            return class_names[str(min_class_id)]
            
        # Fallback ke class ID jika nama kelas tidak tersedia
        return f"class{min_class_id}"
    except Exception as e:
        if logger:
            logger.debug(f"⚠️ Gagal ekstrak kelas dari {label_path}: {str(e)}")
        return None

def get_source_dir(split: str, config: Dict) -> str:
    """
    Dapatkan direktori sumber data split.
    
    Args:
        split: Split dataset ('train', 'valid', 'test')
        config: Konfigurasi aplikasi
        
    Returns:
        Path direktori sumber data
    """
    # Cek apakah ada path spesifik untuk split
    data_dir = config.get('data', {}).get('dir', 'data')
    
    # Jika punya local.split, gunakan itu
    if 'local' in config.get('data', {}) and split in config.get('data', {}).get('local', {}):
        return config['data']['local'][split]
    
    # Fallback ke direktori default
    return os.path.join(data_dir, split)

def update_progress(callback, current: int, total: int, message: str = None, status: str = 'info', **kwargs) -> None:
    """
    Update progress dengan callback dan notifikasi observer.
    
    Args:
        callback: Progress callback function
        current: Progress saat ini
        total: Total progress
        message: Pesan progress
        status: Status progress ('info', 'success', 'warning', 'error')
        **kwargs: Parameter tambahan untuk callback
    """
    # Call progress callback jika ada
    if callback:
        # Pastikan parameter tidak duplikat
        if 'current_progress' in kwargs and 'current_total' in kwargs:
            # Jangan gunakan current_total jika sudah diberikan dalam kwargs
            current_progress = kwargs.pop('current_progress', None)
            if current_progress is not None:
                kwargs['current_progress'] = current_progress
        
        callback(
            progress=current, 
            total=total, 
            message=message or f"Preprocessing progress: {int(current/total*100) if total > 0 else 0}%", 
            status=status, 
            **kwargs
        )
    
    # Notifikasi observer jika tidak disertakan flag suppress_notify
    if not kwargs.get('suppress_notify', False):
        try: 
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Hindari duplikasi parameter
            notify_kwargs = kwargs.copy()
            if 'current_progress' in notify_kwargs and 'current_total' in notify_kwargs:
                current_progress = notify_kwargs.pop('current_progress', None)
                if current_progress is not None:
                    notify_kwargs['current_progress'] = current_progress
            
            notify(
                event_type=EventTopics.PREPROCESSING_PROGRESS, 
                sender="dataset_preprocessor", 
                message=message or f"Preprocessing progress: {int(current/total*100) if total > 0 else 0}%", 
                progress=current, 
                total=total, 
                **notify_kwargs
            )
        except Exception: 
            pass

def resolve_splits(split: Optional[str]) -> List[str]:
    """
    Resolve split parameter menjadi list split yang akan diproses.
    
    Args:
        split: Split yang akan diproses ('train', 'valid', 'test', None untuk semua)
        
    Returns:
        List split yang akan diproses
    """
    # Jika split None atau 'all', proses semua split
    if not split or split.lower() == 'all':
        return ['train', 'valid', 'test']
    # Untuk split 'val', gunakan 'valid'
    elif split.lower() == 'val':
        return ['valid']
    # Untuk nilai split lainnya
    else:
        return [split]

def preprocess_single_image(img_path: Path, labels_dir: Path, 
                           target_images_dir: Path, target_labels_dir: Path,
                           pipeline, storage, file_prefix: str,
                           preprocessing_options: Dict[str, Any] = None,
                           logger=None) -> bool:
    """
    Preprocess satu gambar dan label terkait, dan simpan hasilnya dengan penamaan yang ditingkatkan.
    
    Args:
        img_path: Path ke file gambar
        labels_dir: Direktori berisi file label
        target_images_dir: Direktori output untuk gambar
        target_labels_dir: Direktori output untuk label
        pipeline: Pipeline preprocessing
        storage: Storage untuk menyimpan hasil
        file_prefix: Prefix untuk file output
        preprocessing_options: Opsi preprocessing tambahan
        logger: Logger untuk logging
        
    Returns:
        Boolean menunjukkan keberhasilan preprocessing
    """
    # Generate nama file output berdasarkan nama file input
    img_id = img_path.stem
    label_path = labels_dir / f"{img_id}.txt"
    
    # Lewati jika tidak ada file label yang sesuai
    if not label_path.exists():
        if logger:
            logger.debug(f"⚠️ File label tidak ditemukan untuk {img_id}, dilewati")
        return False
        
    # Proses gambar
    try:
        # Baca gambar
        image = cv2.imread(str(img_path))
        if image is None:
            if logger:
                logger.warning(f"⚠️ Tidak dapat membaca gambar: {img_path}")
            return False
            
        # Ambil opsi preprocessing dari options
        options = preprocessing_options or {}
        img_size = options.get('img_size', [640, 640])
        normalize = options.get('normalize', True)
        preserve_aspect_ratio = options.get('preserve_aspect_ratio', True)
        
        # Setup preprocessing options
        pipeline.set_options(img_size=img_size, normalize=normalize, 
                         preserve_aspect_ratio=preserve_aspect_ratio)
        
        # Preprocess gambar
        processed_image = pipeline.process(image)
        
        # Ekstrak kelas dari file label untuk penamaan file
        banknote_class = extract_class_from_label(label_path, pipeline.config, logger)
        
        # Generate ID unik untuk file
        unique_id = str(uuid.uuid4())[:8]  # 8 karakter pertama dari UUID
        
        # Generate nama file baru dengan format {prefix}_{class}_{unique_id}
        new_filename = f"{file_prefix}_{banknote_class or 'unknown'}_{unique_id}"
        
        # Simpan hasil gambar
        output_path = target_images_dir / f"{new_filename}.jpg"
        
        # Normalisasi jika perlu sebelum menyimpan
        if normalize and processed_image.dtype == np.float32:
            save_image = (processed_image * 255).astype(np.uint8)
        else:
            save_image = processed_image
            
        cv2.imwrite(str(output_path), save_image)
        
        # Salin file label dengan nama baru
        with open(label_path, 'r') as src_file:
            with open(target_labels_dir / f"{new_filename}.txt", 'w') as dst_file:
                dst_file.write(src_file.read())
        
        # Simpan metadata
        metadata = {
            'original_path': str(img_path),
            'original_id': img_id,
            'original_size': (image.shape[1], image.shape[0]),  # width, height
            'processed_size': (processed_image.shape[1], processed_image.shape[0]),
            'preprocessing_timestamp': time.time(),
            'banknote_class': banknote_class,
            'new_filename': new_filename
        }
        
        # Simpan ke storage - dengan penanganan error untuk backward compatibility
        try:
            storage.save_metadata(split=Path(target_images_dir).parent.name, image_id=new_filename, metadata=metadata)
        except (AttributeError, Exception) as e:
            # Jika metode save_metadata tidak ada (backward compatibility)
            if logger:
                logger.debug(f"⚠️ Storage tidak mendukung save_metadata: {str(e)}")
        
        return True
        
    except Exception as e:
        # Tampilkan pesan error dengan nama file yang terpotong
        short_id = img_id
        if len(short_id) > 15:
            short_id = f"...{short_id[-10:]}"
        
        if logger:
            logger.error(f"❌ Error saat preprocessing {short_id}: {str(e)}")
        return False