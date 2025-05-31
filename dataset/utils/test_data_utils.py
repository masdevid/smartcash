"""
File: smartcash/dataset/utils/test_data_utils.py
Deskripsi: Utilitas untuk mempersiapkan data test dengan augmentasi untuk evaluasi model
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from torch.utils.data import DataLoader

def prepare_test_data_for_scenario(
    test_folder: str, 
    scenario_info: Dict[str, Any], 
    augmentation_pipeline: Any = None, 
    batch_size: int = 8, 
    img_size: int = 416
) -> Dict[str, Any]:
    """
    Mempersiapkan data test dengan augmentasi untuk skenario evaluasi tertentu
    
    Args:
        test_folder: Path folder test data
        scenario_info: Informasi skenario evaluasi
        augmentation_pipeline: Pipeline augmentasi untuk skenario
        batch_size: Ukuran batch untuk dataloader
        img_size: Ukuran gambar untuk preprocessing
        
    Returns:
        Dict berisi dataloader dan metadata test data
    """
    try:
        # Validasi folder test
        if not os.path.exists(test_folder):
            return {'success': False, 'error': f"Folder test tidak ditemukan: {test_folder}"}
        
        # Cari folder images
        images_folder = os.path.join(test_folder, 'images')
        if not os.path.exists(images_folder):
            # Coba gunakan folder test langsung jika tidak ada subfolder images
            images_folder = test_folder
        
        # Cari file gambar
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(Path(images_folder).glob(f"*{ext}")))
        
        if not image_files:
            return {'success': False, 'error': f"Tidak ada gambar di folder: {images_folder}"}
        
        # Muat gambar dari file paths
        images = []
        image_paths = []
        
        for img_file in image_files:
            try:
                img_path = str(img_file)
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    image_paths.append(img_path)
            except Exception as e:
                print(f"Error loading image {img_file}: {str(e)}")
        
        if not images:
            return {'success': False, 'error': "Gagal memuat gambar dari folder test"}
        
        # Buat dataloader untuk inference
        from smartcash.model.utils.evaluation_utils import create_inference_dataloader
        
        dataloader = create_inference_dataloader(
            images=images,
            image_paths=image_paths,
            augmentation_pipeline=augmentation_pipeline,
            batch_size=batch_size,
            img_size=img_size
        )
        
        return {
            'success': True,
            'dataloader': dataloader,
            'image_files': image_files,
            'count': len(images),
            'scenario_info': scenario_info
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
