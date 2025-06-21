"""
File: smartcash/dataset/augmentor/strategies/evaluation_augmentation.py
Deskripsi: Strategi augmentasi untuk evaluasi model dengan variasi posisi dan pencahayaan
"""

import albumentations as A
import numpy as np
from typing import Dict, Any, List, Callable, Optional

class EvaluationAugmentationStrategy:
    """Strategi augmentasi untuk evaluasi model dengan berbagai skenario pengujian"""
    
    @staticmethod
    def create_position_augmentation_pipeline() -> Callable:
        """
        Membuat pipeline augmentasi untuk variasi posisi pengambilan gambar
        
        Returns:
            Fungsi augmentasi untuk variasi posisi
        """
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-15, 15), shear=(-10, 10), p=0.7),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        def augment_image(image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]) -> Dict[str, Any]:
            """Fungsi untuk mengaugmentasi gambar dengan variasi posisi"""
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                return {
                    'image': transformed['image'],
                    'bboxes': transformed['bboxes'],
                    'class_labels': transformed['class_labels'],
                    'augmentation_type': 'position'
                }
            except Exception as e:
                # Jika augmentasi gagal, kembalikan data asli
                return {
                    'image': image,
                    'bboxes': bboxes,
                    'class_labels': class_labels,
                    'augmentation_type': 'none',
                    'error': str(e)
                }
        
        return augment_image
    
    @staticmethod
    def create_lighting_augmentation_pipeline() -> Callable:
        """
        Membuat pipeline augmentasi untuk variasi pencahayaan
        
        Returns:
            Fungsi augmentasi untuk variasi pencahayaan
        """
        transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        def augment_image(image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]) -> Dict[str, Any]:
            """Fungsi untuk mengaugmentasi gambar dengan variasi pencahayaan"""
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                return {
                    'image': transformed['image'],
                    'bboxes': transformed['bboxes'],
                    'class_labels': transformed['class_labels'],
                    'augmentation_type': 'lighting'
                }
            except Exception as e:
                # Jika augmentasi gagal, kembalikan data asli
                return {
                    'image': image,
                    'bboxes': bboxes,
                    'class_labels': class_labels,
                    'augmentation_type': 'none',
                    'error': str(e)
                }
        
        return augment_image
    
    @staticmethod
    def create_default_augmentation_pipeline() -> Callable:
        """
        Membuat pipeline augmentasi default (minimal)
        
        Returns:
            Fungsi augmentasi default
        """
        transform = A.Compose([
            # Minimal augmentasi untuk stabilitas
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        def augment_image(image: np.ndarray, bboxes: List[List[float]], class_labels: List[int]) -> Dict[str, Any]:
            """Fungsi untuk mengaugmentasi gambar dengan augmentasi default"""
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                return {
                    'image': transformed['image'],
                    'bboxes': transformed['bboxes'],
                    'class_labels': transformed['class_labels'],
                    'augmentation_type': 'default'
                }
            except Exception as e:
                # Jika augmentasi gagal, kembalikan data asli
                return {
                    'image': image,
                    'bboxes': bboxes,
                    'class_labels': class_labels,
                    'augmentation_type': 'none',
                    'error': str(e)
                }
        
        return augment_image
    
    @staticmethod
    def get_augmentation_pipeline(scenario_info: Dict[str, Any]) -> Callable:
        """
        Membuat pipeline augmentasi berdasarkan tipe skenario
        
        Args:
            scenario_info: Informasi skenario yang dipilih
            
        Returns:
            Fungsi augmentasi yang akan diterapkan pada gambar
        """
        augmentation_type = scenario_info.get('augmentation_type', 'position')
        
        if augmentation_type == 'position':
            return EvaluationAugmentationStrategy.create_position_augmentation_pipeline()
        elif augmentation_type == 'lighting':
            return EvaluationAugmentationStrategy.create_lighting_augmentation_pipeline()
        else:
            # Default augmentation (minimal)
            return EvaluationAugmentationStrategy.create_default_augmentation_pipeline()
    
    @staticmethod
    def apply_augmentation_to_batch(images: List[np.ndarray], 
                                   bboxes: List[List[List[float]]], 
                                   class_labels: List[List[int]], 
                                   aug_pipeline: Callable,
                                   max_workers: int = 4,
                                   logger=None) -> Dict[str, Any]:
        """
        Terapkan augmentasi pada batch gambar secara paralel
        
        Args:
            images: List gambar untuk diaugmentasi
            bboxes: List bounding boxes untuk setiap gambar
            class_labels: List class labels untuk setiap gambar
            aug_pipeline: Pipeline augmentasi yang akan digunakan
            max_workers: Jumlah maksimum worker untuk paralelisasi
            logger: Logger untuk mencatat proses
            
        Returns:
            Dict berisi hasil augmentasi
        """
        from concurrent.futures import ThreadPoolExecutor
        import traceback
        
        # Fungsi untuk mengaugmentasi satu gambar
        def augment_single_image(args):
            idx, image, img_bboxes, img_class_labels = args
            try:
                result = aug_pipeline(image=image, bboxes=img_bboxes, class_labels=img_class_labels)
                return idx, result, None
            except Exception as e:
                error_msg = f"Error augmentasi gambar {idx}: {str(e)}\n{traceback.format_exc()}"
                return idx, None, error_msg
        
        # Buat argumen untuk setiap gambar
        augmentation_args = [(i, img, box, cls) for i, (img, box, cls) in enumerate(zip(images, bboxes, class_labels))]
        
        # Augmentasi secara paralel
        augmented_results = [None] * len(images)
        errors = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, result, error in executor.map(augment_single_image, augmentation_args):
                if error:
                    errors.append(error)
                    # Gunakan gambar asli jika error
                    augmented_results[idx] = {
                        'image': images[idx],
                        'bboxes': bboxes[idx],
                        'class_labels': class_labels[idx],
                        'augmentation_type': 'none',
                        'error': error
                    }
                else:
                    augmented_results[idx] = result
        
        # Log errors jika ada
        if errors and logger:
            for error in errors[:5]:  # Batasi jumlah error yang ditampilkan
                logger.warning(f"⚠️ {error}")
            if len(errors) > 5:
                logger.warning(f"⚠️ ...dan {len(errors) - 5} error lainnya")
        
        # Ekstrak hasil augmentasi
        augmented_images = [result['image'] for result in augmented_results]
        augmented_bboxes = [result['bboxes'] for result in augmented_results]
        augmented_class_labels = [result['class_labels'] for result in augmented_results]
        
        if logger:
            logger.info(f"✅ Augmentasi selesai: {len(augmented_images)} gambar diproses")
        
        return {
            'images': augmented_images,
            'bboxes': augmented_bboxes,
            'class_labels': augmented_class_labels,
            'errors': errors
        }


def get_augmentation_pipeline(scenario_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Membuat pipeline augmentasi berdasarkan skenario evaluasi
    
    Args:
        scenario_id: ID skenario evaluasi
        config: Konfigurasi evaluasi
        
    Returns:
        Dict berisi pipeline augmentasi dan metadata
    """
    try:
        # Dapatkan informasi skenario
        from smartcash.model.utils.scenario_utils import get_scenario_info
        
        scenario_info = get_scenario_info(scenario_id, config)
        
        if not scenario_info:
            # Gunakan default jika skenario tidak ditemukan
            return {
                'success': True,
                'pipeline': EvaluationAugmentationStrategy.create_default_augmentation_pipeline(),
                'type': 'default',
                'message': f"Menggunakan augmentasi default untuk skenario {scenario_id}"
            }
        
        # Dapatkan tipe augmentasi dari skenario
        augmentation_type = scenario_info.get('augmentation_type', 'default')
        
        # Buat pipeline berdasarkan tipe
        if augmentation_type == 'position':
            pipeline = EvaluationAugmentationStrategy.create_position_augmentation_pipeline()
            aug_type = 'position'
        elif augmentation_type == 'lighting':
            pipeline = EvaluationAugmentationStrategy.create_lighting_augmentation_pipeline()
            aug_type = 'lighting'
        else:
            pipeline = EvaluationAugmentationStrategy.create_default_augmentation_pipeline()
            aug_type = 'default'
        
        return {
            'success': True,
            'pipeline': pipeline,
            'type': aug_type,
            'scenario_info': scenario_info
        }
        
    except Exception as e:
        # Fallback ke default jika terjadi error
        return {
            'success': False,
            'pipeline': EvaluationAugmentationStrategy.create_default_augmentation_pipeline(),
            'type': 'default',
            'error': str(e),
            'message': f"Error saat membuat pipeline augmentasi: {str(e)}"
        }
