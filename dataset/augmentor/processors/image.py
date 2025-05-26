"""
File: smartcash/dataset/augmentor/processors/image.py
Deskripsi: Image processing operations dengan one-liner optimized untuk augmentasi dataset
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from smartcash.common.logger import get_logger

# One-liner helper functions
read_image = lambda path: cv2.imread(str(path)) if Path(path).exists() else None
save_image = lambda img, path: cv2.imwrite(str(path), img) if img is not None else False
apply_pipeline = lambda img, pipeline: pipeline(image=img)['image'] if img is not None else None
validate_image = lambda img: img is not None and img.size > 0
get_image_shape = lambda img: img.shape[:2] if validate_image(img) else (0, 0)
resize_image = lambda img, size: cv2.resize(img, size) if validate_image(img) else None

class ImageProcessor:
    """Processor untuk operasi gambar dengan optimized one-liner operations."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi ImageProcessor.
        
        Args:
            logger: Logger untuk logging operations
        """
        self.logger = logger or get_logger(__name__)
        self.processed_count = 0
    
    def read_and_validate(self, image_path: str) -> Optional[np.ndarray]:
        """
        Baca dan validasi gambar dengan one-liner.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Array gambar atau None jika gagal
        """
        try:
            image = read_image(image_path)
            return image if validate_image(image) else None
        except Exception as e:
            self.logger.warning(f"⚠️ Error membaca gambar {image_path}: {str(e)}")
            return None
    
    def process_single_image(
        self, 
        image_path: str, 
        pipeline, 
        output_path: str,
        variations: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Proses single image dengan multiple variations.
        
        Args:
            image_path: Path gambar input
            pipeline: Pipeline augmentasi
            output_path: Path output base
            variations: Jumlah variasi
            
        Returns:
            List hasil processing per variasi
        """
        # Read and validate dengan error handling
        image = self.read_and_validate(image_path)
        if image is None:
            return [{'status': 'error', 'message': f'Gagal membaca gambar {image_path}'}]
        
        results = []
        image_name = Path(image_path).stem
        
        # Process multiple variations dengan optimized loop
        for var_idx in range(variations):
            try:
                # Apply pipeline dengan error handling
                augmented_image = apply_pipeline(image, pipeline)
                if augmented_image is None:
                    augmented_image = image.copy()  # Fallback ke original
                
                # Generate output path dengan variation suffix
                var_output_path = f"{output_path}_{image_name}_var{var_idx+1}.jpg"
                
                # Save dengan validation
                save_success = save_image(augmented_image, var_output_path)
                
                results.append({
                    'status': 'success' if save_success else 'error',
                    'input_path': image_path,
                    'output_path': var_output_path if save_success else None,
                    'variation': var_idx + 1,
                    'original_shape': get_image_shape(image),
                    'augmented_shape': get_image_shape(augmented_image)
                })
                
            except Exception as e:
                results.append({
                    'status': 'error',
                    'message': f'Error variasi {var_idx+1}: {str(e)}',
                    'variation': var_idx + 1
                })
        
        # Update counter dan log
        self.processed_count += 1
        success_count = sum(1 for r in results if r.get('status') == 'success')
        
        if success_count > 0:
            self.logger.debug(f"🖼️ Processed {image_name}: {success_count}/{variations} variasi berhasil")
        
        return results
    
    def batch_resize(
        self, 
        images: List[np.ndarray], 
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """
        Batch resize dengan one-liner optimization.
        
        Args:
            images: List gambar
            target_size: Target size (width, height)
            
        Returns:
            List gambar yang sudah di-resize
        """
        return [resize_image(img, target_size) for img in images if validate_image(img)]
    
    def apply_augmentation_pipeline(
        self, 
        image: np.ndarray, 
        pipeline,
        safe_mode: bool = True
    ) -> np.ndarray:
        """
        Apply augmentation pipeline dengan safe mode.
        
        Args:
            image: Input image
            pipeline: Augmentation pipeline
            safe_mode: Gunakan fallback jika error
            
        Returns:
            Augmented image
        """
        try:
            # Apply pipeline dengan class_labels empty untuk compatibility
            result = pipeline(image=image, class_labels=[])
            return result['image']
        except Exception as e:
            if safe_mode:
                self.logger.warning(f"⚠️ Pipeline error, menggunakan original image: {str(e)}")
                return image.copy()
            else:
                raise e
    
    def get_processing_stats(self) -> Dict[str, int]:
        """
        Dapatkan statistik processing.
        
        Returns:
            Dictionary statistik
        """
        return {
            'processed_images': self.processed_count,
            'timestamp': int(__import__('time').time())
        }
    
    def reset_stats(self) -> None:
        """Reset counter statistik."""
        self.processed_count = 0
    
    @staticmethod
    def validate_pipeline(pipeline) -> bool:
        """
        Validasi pipeline augmentasi.
        
        Args:
            pipeline: Pipeline yang akan divalidasi
            
        Returns:
            Boolean validasi
        """
        try:
            # Test dengan dummy image
            dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = pipeline(image=dummy_image, class_labels=[])
            return 'image' in result and result['image'] is not None
        except Exception:
            return False
    
    @staticmethod
    def create_dummy_image(width: int = 100, height: int = 100, channels: int = 3) -> np.ndarray:
        """
        Buat dummy image untuk testing.
        
        Args:
            width: Lebar image
            height: Tinggi image
            channels: Jumlah channel
            
        Returns:
            Dummy image array
        """
        return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ImageProcessor(processed={self.processed_count})"