"""
File: smartcash/dataset/preprocessor/processors/image_processor.py
Deskripsi: Pure image processor untuk single image processing dengan pipeline integration
"""

import cv2
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.storage.preprocessing_pipeline_manager import PreprocessingPipelineManager


class ImageProcessor:
    """Pure image processor untuk single image dengan preprocessing pipeline integration."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize image processor dengan pipeline dan file handling."""
        self.config = config
        self.logger = logger or get_logger()
        
        # Initialize preprocessing pipeline
        self.pipeline = PreprocessingPipelineManager(config, logger)
        
        # File naming configuration
        self.file_prefix = config.get('preprocessing', {}).get('file_prefix', 'rp')
        
    def process_single_image(self, image_path: Path, source_labels_dir: Path,
                           target_images_dir: Path, target_labels_dir: Path,
                           processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process single image dengan comprehensive pipeline dan file handling.
        
        Args:
            image_path: Path ke image file
            source_labels_dir: Directory labels source
            target_images_dir: Directory target images
            target_labels_dir: Directory target labels
            processing_config: Configuration preprocessing
            
        Returns:
            Dictionary hasil processing
        """
        try:
            # Update pipeline options dengan config
            self.pipeline.set_options(**processing_config)
            
            # Generate target filename
            target_filename = self._generate_target_filename(image_path)
            target_image_path = target_images_dir / f"{target_filename}.jpg"
            
            # Skip jika target sudah ada (tidak force reprocess)
            if target_image_path.exists() and not processing_config.get('force_reprocess', False):
                return {'success': True, 'status': 'skipped', 'message': 'File already exists'}
            
            # Load dan validate image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'success': False, 'status': 'failed', 'message': 'Cannot load image'}
            
            # Apply preprocessing pipeline
            processed_image = self.apply_preprocessing_pipeline(image, processing_config)
            if processed_image is None:
                return {'success': False, 'status': 'failed', 'message': 'Pipeline processing failed'}
            
            # Save processed image
            save_result = self.save_processed_image(processed_image, target_image_path)
            if not save_result['success']:
                return save_result
            
            # Handle label file jika ada
            label_result = self._handle_label_file(
                image_path, source_labels_dir, target_labels_dir, target_filename
            )
            
            return {
                'success': True,
                'status': 'processed',
                'target_path': str(target_image_path),
                'label_handled': label_result['success'],
                'message': 'Image processed successfully'
            }
            
        except Exception as e:
            return {'success': False, 'status': 'failed', 'message': f'Processing error: {str(e)}'}
    
    def apply_preprocessing_pipeline(self, image, processing_config: Dict[str, Any]):
        """Apply preprocessing pipeline ke image dengan error handling."""
        try:
            # Update pipeline dengan current config
            pipeline_options = {
                'img_size': processing_config.get('img_size', [640, 640]),
                'normalize': processing_config.get('normalize', True),
                'preserve_aspect_ratio': processing_config.get('preserve_aspect_ratio', True)
            }
            
            self.pipeline.set_options(**pipeline_options)
            
            # Process image through pipeline
            processed_image = self.pipeline.process(image)
            
            return processed_image
            
        except Exception as e:
            self.logger.debug(f"🔧 Pipeline processing error: {str(e)}")
            return None
    
    def save_processed_image(self, processed_image, target_path: Path) -> Dict[str, Any]:
        """Save processed image dengan quality optimization."""
        try:
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save dengan optimized quality
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # High quality JPEG
            success = cv2.imwrite(str(target_path), processed_image, save_params)
            
            if not success:
                return {'success': False, 'message': 'Failed to save image'}
            
            return {'success': True, 'target_path': str(target_path)}
            
        except Exception as e:
            return {'success': False, 'message': f'Save error: {str(e)}'}
    
    def _generate_target_filename(self, source_path: Path) -> str:
        """Generate target filename dengan prefix dan normalization."""
        # Remove source extension dan add prefix
        base_name = source_path.stem
        
        # Sanitize filename (remove special characters)
        sanitized_name = ''.join(c for c in base_name if c.isalnum() or c in '-_')
        
        # Add prefix jika configured
        if self.file_prefix:
            return f"{self.file_prefix}_{sanitized_name}"
        else:
            return sanitized_name
    
    def _handle_label_file(self, image_path: Path, source_labels_dir: Path,
                         target_labels_dir: Path, target_filename: str) -> Dict[str, Any]:
        """Handle corresponding label file untuk image."""
        try:
            # Find corresponding label file
            label_name = image_path.stem + '.txt'
            source_label_path = source_labels_dir / label_name
            
            if not source_label_path.exists():
                return {'success': True, 'status': 'no_label', 'message': 'No corresponding label file'}
            
            # Ensure target labels directory exists
            target_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy label file ke target dengan new name
            target_label_path = target_labels_dir / f"{target_filename}.txt"
            shutil.copy2(source_label_path, target_label_path)
            
            return {
                'success': True,
                'status': 'copied',
                'target_path': str(target_label_path),
                'message': 'Label file copied successfully'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Label handling error: {str(e)}'}
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Dapatkan status image processor."""
        return {
            'processor_ready': True,
            'pipeline_available': self.pipeline is not None,
            'file_prefix': self.file_prefix,
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        }
    
    def cleanup_processor_state(self) -> None:
        """Cleanup processor state dan resources."""
        # Reset pipeline options ke default
        if self.pipeline:
            self.pipeline.set_options(
                img_size=[640, 640],
                normalize=True,
                preserve_aspect_ratio=True
            )
        self.logger.debug("🧹 Image processor state cleaned up")