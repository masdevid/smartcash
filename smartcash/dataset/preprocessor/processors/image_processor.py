"""
File: smartcash/dataset/preprocessor/processors/image_processor.py
Deskripsi: Updated image processor dengan UUID konsisten dan format naming penelitian
"""

import cv2
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.common.utils.file_naming_manager import FileNamingManager
from smartcash.dataset.preprocessor.storage.preprocessing_pipeline_manager import PreprocessingPipelineManager

class ImageProcessor:
    """Updated image processor dengan UUID consistency dan research naming format"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self.pipeline = PreprocessingPipelineManager(config, logger)
        self.naming_manager = FileNamingManager(config)
        
    def process_single_image(self, image_path: Path, source_labels_dir: Path,
                           target_images_dir: Path, target_labels_dir: Path,
                           processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process dengan UUID konsisten dan research naming"""
        try:
            self.pipeline.set_options(**processing_config)
            
            # Parse atau generate file info dengan UUID konsisten
            original_name = image_path.name
            file_info = self.naming_manager.generate_file_info(original_name, stage='raw')
            
            # Detect class_id dari label untuk nominal determination
            class_id = self._extract_primary_class_from_label(image_path, source_labels_dir)
            if class_id:
                file_info = self.naming_manager.generate_file_info(original_name, class_id, 'raw')
            
            # Generate preprocessed filename
            preprocessed_info = self.naming_manager.generate_file_info(
                original_name, class_id, 'preprocessed'
            )
            target_filename = preprocessed_info.get_filename()
            target_image_path = target_images_dir / target_filename
            
            # Skip jika sudah ada dan tidak force reprocess
            if target_image_path.exists() and not processing_config.get('force_reprocess', False):
                return {'success': True, 'status': 'skipped', 'message': 'File sudah diproses'}
            
            # Load dan process image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'success': False, 'status': 'failed', 'message': 'Cannot load image'}
            
            processed_image = self.apply_preprocessing_pipeline(image, processing_config)
            if processed_image is None:
                return {'success': False, 'status': 'failed', 'message': 'Pipeline processing failed'}
            
            # Save dengan research naming format
            save_result = self.save_processed_image(processed_image, target_image_path)
            if not save_result['success']:
                return save_result
            
            # Handle label dengan UUID konsisten
            label_result = self._handle_label_file_with_uuid(
                image_path, source_labels_dir, target_labels_dir, 
                preprocessed_info, file_info
            )
            
            return {
                'success': True, 'status': 'processed',
                'target_path': str(target_image_path),
                'uuid': file_info.uuid, 'nominal': file_info.nominal,
                'label_handled': label_result['success'],
                'message': f'Processed: {target_filename}'
            }
            
        except Exception as e:
            return {'success': False, 'status': 'failed', 'message': f'Processing error: {str(e)}'}
    
    def apply_preprocessing_pipeline(self, image, processing_config: Dict[str, Any]):
        """Apply preprocessing dengan enhanced options"""
        try:
            pipeline_options = {
                'img_size': processing_config.get('img_size', [640, 640]),
                'normalize': processing_config.get('normalize', True),
                'preserve_aspect_ratio': processing_config.get('preserve_aspect_ratio', True)
            }
            
            self.pipeline.set_options(**pipeline_options)
            return self.pipeline.process(image)
            
        except Exception as e:
            self.logger.debug(f"🔧 Pipeline error: {str(e)}")
            return None
    
    def save_processed_image(self, processed_image, target_path: Path) -> Dict[str, Any]:
        """Save dengan high quality untuk penelitian"""
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Research quality settings
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            success = cv2.imwrite(str(target_path), processed_image, save_params)
            
            if not success:
                return {'success': False, 'message': 'Failed to save processed image'}
            
            return {'success': True, 'target_path': str(target_path)}
            
        except Exception as e:
            return {'success': False, 'message': f'Save error: {str(e)}'}
    
    def _extract_primary_class_from_label(self, image_path: Path, labels_dir: Path) -> Optional[str]:
        """Extract primary class dari label file untuk nominal determination"""
        try:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                return None
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Ambil class yang paling sering muncul
            class_counts = {}
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = str(int(float(parts[0])))
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    except (ValueError, IndexError):
                        continue
            
            if class_counts:
                return max(class_counts.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            self.logger.debug(f"🔍 Class extraction error: {str(e)}")
        
        return None
    
    def _handle_label_file_with_uuid(self, image_path: Path, source_labels_dir: Path,
                                   target_labels_dir: Path, preprocessed_info,
                                   original_info) -> Dict[str, Any]:
        """Handle label dengan UUID consistency"""
        try:
            source_label_path = source_labels_dir / f"{image_path.stem}.txt"
            
            if not source_label_path.exists():
                return {'success': True, 'status': 'no_label', 'message': 'No label file'}
            
            target_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Label filename follows same UUID pattern
            target_label_name = f"{Path(preprocessed_info.get_filename()).stem}.txt"
            target_label_path = target_labels_dir / target_label_name
            
            # Copy dengan validation
            shutil.copy2(source_label_path, target_label_path)
            
            return {
                'success': True, 'status': 'copied',
                'target_path': str(target_label_path),
                'uuid': original_info.uuid,
                'message': 'Label processed with UUID consistency'
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Label error: {str(e)}'}
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Status dengan UUID tracking info"""
        return {
            'processor_ready': True,
            'pipeline_available': self.pipeline is not None,
            'naming_manager_ready': self.naming_manager is not None,
            'uuid_registry_size': len(self.naming_manager.uuid_registry),
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
            'research_naming_active': True
        }
    
    def cleanup_processor_state(self) -> None:
        """Cleanup dengan UUID registry preservation"""
        if self.pipeline:
            self.pipeline.reset_pipeline_options()
        # Preserve UUID registry untuk consistency
        self.logger.debug("🧹 Processor cleaned, UUID registry preserved")