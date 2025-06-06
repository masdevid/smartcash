"""
File: smartcash/dataset/preprocessor/processors/image_processor.py
Deskripsi: Enhanced image processor dengan reuse dataset_file_renamer untuk eliminasi redundansi
"""

import cv2
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger
from smartcash.dataset.organizer.dataset_file_renamer import create_dataset_renamer
from smartcash.dataset.preprocessor.storage.preprocessing_pipeline_manager import PreprocessingPipelineManager


class ImageProcessor:
    """Enhanced image processor dengan UUID reuse dan eliminasi redundansi."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self.pipeline = PreprocessingPipelineManager(config, logger)
        # REUSE: Dataset renamer untuk UUID consistency
        self.renamer = create_dataset_renamer(config)
        
    def process_single_image(self, image_path: Path, source_labels_dir: Path,
                           target_images_dir: Path, target_labels_dir: Path,
                           processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process dengan UUID reuse - eliminasi duplikasi dengan dataset_file_renamer"""
        try:
            self.pipeline.set_options(**processing_config)
            
            # REUSE: UUID management dari dataset_file_renamer
            original_name = image_path.name
            existing_uuid = self.renamer.naming_manager.get_consistent_uuid(original_name)
            
            # Generate target filename dengan consistent UUID
            parsed = self.renamer.naming_manager.parse_existing_filename(original_name)
            if parsed:
                target_filename = f"pre_{parsed.get_filename()}"
            else:
                class_id = self._extract_primary_class_from_label(image_path, source_labels_dir)
                file_info = self.renamer.naming_manager.generate_file_info(original_name, class_id, 'preprocessed')
                target_filename = file_info.get_filename()
            
            target_image_path = target_images_dir / target_filename
            
            # Skip logic with force reprocess check
            if target_image_path.exists() and not processing_config.get('force_reprocess', False):
                return {'success': True, 'status': 'skipped', 'message': f'Already processed: {target_filename}'}
            
            # Process image
            processed_result = self._process_image_with_pipeline(image_path, target_image_path, processing_config)
            if not processed_result['success']:
                return processed_result
            
            # Handle label dengan UUID consistency
            label_result = self._handle_label_with_uuid_consistency(
                image_path, source_labels_dir, target_labels_dir, target_filename
            )
            
            return {
                'success': True, 'status': 'processed', 'target_path': str(target_image_path),
                'target_filename': target_filename, 'uuid': existing_uuid,
                'label_processed': label_result['success'], 'message': f'Processed: {target_filename}'
            }
            
        except Exception as e:
            return {'success': False, 'status': 'failed', 'message': f'Processing error: {str(e)}'}
    
    def batch_rename_then_process(self, source_split_dir: Path, target_split_dir: Path,
                                processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """ENHANCED: Batch rename kemudian process untuk consistency"""
        try:
            # Step 1: Rename files untuk UUID consistency
            rename_result = self.renamer.batch_rename_dataset(
                str(source_split_dir), backup=False, 
                progress_callback=lambda p, m: self.logger.debug(f"🔄 Rename: {m}")
            )
            
            if not rename_result.get('success', False):
                return {'success': False, 'message': f"Rename failed: {rename_result.get('message')}"}
            
            # Step 2: Process dengan UUID consistent files
            return self.batch_process_split_with_uuid_consistency(
                source_split_dir, target_split_dir, processing_config
            )
            
        except Exception as e:
            return {'success': False, 'message': f'Batch rename+process error: {str(e)}'}
    
    def batch_process_split_with_uuid_consistency(self, source_split_dir: Path, target_split_dir: Path,
                                                processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Batch process dengan UUID consistency - eliminasi duplikasi"""
        return batch_process_split(
            'preprocessed', str(source_split_dir), str(target_split_dir), 
            {**processing_config, 'processor': self}
        )
    
    def _process_image_with_pipeline(self, source_path: Path, target_path: Path, 
                                   processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process image dengan pipeline - kept minimal for reuse"""
        try:
            image = cv2.imread(str(source_path))
            if image is None: return {'success': False, 'message': f'Cannot load: {source_path.name}'}
            
            processed_image = self.pipeline.process(image, processing_config)
            if processed_image is None: return {'success': False, 'message': 'Pipeline failed'}
            
            target_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(target_path), processed_image, 
                                [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            
            return {'success': success, 'processed_path': str(target_path)} if success else \
                   {'success': False, 'message': 'Save failed'}
            
        except Exception as e:
            return {'success': False, 'message': f'Pipeline error: {str(e)}'}
    
    def _extract_primary_class_from_label(self, image_path: Path, labels_dir: Path) -> Optional[str]:
        """Extract primary class - simplified"""
        try:
            label_path = labels_dir / f"{image_path.stem}.txt"
            return self._parse_primary_class_from_file(label_path) if label_path.exists() else \
                   self._extract_class_from_filename(image_path.name)
        except Exception:
            return None
    
    def _parse_primary_class_from_file(self, label_path: Path) -> Optional[str]:
        """Parse class dari label - one-liner approach"""
        try:
            class_freq = {}
            with open(label_path, 'r') as f:
                [class_freq.update({str(int(float(parts[0]))): class_freq.get(str(int(float(parts[0]))), 0) + 1}) 
                 for line in f if (parts := line.strip().split()) and len(parts) >= 5]
            return max(class_freq.items(), key=lambda x: x[1])[0] if class_freq else None
        except Exception:
            return None
    
    def _extract_class_from_filename(self, filename: str) -> Optional[str]:
        """Extract class dari filename - one-liner mapping"""
        denomination_map = {'1k': '0', '2k': '1', '5k': '2', '10k': '3', '20k': '4', '50k': '5', '100k': '6'}
        filename_lower = filename.lower()
        return next((cid for pattern, cid in denomination_map.items() if pattern in filename_lower), None)
    
    def _handle_label_with_uuid_consistency(self, image_path: Path, source_labels_dir: Path,
                                          target_labels_dir: Path, target_filename: str) -> Dict[str, Any]:
        """Handle label dengan UUID consistency - simplified"""
        try:
            source_label = source_labels_dir / f"{image_path.stem}.txt"
            if not source_label.exists():
                return {'success': True, 'status': 'no_label'}
            
            target_labels_dir.mkdir(parents=True, exist_ok=True)
            target_label = target_labels_dir / f"{Path(target_filename).stem}.txt"
            
            # Copy dengan validation
            return self._copy_and_validate_label(source_label, target_label)
            
        except Exception as e:
            return {'success': False, 'message': f'Label error: {str(e)}'}
    
    def _copy_and_validate_label(self, source_path: Path, target_path: Path) -> Dict[str, Any]:
        """Copy label dengan validation - one-liner approach"""
        try:
            valid_lines = []
            with open(source_path, 'r') as f:
                for line in f:
                    if (parts := line.strip().split()) and len(parts) >= 5:
                        try:
                            class_id, coords = int(float(parts[0])), [float(x) for x in parts[1:5]]
                            if all(0.0 <= x <= 1.0 for x in coords) and all(w > 0.001 for w in coords[2:4]):
                                valid_lines.append(f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                        except (ValueError, IndexError):
                            continue
            
            with open(target_path, 'w') as f: f.writelines(valid_lines)
            return {'success': True, 'valid_count': len(valid_lines)}
            
        except Exception:
            # Fallback: simple copy
            try:
                shutil.copy2(source_path, target_path)
                return {'success': True, 'fallback': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Enhanced status dengan renamer integration"""
        return {
            'processor_ready': True, 'pipeline_available': self.pipeline is not None,
            'renamer_integrated': self.renamer is not None,
            'uuid_registry_size': len(self.renamer.naming_manager.uuid_registry),
            'supported_formats': ['.jpg', '.jpeg', '.png', '.bmp'],
            'uuid_consistency_enabled': True, 'redundancy_eliminated': True
        }
    
    def cleanup_processor_state(self) -> None:
        """Cleanup dengan UUID registry preservation"""
        self.pipeline and self.pipeline.reset_pipeline_options()
        self.logger.debug("🧹 Processor cleaned, UUID registry preserved")


# REUSE: One-liner utilities dengan existing implementations
create_uuid_processor = lambda config: ImageProcessor(config)
process_with_uuid_reuse = lambda img_path, src_labels, tgt_imgs, tgt_labels, config: create_uuid_processor(config).process_single_image(Path(img_path), Path(src_labels), Path(tgt_imgs), Path(tgt_labels), config)
batch_process_split = lambda stage, src_dir, tgt_dir, config: create_uuid_processor(config['processor'].config if 'processor' in config else config).batch_process_split_with_uuid_consistency(Path(src_dir), Path(tgt_dir), config)
batch_rename_process = lambda src_dir, tgt_dir, config: create_uuid_processor(config).batch_rename_then_process(Path(src_dir), Path(tgt_dir), config)