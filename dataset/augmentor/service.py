"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Fixed main orchestrator service dengan smart directory detection dan proper error handling
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count
from .communicator import UICommunicator, create_communicator
from .config import create_aug_config, extract_ui_config
from .types import AugmentationResult, ProcessingStats, AugConfig
from .core.engine import AugmentationEngine
from .core.normalizer import NormalizationEngine
from .utils.cleaner import AugmentedDataCleaner
from .utils.dataset_detector import detect_dataset_structure

class AugmentationService:
    """Fixed main orchestrator service dengan smart directory detection dan flow yang benar"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        """Initialize augmentation service dengan UI communication bridge."""
        self.config = config
        self.aug_config = create_aug_config(config)
        self.ui_components = ui_components or {}
        self.comm = create_communicator(ui_components)
        
        # Initialize engines
        self.engine = AugmentationEngine(config, self.comm)
        self.normalizer = NormalizationEngine(config, self.comm)
        self.cleaner = AugmentedDataCleaner(config, self.comm)
        
        self.logger = self.comm.logger
        self.stats = defaultdict(int)
        
    def augment_raw_dataset(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Fixed augmentasi dataset dengan smart directory detection.
        Flow: /data → /data/augmented
        """
        operation_name = "Augmentasi Dataset"
        self.comm.start_operation(operation_name)
        
        try:
            # Smart directory detection
            raw_dir = self.aug_config.raw_dir
            detection_result = detect_dataset_structure(raw_dir)
            
            if detection_result['status'] == 'error':
                error_msg = f"Raw directory tidak valid: {detection_result['message']}"
                self.comm.error_operation(operation_name, error_msg)
                return self._create_error_result(error_msg)
            
            # Validate dataset structure
            if detection_result['total_images'] == 0:
                error_msg = f"Tidak ada gambar ditemukan di {raw_dir}. Struktur: {detection_result['structure_type']}"
                self.comm.error_operation(operation_name, error_msg)
                return self._create_error_result(error_msg)
            
            self.logger.info(f"🔍 Dataset terdeteksi: {detection_result['structure_type']}, {detection_result['total_images']} gambar")
            
            # Setup progress callback bridge
            if progress_callback:
                self._setup_progress_bridge(progress_callback)
            
            # Execute augmentation dengan engine
            aug_result = self.engine.process_raw_data(raw_dir, self.aug_config.aug_dir)
            
            if aug_result['status'] == 'success':
                success_msg = f"Augmentasi berhasil: {aug_result['total_generated']} gambar dihasilkan"
                self.comm.complete_operation(operation_name, success_msg)
                
                # Add dataset info ke result
                aug_result['dataset_info'] = detection_result
                return aug_result
            else:
                self.comm.error_operation(operation_name, aug_result.get('message', 'Error tidak diketahui'))
                return aug_result
                
        except Exception as e:
            error_msg = f"Error pada augmentasi raw dataset: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.comm.error_operation(operation_name, error_msg)
            return self._create_error_result(error_msg)
    
    def normalize_augmented_dataset(self, target_split: str = "train", 
                                  progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Fixed normalisasi dataset augmented ke preprocessed.
        Flow: /data/augmented → /data/preprocessed/{split}
        """
        operation_name = f"Normalisasi ke Split {target_split}"
        self.comm.start_operation(operation_name)
        
        try:
            # Validate augmented directory
            aug_dir = self.aug_config.aug_dir
            if not os.path.exists(aug_dir):
                error_msg = f"Augmented directory tidak ditemukan: {aug_dir}"
                self.comm.error_operation(operation_name, error_msg)
                return self._create_error_result(error_msg)
            
            # Check augmented files
            aug_detection = detect_dataset_structure(aug_dir)
            if aug_detection['total_images'] == 0:
                error_msg = f"Tidak ada file augmented ditemukan di {aug_dir}"
                self.comm.error_operation(operation_name, error_msg)
                return self._create_error_result(error_msg)
            
            self.logger.info(f"🔄 Memulai normalisasi: {aug_dir} → preprocessed/{target_split}")
            self.logger.info(f"📊 File augmented: {aug_detection['total_images']} gambar, {aug_detection['total_labels']} label")
            
            # Setup progress callback bridge
            if progress_callback:
                self._setup_progress_bridge(progress_callback)
            
            # Execute normalization dengan normalizer engine
            norm_result = self.normalizer.normalize_augmented_data(
                aug_dir, self.aug_config.prep_dir, target_split
            )
            
            if norm_result['status'] == 'success':
                success_msg = f"Normalisasi berhasil: {norm_result['total_normalized']} file dinormalisasi"
                self.comm.complete_operation(operation_name, success_msg)
                
                # Add detection info
                norm_result['augmented_info'] = aug_detection
                return norm_result
            else:
                self.comm.error_operation(operation_name, norm_result.get('message', 'Error tidak diketahui'))
                return norm_result
                
        except Exception as e:
            error_msg = f"Error pada normalisasi dataset: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.comm.error_operation(operation_name, error_msg)
            return self._create_error_result(error_msg)
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", 
                                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Fixed full pipeline augmentasi dengan smart detection dan validation.
        """
        pipeline_name = "Full Augmentation Pipeline"
        self.comm.start_operation(pipeline_name)
        
        start_time = time.time()
        results = {}
        
        try:
            # Pre-flight validation
            raw_dir = self.aug_config.raw_dir
            detection_result = detect_dataset_structure(raw_dir)
            
            if detection_result['status'] == 'error' or detection_result['total_images'] == 0:
                error_msg = f"Dataset tidak valid untuk augmentasi: {detection_result.get('message', 'Tidak ada gambar ditemukan')}"
                self.comm.error_operation(pipeline_name, error_msg)
                return self._create_pipeline_error_result(error_msg, results)
            
            self.logger.info(f"🚀 Pipeline dimulai untuk dataset: {detection_result['structure_type']}")
            self.logger.info(f"📊 Input: {detection_result['total_images']} gambar, {detection_result['total_labels']} label")
            
            # Step 1: Augmentasi raw dataset
            self.logger.info("📊 Step 1/2: Augmentasi raw dataset")
            self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 10, 100, "Step 1: Augmentasi raw dataset")
            
            aug_result = self.augment_raw_dataset(progress_callback)
            results['augmentation'] = aug_result
            
            if aug_result['status'] != 'success':
                error_msg = f"Pipeline gagal pada step augmentasi: {aug_result.get('message')}"
                self.comm.error_operation(pipeline_name, error_msg)
                return self._create_pipeline_error_result(error_msg, results)
            
            # Step 2: Normalisasi ke preprocessed
            self.logger.info("📊 Step 2/2: Normalisasi ke preprocessed")
            self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 60, 100, "Step 2: Normalisasi ke preprocessed")
            
            norm_result = self.normalize_augmented_dataset(target_split, progress_callback)
            results['normalization'] = norm_result
            
            if norm_result['status'] != 'success':
                error_msg = f"Pipeline gagal pada step normalisasi: {norm_result.get('message')}"
                self.comm.error_operation(pipeline_name, error_msg)
                return self._create_pipeline_error_result(error_msg, results)
            
            # Pipeline success summary
            total_time = time.time() - start_time
            pipeline_summary = self._create_pipeline_success_result(results, total_time, target_split, detection_result)
            
            success_msg = f"Pipeline selesai: {pipeline_summary['total_files']} file → {pipeline_summary['final_output']}"
            self.comm.complete_operation(pipeline_name, success_msg)
            
            return pipeline_summary
            
        except Exception as e:
            error_msg = f"Error pada full pipeline: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.comm.error_operation(pipeline_name, error_msg)
            return self._create_pipeline_error_result(error_msg, results)
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, 
                             progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Fixed cleanup dengan progress tracking."""
        operation_name = "Cleanup Augmented Data"
        self.comm.start_operation(operation_name)
        
        try:
            # Setup progress callback bridge
            if progress_callback:
                self._setup_progress_bridge(progress_callback)
            
            # Execute cleanup dengan cleaner
            cleanup_result = self.cleaner.cleanup_all_augmented_files(include_preprocessed)
            
            if cleanup_result['status'] == 'success':
                success_msg = f"Cleanup berhasil: {cleanup_result['total_deleted']} file dihapus"
                self.comm.complete_operation(operation_name, success_msg)
                return cleanup_result
            else:
                self.comm.error_operation(operation_name, cleanup_result.get('message', 'Error cleanup'))
                return cleanup_result
                
        except Exception as e:
            error_msg = f"Error pada cleanup: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.comm.error_operation(operation_name, error_msg)
            return self._create_error_result(error_msg)
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Get status dengan dataset structure detection."""
        aug_dir = self.aug_config.aug_dir
        prep_dir = self.aug_config.prep_dir
        raw_dir = self.aug_config.raw_dir
        
        # Detect raw dataset structure
        raw_detection = detect_dataset_structure(raw_dir)
        
        # Detect augmented files
        aug_detection = detect_dataset_structure(aug_dir) if os.path.exists(aug_dir) else {'total_images': 0, 'total_labels': 0}
        
        # Count preprocessed files (train split)
        preprocessed_files = 0
        train_dir = os.path.join(prep_dir, 'train', 'images')
        if os.path.exists(train_dir):
            preprocessed_files = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))])
        
        status = {
            'raw_dataset': {
                'exists': raw_detection['status'] == 'success',
                'structure_type': raw_detection.get('structure_type', 'unknown'),
                'total_images': raw_detection.get('total_images', 0),
                'total_labels': raw_detection.get('total_labels', 0),
                'recommendations': raw_detection.get('recommendations', [])
            },
            'augmented_dataset': {
                'exists': os.path.exists(aug_dir),
                'total_images': aug_detection.get('total_images', 0),
                'total_labels': aug_detection.get('total_labels', 0)
            },
            'preprocessed_dataset': {
                'exists': os.path.exists(prep_dir),
                'total_files': preprocessed_files
            },
            'ready_for_augmentation': (
                raw_detection['status'] == 'success' and 
                raw_detection.get('total_images', 0) > 0
            )
        }
        
        return status
    
    def _setup_progress_bridge(self, progress_callback: callable) -> None:
        """Setup bridge untuk external progress callback."""
        if hasattr(self.comm, 'report_progress_with_callback'):
            self.comm.report_progress_with_callback = progress_callback
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {'status': 'error', 'message': error_message, 'timestamp': time.time()}
    
    def _create_pipeline_success_result(self, results: Dict, total_time: float, target_split: str, detection_result: Dict) -> Dict[str, Any]:
        """Create enhanced pipeline success result."""
        aug_result = results.get('augmentation', {})
        norm_result = results.get('normalization', {})
        
        return {
            'status': 'success',
            'pipeline': 'full_augmentation',
            'total_time': total_time,
            'target_split': target_split,
            'total_files': aug_result.get('total_generated', 0),
            'final_output': f"{norm_result.get('target_dir', 'preprocessed')}/{target_split}",
            'input_dataset': {
                'structure': detection_result.get('structure_type', 'unknown'),
                'original_images': detection_result.get('total_images', 0),
                'original_labels': detection_result.get('total_labels', 0)
            },
            'steps': {
                'augmentation': {
                    'status': aug_result.get('status'),
                    'generated': aug_result.get('total_generated', 0),
                    'time': aug_result.get('processing_time', 0)
                },
                'normalization': {
                    'status': norm_result.get('status'),
                    'normalized': norm_result.get('total_normalized', 0),
                    'time': norm_result.get('processing_time', 0)
                }
            }
        }
    
    def _create_pipeline_error_result(self, error_message: str, partial_results: Dict) -> Dict[str, Any]:
        """Create pipeline error result dengan partial results."""
        return {
            'status': 'error',
            'message': error_message,
            'partial_results': partial_results,
            'timestamp': time.time()
        }

# Factory functions untuk service creation
def create_augmentation_service(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> AugmentationService:
    """Factory function untuk create augmentation service."""
    return AugmentationService(config, ui_components)

def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    """Factory function untuk create service dari UI components."""
    config = extract_ui_config(ui_components)
    return AugmentationService(config, ui_components)

# One-liner service operations dengan smart detection
augment_raw_data = lambda config, ui_components=None: create_augmentation_service(config, ui_components).augment_raw_dataset()
normalize_augmented_data = lambda config, split='train', ui_components=None: create_augmentation_service(config, ui_components).normalize_augmented_dataset(split)
run_full_pipeline = lambda config, split='train', ui_components=None: create_augmentation_service(config, ui_components).run_full_augmentation_pipeline(split)
cleanup_augmented_files = lambda config, include_prep=True, ui_components=None: create_augmentation_service(config, ui_components).cleanup_augmented_data(include_prep)
get_dataset_status = lambda config, ui_components=None: create_augmentation_service(config, ui_components).get_augmentation_status()