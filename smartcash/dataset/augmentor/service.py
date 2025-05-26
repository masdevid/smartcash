"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Main orchestrator service untuk augmentasi dengan flow yang benar - Raw → Augmented → Preprocessed
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

class AugmentationService:
    """Main orchestrator service untuk augmentasi dengan simplified dan focused flow"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        """
        Initialize augmentation service dengan UI communication bridge.
        
        Args:
            config: Dictionary konfigurasi aplikasi
            ui_components: Dictionary komponen UI untuk communication
        """
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
        Augmentasi dataset dari raw data dengan progress tracking.
        Flow: /data → /data/augmented
        
        Args:
            progress_callback: Callback untuk progress updates
            
        Returns:
            Dictionary hasil augmentasi
        """
        operation_name = "Augmentasi Dataset"
        self.comm.start_operation(operation_name)
        
        try:
            # Validasi raw directory
            raw_dir = self.aug_config.raw_dir
            if not os.path.exists(raw_dir):
                error_msg = f"Raw directory tidak ditemukan: {raw_dir}"
                self.comm.error_operation(operation_name, error_msg)
                return self._create_error_result(error_msg)
            
            self.logger.info(f"🚀 Memulai augmentasi raw dataset: {raw_dir}")
            
            # Setup progress callback bridge
            if progress_callback:
                self._setup_progress_bridge(progress_callback)
            
            # Execute augmentation dengan engine
            aug_result = self.engine.process_raw_data(raw_dir, self.aug_config.aug_dir)
            
            if aug_result['status'] == 'success':
                success_msg = f"Augmentasi berhasil: {aug_result['total_generated']} gambar dihasilkan"
                self.comm.complete_operation(operation_name, success_msg)
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
        Normalisasi dataset augmented ke preprocessed.
        Flow: /data/augmented → /data/preprocessed/{split}
        
        Args:
            target_split: Target split untuk normalisasi (train/valid/test)
            progress_callback: Callback untuk progress updates
            
        Returns:
            Dictionary hasil normalisasi
        """
        operation_name = f"Normalisasi ke Split {target_split}"
        self.comm.start_operation(operation_name)
        
        try:
            # Validasi augmented directory
            aug_dir = self.aug_config.aug_dir
            if not os.path.exists(aug_dir):
                error_msg = f"Augmented directory tidak ditemukan: {aug_dir}"
                self.comm.error_operation(operation_name, error_msg)
                return self._create_error_result(error_msg)
            
            self.logger.info(f"🔄 Memulai normalisasi: {aug_dir} → preprocessed/{target_split}")
            
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
        Jalankan full pipeline augmentasi: Raw → Augmented → Preprocessed.
        
        Args:
            target_split: Target split untuk hasil akhir
            progress_callback: Callback untuk progress updates
            
        Returns:
            Dictionary hasil pipeline lengkap
        """
        pipeline_name = "Full Augmentation Pipeline"
        self.comm.start_operation(pipeline_name)
        
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: Augmentasi raw dataset
            self.logger.info("📊 Step 1/2: Augmentasi raw dataset")
            if self.comm: self.comm.progress("overall", 10, 100, "Step 1: Augmentasi raw dataset")
            
            aug_result = self.augment_raw_dataset(progress_callback)
            results['augmentation'] = aug_result
            
            if aug_result['status'] != 'success':
                error_msg = f"Pipeline gagal pada step augmentasi: {aug_result.get('message')}"
                self.comm.error_operation(pipeline_name, error_msg)
                return self._create_pipeline_error_result(error_msg, results)
            
            # Step 2: Normalisasi ke preprocessed
            self.logger.info("📊 Step 2/2: Normalisasi ke preprocessed")
            if self.comm: self.comm.progress("overall", 60, 100, "Step 2: Normalisasi ke preprocessed")
            
            norm_result = self.normalize_augmented_dataset(target_split, progress_callback)
            results['normalization'] = norm_result
            
            if norm_result['status'] != 'success':
                error_msg = f"Pipeline gagal pada step normalisasi: {norm_result.get('message')}"
                self.comm.error_operation(pipeline_name, error_msg)
                return self._create_pipeline_error_result(error_msg, results)
            
            # Pipeline success summary
            total_time = time.time() - start_time
            pipeline_summary = self._create_pipeline_success_result(results, total_time, target_split)
            
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
        """
        Cleanup data augmentasi dengan prefix aug_*.
        
        Args:
            include_preprocessed: Apakah cleanup juga preprocessed files
            progress_callback: Callback untuk progress updates
            
        Returns:
            Dictionary hasil cleanup
        """
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
        """
        Dapatkan status augmentasi saat ini.
        
        Returns:
            Dictionary status augmentasi
        """
        aug_dir = self.aug_config.aug_dir
        prep_dir = self.aug_config.prep_dir
        
        status = {
            'raw_exists': os.path.exists(self.aug_config.raw_dir),
            'augmented_exists': os.path.exists(aug_dir),
            'preprocessed_exists': os.path.exists(prep_dir),
            'augmented_files': 0,
            'preprocessed_files': 0
        }
        
        # Count augmented files
        if status['augmented_exists']:
            aug_stats = self.normalizer.get_augmented_stats(aug_dir)
            status['augmented_files'] = aug_stats.get('aug_images', 0)
        
        # Count preprocessed files (train split)
        train_dir = os.path.join(prep_dir, 'train', 'images')
        if os.path.exists(train_dir):
            status['preprocessed_files'] = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))])
        
        return status
    
    def _setup_progress_bridge(self, progress_callback: callable) -> None:
        """Setup bridge untuk external progress callback."""
        if hasattr(self.comm, 'report_progress_with_callback'):
            self.comm.report_progress_with_callback = progress_callback
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'status': 'error',
            'message': error_message,
            'timestamp': time.time()
        }
    
    def _create_pipeline_success_result(self, results: Dict, total_time: float, target_split: str) -> Dict[str, Any]:
        """Create pipeline success result summary."""
        aug_result = results.get('augmentation', {})
        norm_result = results.get('normalization', {})
        
        return {
            'status': 'success',
            'pipeline': 'full_augmentation',
            'total_time': total_time,
            'target_split': target_split,
            'total_files': aug_result.get('total_generated', 0),
            'final_output': f"{norm_result.get('target_dir', 'preprocessed')}/{target_split}",
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

# One-liner service operations
augment_raw_data = lambda config, ui_components=None: create_augmentation_service(config, ui_components).augment_raw_dataset()
normalize_augmented_data = lambda config, split='train', ui_components=None: create_augmentation_service(config, ui_components).normalize_augmented_dataset(split)
run_full_pipeline = lambda config, split='train', ui_components=None: create_augmentation_service(config, ui_components).run_full_augmentation_pipeline(split)
cleanup_augmented_files = lambda config, include_prep=True, ui_components=None: create_augmentation_service(config, ui_components).cleanup_augmented_data(include_prep)