"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Updated core engine menggunakan SRP utils modules dengan one-liner style
"""

import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from smartcash.dataset.augmentor.utils.config_extractor import create_split_aware_context
from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
from smartcash.dataset.augmentor.utils.file_operations import smart_find_images_split_aware
from smartcash.dataset.augmentor.utils.batch_processor import process_batch_split_aware
from smartcash.dataset.augmentor.utils.cleanup_operations import cleanup_split_aware
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker

from smartcash.dataset.augmentor.strategies.balancer import ClassBalancingStrategy
from smartcash.dataset.augmentor.strategies.selector import FileSelectionStrategy
from smartcash.dataset.augmentor.strategies.priority import PriorityCalculator
from smartcash.dataset.augmentor.core.pipeline import PipelineFactory

class AugmentationEngine:
    """Updated core engine menggunakan SRP utils modules"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        # Create context menggunakan SRP module
        self.context = create_split_aware_context(config, communicator)
        self.config, self.progress, self.paths = self.context['config'], self.context['progress'], self.context['paths']
        self.comm = self.context.get('comm')
        
        # Initialize strategies
        self.balancer = ClassBalancingStrategy(config)
        self.selector = FileSelectionStrategy(config)
        self.priority_calc = PriorityCalculator(config)
        self.pipeline_factory = PipelineFactory(config)
        
        self.target_split = config.get('target_split', 'train')
    
    def run_augmentation_pipeline(self, target_split: str = None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute pipeline menggunakan SRP modules"""
        start_time = time.time()
        actual_target_split = target_split or self.target_split
        
        self.comm and self.comm.start_operation("Augmentation Pipeline", 100)
        
        try:
            # Phase 1: Dataset Detection (0-15%) - menggunakan SRP detector
            self.progress.progress("overall", 5, 100, f"Detecting {actual_target_split} structure")
            dataset_info = detect_split_structure(self.paths['raw_dir'])
            if dataset_info['status'] == 'error':
                return self._error_result(f"Detection failed: {dataset_info['message']}")
            
            # Phase 2: File Analysis (15-30%) - menggunakan SRP file operations
            self.progress.progress("overall", 20, 100, f"Analyzing {actual_target_split} files")
            analysis_result = self._analyze_split_files(actual_target_split)
            if not analysis_result['success']:
                return self._error_result(f"Analysis failed: {analysis_result['message']}")
            
            # Phase 3: Strategy Planning (30-40%) - menggunakan strategies
            self.progress.progress("overall", 35, 100, "Planning augmentation strategy")
            strategy_result = self._plan_augmentation_strategy(analysis_result['metadata'])
            
            # Phase 4: Augmentation Execution (40-90%) - menggunakan batch processor
            self.progress.progress("overall", 45, 100, f"Executing augmentation for {len(strategy_result['selected_files'])} files")
            execution_result = self._execute_split_augmentation(strategy_result, actual_target_split, progress_callback)
            
            # Phase 5: Results Summary (90-100%)
            self.progress.progress("overall", 95, 100, "Finalizing results")
            result = self._create_success_result(execution_result, time.time() - start_time, actual_target_split)
            
            self.comm and self.comm.complete_operation("Augmentation Pipeline", 
                f"Pipeline completed: {result['total_generated']} files generated")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.comm and self.comm.error_operation("Augmentation Pipeline", error_msg)
            return self._error_result(error_msg)
    
    def _analyze_split_files(self, target_split: str) -> Dict[str, Any]:
        """Analyze files menggunakan SRP file operations"""
        try:
            # Get source files menggunakan smart finder
            source_files = smart_find_images_split_aware(self.paths['raw_dir'], target_split)
            if not source_files:
                return {'success': False, 'message': f'No source files found for split {target_split}'}
            
            # Extract metadata using batch processing
            metadata_extractor = lambda file_path: self._extract_file_metadata(file_path, target_split)
            metadata_results = process_batch_split_aware(source_files, metadata_extractor, 
                                                       progress_tracker=self.progress, 
                                                       operation_name="file analysis",
                                                       split_context=target_split)
            
            # Process metadata results
            files_metadata = {r['file_path']: r['metadata'] for r in metadata_results if r.get('status') == 'success'}
            class_distribution = self._aggregate_class_distribution(files_metadata)
            
            return {
                'success': True, 'total_files': len(source_files), 'valid_files': len(files_metadata),
                'metadata': files_metadata, 'class_distribution': class_distribution, 'target_split': target_split
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Analysis error: {str(e)}'}
    
    def _plan_augmentation_strategy(self, files_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Plan strategy menggunakan balancer, priority, dan selector"""
        # Calculate class needs using balancer
        class_distribution = self._aggregate_class_distribution(files_metadata)
        target_count = self.config.get('target_count', 1000)
        class_needs = self.balancer.calculate_balancing_needs(class_distribution, target_count)
        
        # Calculate file priorities
        file_priorities = self.priority_calc.calculate_augmentation_priority(files_metadata, class_needs, class_distribution)
        
        # Select files using selector
        selected_files = self.selector.select_prioritized_files(class_needs, files_metadata)
        
        return {
            'class_needs': class_needs, 'file_priorities': file_priorities, 'selected_files': selected_files,
            'strategy_quality': self._assess_strategy_quality(class_needs, selected_files, files_metadata)
        }
    
    def _execute_split_augmentation(self, strategy_result: Dict[str, Any], target_split: str, 
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute augmentation menggunakan batch processor"""
        selected_files = strategy_result['selected_files']
        if not selected_files:
            return {'status': 'error', 'message': 'No files selected for augmentation'}
        
        # Setup output directories menggunakan SRP path operations
        from smartcash.dataset.augmentor.utils.path_operations import ensure_split_dirs
        ensure_split_dirs(self.paths['aug_dir'], target_split)
        
        # Create pipeline
        aug_types = self.config.get('types', ['combined'])
        pipeline = self.pipeline_factory.create_pipeline(aug_types[0] if aug_types else 'combined', 
                                                        self.config.get('intensity', 0.7))
        
        # Process files using batch processor
        augmentation_processor = lambda file_path: self._augment_single_file(file_path, pipeline, target_split)
        aug_results = process_batch_split_aware(selected_files, augmentation_processor,
                                              progress_tracker=self.progress,
                                              operation_name="augmentation", 
                                              split_context=target_split)
        
        return {'aug_results': aug_results, 'pipeline_type': aug_types[0] if aug_types else 'combined'}
    
    def cleanup_augmented_data(self, target_split: str = None, include_preprocessed: bool = True) -> Dict[str, Any]:
        """Cleanup menggunakan SRP cleanup operations"""
        actual_target_split = target_split or self.target_split
        return cleanup_split_aware(self.paths['aug_dir'], 
                                 self.paths['prep_dir'] if include_preprocessed else None, 
                                 actual_target_split, self.progress)
    
    def _extract_file_metadata(self, file_path: str, target_split: str) -> Dict[str, Any]:
        """Extract metadata dari single file"""
        try:
            from smartcash.dataset.augmentor.utils.bbox_operations import load_yolo_labels
            from pathlib import Path
            
            # Load labels
            label_path = str(Path(file_path).parent.parent / 'labels' / f"{Path(file_path).stem}.txt")
            bboxes, class_labels = load_yolo_labels(label_path)
            
            # Calculate metadata
            class_counts = defaultdict(int)
            for cls in class_labels:
                class_counts[str(cls)] += 1
            
            primary_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else '0'
            
            return {
                'status': 'success', 'file_path': file_path,
                'metadata': {
                    'classes': set(map(str, class_labels)), 'class_counts': dict(class_counts),
                    'total_instances': len(bboxes), 'num_classes': len(set(class_labels)),
                    'primary_class': primary_class, 'has_labels': len(bboxes) > 0
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'file_path': file_path, 'error': str(e)}
    
    def _augment_single_file(self, file_path: str, pipeline, target_split: str) -> Dict[str, Any]:
        """Augment single file dengan pipeline"""
        try:
            import cv2
            from smartcash.dataset.augmentor.utils.bbox_operations import load_yolo_labels, save_validated_labels
            from smartcash.dataset.augmentor.utils.path_operations import get_stem
            from pathlib import Path
            
            # Load image dan labels
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Cannot load image'}
            
            label_path = str(Path(file_path).parent.parent / 'labels' / f"{Path(file_path).stem}.txt")
            bboxes, class_labels = load_yolo_labels(label_path)
            
            # Generate variants
            num_variations = self.config.get('num_variations', 2)
            generated_count = 0
            
            for i in range(num_variations):
                if self._generate_variant(image, bboxes, class_labels, pipeline, file_path, i + 1, target_split):
                    generated_count += 1
            
            return {'status': 'success', 'file': file_path, 'generated': generated_count}
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _generate_variant(self, image, bboxes: List, class_labels: List, pipeline, 
                         source_file: str, variant_num: int, target_split: str) -> bool:
        """Generate single variant"""
        try:
            import cv2
            from pathlib import Path
            from smartcash.dataset.augmentor.utils.bbox_operations import save_validated_labels
            
            # Apply augmentation
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image, aug_bboxes, aug_class_labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
            
            # Generate output paths
            stem = Path(source_file).stem
            aug_filename = f"aug_{stem}_{variant_num:03d}.jpg"
            
            aug_img_path = Path(self.paths['aug_dir']) / target_split / 'images' / aug_filename
            aug_label_path = Path(self.paths['aug_dir']) / target_split / 'labels' / f"{Path(aug_filename).stem}.txt"
            
            # Save image
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            img_saved = cv2.imwrite(str(aug_img_path), aug_image, save_params)
            
            # Save labels
            label_saved = save_validated_labels(aug_bboxes, aug_class_labels, str(aug_label_path)) if aug_bboxes else True
            
            return img_saved and label_saved
            
        except Exception:
            return False
    
    def _aggregate_class_distribution(self, files_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate class distribution dari metadata"""
        class_distribution = defaultdict(int)
        for metadata in files_metadata.values():
            for cls, count in metadata.get('class_counts', {}).items():
                class_distribution[cls] += count
        return dict(class_distribution)
    
    def _assess_strategy_quality(self, class_needs: Dict[str, int], selected_files: List[str], 
                               files_metadata: Dict[str, Dict[str, Any]]) -> str:
        """Assess strategy quality"""
        total_needed = sum(class_needs.values())
        classes_with_needs = len([n for n in class_needs.values() if n > 0])
        
        if not selected_files or total_needed == 0:
            return 'poor'
        elif len(selected_files) >= total_needed * 0.8 and classes_with_needs >= 5:
            return 'excellent'
        elif len(selected_files) >= total_needed * 0.5:
            return 'good'
        else:
            return 'moderate'
    
    def _create_success_result(self, execution_result: Dict[str, Any], processing_time: float, target_split: str) -> Dict[str, Any]:
        """Create success result"""
        aug_results = execution_result.get('aug_results', [])
        successful = [r for r in aug_results if r.get('status') == 'success']
        total_generated = sum(r.get('generated', 0) for r in successful)
        
        return {
            'status': 'success', 'total_generated': total_generated, 'processed_files': len(aug_results),
            'success_rate': len(successful) / len(aug_results) * 100 if aug_results else 0,
            'processing_time': processing_time, 'target_split': target_split,
            'pipeline_type': execution_result.get('pipeline_type', 'combined'),
            'augmentation_speed': len(aug_results) / processing_time if processing_time > 0 else 0
        }
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """Create error result"""
        return {'status': 'error', 'message': message, 'total_generated': 0}

# One-liner utilities menggunakan SRP modules
create_augmentation_engine = lambda config, communicator=None: AugmentationEngine(config, communicator)
run_augmentation_pipeline = lambda config, communicator=None, target_split='train': create_augmentation_engine(config, communicator).run_augmentation_pipeline(target_split)
cleanup_augmented_data = lambda config, target_split='train', include_preprocessed=True: create_augmentation_engine(config).cleanup_augmented_data(target_split, include_preprocessed)