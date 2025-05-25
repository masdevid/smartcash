"""
File: smartcash/dataset/preprocessor/core/preprocessing_coordinator.py
Deskripsi: Fixed koordinator untuk pemrosesan multi-split - resolved callback parameter conflicts
"""

import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.processors.split_processor import SplitProcessor
from smartcash.dataset.utils.move_utils import calculate_total_images


class SplitCoordinator:
    """Fixed koordinator untuk pemrosesan multi-split - no parameter conflicts in callbacks."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize coordinator dengan configuration dan dependencies."""
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        self.active_processors: Dict[str, SplitProcessor] = {}
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk coordination updates."""
        self._progress_callback = callback
    
    def resolve_target_splits(self, split_request: str) -> List[str]:
        """Resolve target splits dari request dengan intelligent mapping."""
        split_mapping = {'val': 'valid', 'validation': 'valid'}
        normalized_request = split_mapping.get(split_request.lower(), split_request.lower())
        
        if normalized_request == 'all':
            return ['train', 'valid', 'test']
        elif normalized_request in ['train', 'valid', 'test']:
            return [normalized_request]
        else:
            self.logger.warning(f"⚠️ Unknown split request: {split_request}, using 'all'")
            return ['train', 'valid', 'test']
    
    def calculate_processing_order(self, target_splits: List[str]) -> List[Dict[str, Any]]:
        """Hitung optimal processing order berdasarkan ukuran dataset dan dependencies."""
        image_counts = calculate_total_images(target_splits, self.config)
        
        processing_order = []
        for split in target_splits:
            split_count = image_counts.get(split, 0)
            processing_order.append({
                'split': split,
                'image_count': split_count,
                'priority': self._calculate_split_priority(split, split_count),
                'estimated_time': split_count * 0.1
            })
        
        processing_order.sort(key=lambda x: (x['priority'], -x['image_count']))
        self.logger.info(f"📊 Processing order: {[item['split'] for item in processing_order]}")
        return processing_order
    
    def coordinate_parallel_splits(self, target_splits: List[str], processing_config: Dict[str, Any], 
                                 force_reprocess: bool = False) -> Dict[str, Any]:
        """Fixed koordinasi pemrosesan parallel - resolved callback parameter conflicts."""
        start_time = time.time()
        processing_order = self.calculate_processing_order(target_splits)
        total_images = sum(item['image_count'] for item in processing_order)
        
        self._notify_coordination_progress(15, f"Memulai koordinasi {len(target_splits)} splits", 
                                         total_files=total_images)
        
        split_results = {}
        max_workers = min(3, len(processing_order))
        
        self.logger.info(f"🔧 Processing config keys: {list(processing_config.keys())}")
        self.logger.info(f"📂 Target splits: {target_splits}")
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SplitCoord") as executor:
                future_to_split = {}
                
                for i, split_info in enumerate(processing_order):
                    split_name = split_info['split']
                    
                    processor = SplitProcessor(self.config, self.logger)
                    # FIXED: Create callback dengan resolved parameters
                    processor.register_progress_callback(
                        self._create_split_progress_callback_fixed(split_name, i, len(processing_order), total_images)
                    )
                    
                    self.active_processors[split_name] = processor
                    
                    future = executor.submit(
                        processor.process_split_dataset,
                        split_name,
                        processing_config,
                        force_reprocess
                    )
                    future_to_split[future] = split_name
                
                # Collect results
                completed_count = 0
                for future in as_completed(future_to_split):
                    split_name = future_to_split[future]
                    completed_count += 1
                    
                    try:
                        result = future.result()
                        split_results[split_name] = result
                        
                        coord_progress = 15 + (completed_count / len(processing_order)) * 75
                        self._notify_coordination_progress(
                            int(coord_progress), 
                            f"Split {split_name} selesai: {result.get('processed', 0)} gambar",
                            split_progress={split_name: result}
                        )
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error processing split {split_name}: {str(e)}")
                        split_results[split_name] = {
                            'success': False, 'error': str(e), 'processed': 0, 'failed': 1
                        }
        
        except Exception as e:
            self.logger.error(f"❌ Coordination error: {str(e)}")
            return {'success': False, 'message': f"Coordination failed: {str(e)}"}
        
        finally:
            self.active_processors.clear()
        
        # Validate results
        successful_splits = [s for s, r in split_results.items() if r.get('success', False)]
        coordination_time = time.time() - start_time
        
        self.logger.success(f"✅ Koordinasi selesai: {len(successful_splits)}/{len(target_splits)} splits berhasil")
        
        return {
            'success': len(successful_splits) > 0,
            'message': f'{len(successful_splits)} dari {len(target_splits)} splits berhasil diproses',
            'split_results': split_results,
            'successful_splits': successful_splits,
            'coordination_time': coordination_time,
            'total_target_images': total_images
        }
    
    def _create_split_progress_callback_fixed(self, split_name: str, split_index: int, 
                                            total_splits: int, total_images: int) -> Callable:
        """FIXED: Create progress callback dengan no parameter conflicts."""
        def split_progress_callback_fixed(progress=0, message="", **other_kwargs):
            """FIXED: Explicit parameter names untuk avoid conflicts."""
            if self._progress_callback:
                try:
                    # Map split progress ke coordination progress
                    base_progress = 15 + (split_index / total_splits) * 75
                    mapped_progress = base_progress + (progress / 100) * (75 / total_splits)
                    
                    # Build message dengan fallback
                    display_message = message or f'Processing {split_name}'
                    
                    # FIXED: Call dengan explicit keyword mapping
                    self._progress_callback(
                        progress=int(mapped_progress),
                        message=display_message,
                        split=split_name,
                        step=2,
                        split_step=f'{split_name} ({split_index+1}/{total_splits})',
                        total_files_all=total_images
                    )
                except Exception as e:
                    self.logger.debug(f"🔧 Split callback error: {str(e)}")
        
        return split_progress_callback_fixed
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Dapatkan summary status coordination."""
        return {
            'active_processors': len(self.active_processors),
            'processor_status': {
                name: processor.get_processor_status() 
                for name, processor in self.active_processors.items()
            },
            'ready_for_coordination': True
        }
    
    def cleanup_coordination_state(self) -> None:
        """Cleanup coordination state dan active processors."""
        for processor in self.active_processors.values():
            processor.cleanup_processor_state()
        self.active_processors.clear()
        self.logger.debug("🧹 Coordination state cleaned up")
    
    def _calculate_split_priority(self, split: str, image_count: int) -> int:
        """Calculate processing priority untuk split (lower = higher priority)."""
        priority_map = {'train': 0, 'valid': 1, 'test': 2}
        base_priority = priority_map.get(split, 3)
        size_factor = max(0, 1000 - image_count) // 100
        return base_priority + size_factor
    
    def _notify_coordination_progress(self, progress: int, message_text: str, **kwargs):
        """FIXED: Internal coordination progress notification dengan clean parameters."""
        if self._progress_callback:
            try:
                # FIXED: Create clean parameter set untuk avoid conflicts
                clean_progress_params = {
                    'progress': progress, 
                    'message': message_text, 
                    'step': 2,
                    'split_step': "Koordinasi"
                }
                # Add additional kwargs selectively
                for key, value in kwargs.items():
                    if key not in clean_progress_params:
                        clean_progress_params[key] = value
                
                self._progress_callback(**clean_progress_params)
            except Exception as e:
                self.logger.debug(f"🔧 Coordination progress error: {str(e)}")