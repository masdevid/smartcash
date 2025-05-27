"""
File: smartcash/dataset/preprocessor/core/preprocessing_coordinator.py
Deskripsi: Koordinator preprocessing dengan callback yang diperbaiki dan fixed parameter conflicts
"""

import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.processors.split_processor import SplitProcessor
from smartcash.dataset.utils.move_utils import calculate_total_images

class SplitCoordinator:
    """Koordinator untuk pemrosesan multi-split dengan fixed callback handling"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        self.active_processors: Dict[str, SplitProcessor] = {}
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback"""
        self._progress_callback = callback
    
    def resolve_target_splits(self, split_request: str) -> List[str]:
        """Resolve target splits dengan val->valid mapping"""
        split_mapping = {'val': 'valid', 'validation': 'valid'}
        normalized = split_mapping.get(split_request.lower(), split_request.lower())
        
        if normalized == 'all':
            return ['train', 'valid', 'test']
        elif normalized in ['train', 'valid', 'test']:
            return [normalized]
        else:
            self.logger.warning(f"⚠️ Unknown split: {split_request}, using 'all'")
            return ['train', 'valid', 'test']
    
    def coordinate_parallel_splits(self, target_splits: List[str], processing_config: Dict[str, Any], 
                                 force_reprocess: bool = False) -> Dict[str, Any]:
        """Koordinasi parallel splits dengan fixed progress tracking"""
        start_time = time.time()
        image_counts = calculate_total_images(target_splits, self.config)
        total_images = sum(image_counts.values())
        
        self._notify_progress(40, f"🎯 Starting {len(target_splits)} splits processing", total_files=total_images)
        
        split_results = {}
        max_workers = min(3, len(target_splits))
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SplitCoord") as executor:
                future_to_split = {}
                
                for i, split_name in enumerate(target_splits):
                    processor = SplitProcessor(self.config, self.logger)
                    
                    # Fixed callback - avoid parameter conflicts
                    def create_split_callback(split_idx, total_splits):
                        def split_callback(progress, message):
                            mapped_progress = 40 + (split_idx / total_splits * 40) + (progress / 100 * 40 / total_splits)
                            self._notify_progress(int(mapped_progress), f"Split {split_name}: {message}")
                        return split_callback
                    
                    processor.register_progress_callback(create_split_callback(i, len(target_splits)))
                    self.active_processors[split_name] = processor
                    
                    future = executor.submit(
                        processor.process_split_dataset,
                        split_name, processing_config, force_reprocess
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
                        
                        progress = 40 + (completed_count / len(target_splits)) * 40
                        processed = result.get('processed', 0)
                        self._notify_progress(
                            int(progress), 
                            f"✅ Split {split_name}: {processed} gambar diproses"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"❌ Error processing {split_name}: {str(e)}")
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
        
        success = len(successful_splits) > 0
        message = f'{len(successful_splits)} dari {len(target_splits)} splits berhasil diproses'
        
        if success:
            self.logger.success(f"✅ Koordinasi selesai: {message}")
        else:
            self.logger.error(f"❌ Koordinasi gagal: {message}")
        
        return {
            'success': success,
            'message': message,
            'split_results': split_results,
            'successful_splits': successful_splits,
            'coordination_time': coordination_time,
            'total_target_images': total_images
        }
    
    def get_coordination_summary(self) -> Dict[str, Any]:
        """Get coordination status summary"""
        return {
            'active_processors': len(self.active_processors),
            'processor_status': {
                name: processor.get_processor_status() 
                for name, processor in self.active_processors.items()
            },
            'ready_for_coordination': True
        }
    
    def cleanup_coordination_state(self) -> None:
        """Cleanup coordination state"""
        for processor in self.active_processors.values():
            processor.cleanup_processor_state()
        self.active_processors.clear()
    
    def _notify_progress(self, progress: int, message: str, **kwargs):
        """Internal progress notification dengan clean parameters"""
        if self._progress_callback:
            try:
                self._progress_callback(progress=progress, message=message, **kwargs)
            except Exception as e:
                self.logger.debug(f"🔧 Coordination progress error: {str(e)}")