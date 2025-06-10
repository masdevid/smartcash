"""
File: smartcash/ui/dataset/preprocessing/utils/progress_utils.py
Deskripsi: Enhanced progress utilities dengan triple tracker kompatibel dan multi-split support
"""

from typing import Dict, Any, Callable, List
from .ui_utils import is_milestone_step

def create_enhanced_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """Create enhanced progress callback dengan triple tracker support"""
    def enhanced_progress_callback(step: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Enhanced overall progress calculation
                overall_progress = min(100, (current / max(total, 1)) * 100)
                progress_tracker.update_overall(int(overall_progress), message)
                
                # Enhanced step progress mapping
                step_progress = map_enhanced_step_to_progress(step, current, total)
                step_name = _get_enhanced_step_display_name(step)
                progress_tracker.update_step(step_progress, f"Step: {step_name}")
                
                # Enhanced current operation progress
                current_progress = min(100, (current / max(total, 1)) * 100)
                progress_tracker.update_current(int(current_progress), f"Processing: {current}/{total}")
            
            # Only log enhanced milestones untuk prevent flooding
            if is_enhanced_milestone_step(step, current):
                logger = ui_components.get('logger')
                if logger:
                    enhanced_message = f"🔄 {message} ({current}/{total})"
                    logger.info(enhanced_message)
                    
        except Exception:
            pass  # Silent fail untuk prevent blocking
    
    return enhanced_progress_callback

def map_enhanced_step_to_progress(step: str, current: int, total: int) -> int:
    """Enhanced step mapping dengan multi-split awareness"""
    enhanced_step_ranges = {
        # Validation phase
        'validate': (0, 15),
        'check_splits': (15, 20),
        'validate_images': (20, 25),
        
        # Analysis phase  
        'analyze': (25, 30),
        'split_analysis': (30, 35),
        
        # Processing phase
        'normalize': (35, 65),
        'resize': (65, 80),
        'aspect_ratio': (80, 85),
        
        # Saving phase
        'save': (85, 95),
        'multi_split_save': (95, 98),
        
        # Finalization
        'finalize': (98, 100),
        'validation_final': (98, 100)
    }
    
    # Get step base name (remove suffixes)
    normalized_step = step.lower().split('_')[0]
    
    if normalized_step in enhanced_step_ranges:
        start, end = enhanced_step_ranges[normalized_step]
        if total > 0:
            step_progress = start + ((current / total) * (end - start))
            return min(100, max(0, int(step_progress)))
    
    # Fallback calculation
    return min(100, (current / max(total, 1)) * 100)

def setup_enhanced_progress_tracker(ui_components: Dict[str, Any], operation_name: str = "Enhanced Dataset Preprocessing"):
    """Setup enhanced progress tracker dengan triple level support"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Enhanced steps untuk multi-split preprocessing
        enhanced_steps = [
            "validate", "analyze", "normalize", "resize", 
            "aspect_ratio", "save", "finalize"
        ]
        
        # Enhanced step weights untuk realistic progress
        enhanced_step_weights = {
            "validate": 20,
            "analyze": 10, 
            "normalize": 30,
            "resize": 20,
            "aspect_ratio": 5,
            "save": 10,
            "finalize": 5
        }
        
        # Show dengan enhanced steps
        progress_tracker.show(operation_name, enhanced_steps, enhanced_step_weights)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    logger = ui_components.get('logger')
    if logger:
        logger.info(f"🚀 Memulai {operation_name.lower()} dengan enhanced features")

def complete_progress_tracker(ui_components: Dict[str, Any], message: str):
    """Complete enhanced progress tracker dengan success state"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'complete'):
        progress_tracker.complete(message)
    
    # Log completion dengan enhanced message
    logger = ui_components.get('logger')
    if logger:
        logger.success(f"✅ {message}")

def error_progress_tracker(ui_components: Dict[str, Any], error_msg: str):
    """Set enhanced error state pada progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'error'):
        progress_tracker.error(error_msg)
    
    # Log error dengan enhanced formatting
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"❌ {error_msg}")

def reset_progress_tracker(ui_components: Dict[str, Any]):
    """Reset enhanced progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'reset'):
        progress_tracker.reset()
    
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔄 Progress tracker direset")

def update_multi_split_progress(ui_components: Dict[str, Any], current_split: str, 
                               split_index: int, total_splits: int, 
                               split_progress: int, overall_message: str = None):
    """Update progress untuk multi-split processing"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Calculate overall progress berdasarkan split completion
        overall_progress = ((split_index / total_splits) * 100) + ((split_progress / total_splits))
        overall_progress = min(100, max(0, int(overall_progress)))
        
        # Update dengan split-aware messages
        split_message = overall_message or f"Processing {current_split} split"
        progress_tracker.update_overall(overall_progress, split_message)
        
        # Update step progress untuk current split
        progress_tracker.update_step(split_progress, f"Split: {current_split}")
        
        # Update current operation
        progress_tracker.update_current(split_progress, f"{current_split}: {split_progress}%")

def _get_enhanced_step_display_name(step: str) -> str:
    """Get enhanced display name untuk steps"""
    step_display_names = {
        'validate': 'Validasi Dataset',
        'check_splits': 'Check Multi-Split',
        'validate_images': 'Validasi Gambar',
        'analyze': 'Analisis Data',
        'split_analysis': 'Analisis Split',
        'normalize': 'Normalisasi Pixel',
        'resize': 'Resize Gambar',
        'aspect_ratio': 'Aspect Ratio',
        'save': 'Menyimpan Hasil',
        'multi_split_save': 'Save Multi-Split',
        'finalize': 'Finalisasi',
        'validation_final': 'Validasi Akhir'
    }
    
    return step_display_names.get(step.lower(), step.replace('_', ' ').title())

def is_enhanced_milestone_step(step: str, progress: int) -> bool:
    """Enhanced milestone detection untuk prevent log flooding"""
    # Major milestones untuk enhanced preprocessing
    enhanced_milestone_steps = [
        'validate', 'analyze', 'normalize', 'resize', 
        'aspect_ratio', 'save', 'finalize', 'complete'
    ]
    
    # Milestone progress points
    milestone_progress_points = [0, 10, 25, 50, 75, 90, 100]
    
    return (
        step.lower() in enhanced_milestone_steps or 
        progress in milestone_progress_points or 
        progress % 20 == 0  # Every 20% for enhanced tracking
    )

def track_batch_progress(ui_components: Dict[str, Any], batch_index: int, 
                        total_batches: int, batch_size: int, 
                        current_step: str = "processing"):
    """Track progress untuk batch processing"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Calculate batch progress
        batch_progress = min(100, (batch_index / max(total_batches, 1)) * 100)
        
        # Enhanced batch message
        processed_items = min(batch_index * batch_size, total_batches * batch_size)
        total_items = total_batches * batch_size
        
        batch_message = f"Batch {batch_index}/{total_batches} ({processed_items}/{total_items} items)"
        
        # Update current operation dengan batch info
        progress_tracker.update_current(int(batch_progress), batch_message)
        
        # Log milestone batches
        if batch_index % max(1, total_batches // 10) == 0:  # Every 10% of batches
            logger = ui_components.get('logger')
            if logger:
                logger.info(f"📦 {batch_message} - {current_step}")

def create_split_aware_callback(ui_components: Dict[str, Any], current_split: str, 
                               split_index: int, total_splits: int) -> Callable:
    """Create callback yang aware terhadap multi-split processing"""
    def split_aware_callback(step: str, current: int, total: int, message: str):
        try:
            # Calculate split-relative progress
            split_progress = min(100, (current / max(total, 1)) * 100)
            
            # Update dengan split context
            update_multi_split_progress(
                ui_components, current_split, split_index, 
                total_splits, int(split_progress), message
            )
            
            # Enhanced logging untuk split milestones
            if is_enhanced_milestone_step(step, split_progress):
                logger = ui_components.get('logger')
                if logger:
                    split_emoji = {'train': '🏋️', 'valid': '✅', 'test': '🧪'}.get(current_split, '📁')
                    logger.info(f"{split_emoji} {current_split}: {message} ({current}/{total})")
                    
        except Exception:
            pass  # Silent fail untuk prevent blocking
    
    return split_aware_callback

# Enhanced utility functions
def get_progress_tracker_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get current status dari enhanced progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if not progress_tracker:
        return {'available': False}
    
    return {
        'available': True,
        'has_show_method': hasattr(progress_tracker, 'show'),
        'has_update_methods': all(
            hasattr(progress_tracker, method) for method in 
            ['update_overall', 'update_step', 'update_current']
        ),
        'has_state_methods': all(
            hasattr(progress_tracker, method) for method in 
            ['complete', 'error', 'reset']
        )
    }

def validate_progress_tracker_compatibility(ui_components: Dict[str, Any]) -> bool:
    """Validate bahwa progress tracker kompatibel dengan enhanced features"""
    status = get_progress_tracker_status(ui_components)
    
    required_features = [
        'available', 'has_show_method', 
        'has_update_methods', 'has_state_methods'
    ]
    
    is_compatible = all(status.get(feature, False) for feature in required_features)
    
    if not is_compatible:
        logger = ui_components.get('logger')
        if logger:
            missing_features = [f for f in required_features if not status.get(f, False)]
            logger.warning(f"⚠️ Progress tracker tidak fully compatible: missing {missing_features}")
    
    return is_compatible