"""
File: smartcash/ui/dataset/augmentation/utils/progress_utils.py
Deskripsi: Progress tracking utilities untuk augmentation operations
"""

from typing import Dict, Any, Callable
import time

class AugmentationProgressManager:
    """Progress manager untuk augmentation operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.last_update = {'time': 0, 'percentage': -1, 'step': ''}
    
    def create_progress_callback(self) -> Callable:
        """Create progress callback untuk service integration"""
        def callback(step: str, current: int, total: int, message: str):
            current_time = time.time()
            percentage = min(100, max(0, int((current / max(1, total)) * 100)))
            
            # Update logic dengan throttling
            should_update = (
                step != self.last_update['step'] or
                percentage in [0, 25, 50, 75, 100] or
                (current_time - self.last_update['time'] > 0.5)
            )
            
            if should_update:
                self.last_update.update({
                    'time': current_time, 
                    'percentage': percentage, 
                    'step': step
                })
                
                # Log progress
                from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
                step_emoji = {
                    'overall': '🎯', 
                    'step': '📊', 
                    'current': '⚡'
                }.get(step, '📈')
                
                progress_msg = f"{step_emoji} {step.title()}: {percentage}% - {message}"
                log_to_ui(self.ui_components, progress_msg, 'info')
                
                # Update tracker
                self.update_tracker(step, percentage, message)
        
        return callback
    
    def update_tracker(self, level: str, percentage: int, message: str):
        """Update progress tracker dengan new API"""
        try:
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker:
                if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(percentage, message)
                elif level == 'step' and hasattr(progress_tracker, 'update_step'):
                    progress_tracker.update_step(percentage, message)
                elif level == 'current' and hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(percentage, message)
                elif hasattr(progress_tracker, 'update'):
                    progress_tracker.update(level, percentage, message)
        except Exception:
            pass
    
    def show_for_operation(self, operation_name: str):
        """Show progress untuk operation tertentu"""
        try:
            config = self.ui_components.get('config', {})
            progress_config = config.get('progress', {}).get('operations', {})
            
            op_config = progress_config.get(operation_name, {
                'steps': ["prepare", "process", "complete"],
                'weights': {"prepare": 20, "process": 60, "complete": 20}
            })
            
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'show'):
                progress_tracker.show(
                    operation=operation_name.replace('_', ' ').title(),
                    steps=op_config.get('steps'),
                    step_weights=op_config.get('weights')
                )
        except Exception:
            pass
    
    def complete_operation(self, message: str):
        """Complete operation dengan success message"""
        try:
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'complete'):
                progress_tracker.complete(message)
        except Exception:
            pass
    
    def error_operation(self, message: str):
        """Error operation dengan error message"""
        try:
            progress_tracker = self.ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'error'):
                progress_tracker.error(message)
        except Exception:
            pass

# Factory function
def create_progress_manager(ui_components: Dict[str, Any]) -> AugmentationProgressManager:
    """Factory untuk progress manager"""
    return AugmentationProgressManager(ui_components)