"""
File: smartcash/ui/dataset/preprocessing/utils/progress_bridge.py
Deskripsi: Bridge untuk menghubungkan service layer progress dengan UI progress tracking
"""

from typing import Dict, Any, Callable

class PreprocessingProgressBridge:
    """Bridge untuk mapping service progress ke UI progress components."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
    
    def notify_progress(self, **kwargs) -> None:
        """Central progress notification untuk semua service layers."""
        try:
            # Extract progress data
            overall = kwargs.get('progress', 0)
            step = kwargs.get('step', 0)
            message = kwargs.get('message', 'Processing...')
            
            # Map to multi-level progress
            self._update_overall_progress(overall, message)
            self._update_step_progress(step, kwargs.get('split_step', ''))
            self._update_current_progress(kwargs.get('current_progress', 0), 
                                        kwargs.get('split', ''))
        except Exception as e:
            self.logger and self.logger.debug(f"🔧 Progress bridge error: {str(e)}")
    
    def _update_overall_progress(self, progress: int, message: str) -> None:
        """Update overall progress dengan bounds checking."""
        update_fn = self.ui_components.get('update_progress')
        if update_fn:
            update_fn('overall', max(0, min(100, progress)), message)
    
    def _update_step_progress(self, step: int, step_name: str) -> None:
        """Update step progress dengan descriptive messaging."""
        update_fn = self.ui_components.get('update_progress')
        if update_fn and step > 0:
            step_messages = {1: "Validasi", 2: "Pemrosesan", 3: "Finalisasi"}
            message = f"{step_messages.get(step, 'Step')} {step_name}".strip()
            update_fn('step', min(100, step * 33), message)
    
    def _update_current_progress(self, progress: int, context: str) -> None:
        """Update current progress untuk detail operations."""
        update_fn = self.ui_components.get('update_progress')
        if update_fn and progress > 0:
            message = f"Processing {context}" if context else "Processing"
            update_fn('current', max(0, min(100, progress)), message)

def create_preprocessing_progress_bridge(ui_components: Dict[str, Any]) -> PreprocessingProgressBridge:
    """Factory untuk membuat progress bridge instance."""
    return PreprocessingProgressBridge(ui_components)