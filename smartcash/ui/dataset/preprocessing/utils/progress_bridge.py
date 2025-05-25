"""
File: smartcash/ui/dataset/preprocessing/utils/progress_bridge.py
Deskripsi: Fixed bridge untuk menghubungkan service layer progress dengan UI - resolved parameter conflicts
"""

from typing import Dict, Any, Callable

class PreprocessingProgressBridge:
    """Fixed bridge untuk mapping service progress ke UI progress components - no parameter conflicts."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
    
    def notify_progress(self, **kwargs) -> None:
        """Fixed central progress notification dengan resolved parameter conflicts."""
        try:
            # FIXED: Extract dengan safe parameter handling untuk avoid conflicts
            progress_value = kwargs.get('progress', kwargs.get('overall_progress', 0))
            step_value = kwargs.get('step', 0)
            message_value = kwargs.get('message', 'Processing...')
            current_progress_value = kwargs.get('current_progress', 0)
            
            # Map to multi-level progress dengan clean parameters
            self._update_overall_progress(progress_value, message_value)
            self._update_step_progress(step_value, kwargs.get('split_step', ''))
            self._update_current_progress(current_progress_value, kwargs.get('split', ''))
            
        except Exception as e:
            self.logger and self.logger.debug(f"🔧 Progress bridge error: {str(e)}")
    
    def _update_overall_progress(self, progress_val: int, message_val: str) -> None:
        """Update overall progress dengan bounds checking."""
        update_fn = self.ui_components.get('update_progress')
        if update_fn:
            # FIXED: Pass positional arguments untuk avoid keyword conflicts
            update_fn('overall', max(0, min(100, progress_val)), message_val)
    
    def _update_step_progress(self, step_val: int, step_name: str) -> None:
        """Update step progress dengan descriptive messaging."""
        update_fn = self.ui_components.get('update_progress')
        if update_fn and step_val > 0:
            step_messages = {1: "Validasi", 2: "Pemrosesan", 3: "Finalisasi"}
            message_text = f"{step_messages.get(step_val, 'Step')} {step_name}".strip()
            # FIXED: Pass positional arguments untuk avoid keyword conflicts
            update_fn('step', min(100, step_val * 33), message_text)
    
    def _update_current_progress(self, progress_val: int, context: str) -> None:
        """Update current progress untuk detail operations."""
        update_fn = self.ui_components.get('update_progress')
        if update_fn and progress_val > 0:
            message_text = f"Processing {context}" if context else "Processing"
            # FIXED: Pass positional arguments untuk avoid keyword conflicts
            update_fn('current', max(0, min(100, progress_val)), message_text)

def create_preprocessing_progress_bridge(ui_components: Dict[str, Any]) -> PreprocessingProgressBridge:
    """Factory untuk membuat progress bridge instance."""
    return PreprocessingProgressBridge(ui_components)