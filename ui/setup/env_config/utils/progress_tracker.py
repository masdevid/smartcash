"""
File: smartcash/ui/setup/env_config/utils/progress_tracker.py
Deskripsi: Utility untuk tracking progress dengan step management
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.setup.env_config.constants import PROGRESS_STEPS

class ProgressTracker:
    """🎯 Progress tracker dengan step management dan UI callback"""
    
    def __init__(self, update_callback: Optional[Callable[[str, int, bool], None]] = None):
        self.update_callback = update_callback
        self.current_step = 'start'
        self.current_progress = 0
        self.step_history = []
        
    def start_step(self, step_name: str, custom_message: Optional[str] = None) -> None:
        """🚀 Mulai step baru"""
        if step_name not in PROGRESS_STEPS:
            return
            
        self.current_step = step_name
        step_config = PROGRESS_STEPS[step_name]
        
        # Update ke progress awal step
        self.current_progress = step_config['range'][0]
        message = custom_message or step_config['label']
        
        self.step_history.append({
            'step': step_name,
            'status': 'started',
            'message': message,
            'progress': self.current_progress
        })
        
        if self.update_callback:
            self.update_callback(message, self.current_progress, False)
    
    def update_step_progress(self, progress_within_step: int, message: Optional[str] = None) -> None:
        """📊 Update progress dalam step saat ini"""
        if self.current_step not in PROGRESS_STEPS:
            return
            
        step_config = PROGRESS_STEPS[self.current_step]
        start_range, end_range = step_config['range']
        
        # Calculate absolute progress
        step_progress = max(0, min(100, progress_within_step))
        absolute_progress = start_range + (step_progress * (end_range - start_range) // 100)
        
        self.current_progress = absolute_progress
        display_message = message or step_config['label']
        
        if self.update_callback:
            self.update_callback(display_message, self.current_progress, False)
    
    def complete_step(self, success: bool = True, message: Optional[str] = None) -> None:
        """✅ Complete step saat ini"""
        if self.current_step not in PROGRESS_STEPS:
            return
            
        step_config = PROGRESS_STEPS[self.current_step]
        
        # Update ke progress akhir step
        self.current_progress = step_config['range'][1]
        display_message = message or step_config['label']
        
        # Update history
        if self.step_history:
            self.step_history[-1]['status'] = 'completed' if success else 'failed'
            self.step_history[-1]['final_progress'] = self.current_progress
        
        if self.update_callback:
            self.update_callback(display_message, self.current_progress, not success)
    
    def skip_to_step(self, step_name: str) -> None:
        """⏭️ Skip ke step tertentu"""
        if step_name not in PROGRESS_STEPS:
            return
            
        self.current_step = step_name
        step_config = PROGRESS_STEPS[step_name]
        self.current_progress = step_config['range'][0]
        
        if self.update_callback:
            self.update_callback(step_config['label'], self.current_progress, False)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """📋 Get summary progress"""
        return {
            'current_step': self.current_step,
            'current_progress': self.current_progress,
            'total_steps': len(PROGRESS_STEPS),
            'completed_steps': len([h for h in self.step_history if h.get('status') == 'completed']),
            'step_history': self.step_history
        }
    
    def reset(self) -> None:
        """🔄 Reset progress tracker"""
        self.current_step = 'start'
        self.current_progress = 0
        self.step_history = []