"""
File: smartcash/ui/components/progress_tracker/progress_tracker.py
Deskripsi: Main progress tracker class dengan tqdm integration dan callback system
"""

import time
from typing import Dict, List, Optional, Callable
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.callback_manager import CallbackManager
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager
from smartcash.ui.components.progress_tracker.ui_components import UIComponentsManager

class ProgressTracker:
    """Main progress tracker dengan tqdm integration dan callback system"""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        self.config = config or ProgressConfig()
        self.callback_manager = CallbackManager()
        self.ui_manager = UIComponentsManager(self.config)
        self.tqdm_manager = TqdmManager(self.ui_manager.progress_output)
        
        self.bar_configs = self.config.get_level_configs()
        self.active_levels = [config.name for config in self.bar_configs if config.visible]
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        
        self._register_default_callbacks()
    
    @property
    def container(self):
        """Expose container untuk backward compatibility"""
        return self.ui_manager.container
    
    @property
    def status_widget(self):
        """Expose status widget untuk backward compatibility"""
        return self.ui_manager.status_widget
    
    @property
    def step_info_widget(self):
        """Expose step info widget untuk backward compatibility"""
        return self.ui_manager.step_info_widget
    
    def _register_default_callbacks(self):
        """Register default callbacks untuk common operations"""
        if self.config.level == ProgressLevel.TRIPLE and self.config.auto_advance:
            self.on_step_complete(self._auto_advance_step)
        self.on_complete(lambda: self._delayed_hide())
        self.on_progress_update(self._sync_progress_state)
    
    def on_progress_update(self, callback: Callable[[str, int, str], None]) -> str:
        """Register callback untuk progress updates"""
        return self.callback_manager.register('progress_update', callback)
    
    def on_step_complete(self, callback: Callable[[str, int], None]) -> str:
        """Register callback untuk step completion (TRIPLE level only)"""
        return self.callback_manager.register('step_complete', callback)
    
    def on_complete(self, callback: Callable[[], None]) -> str:
        """Register callback untuk operation completion"""
        return self.callback_manager.register('complete', callback)
    
    def on_error(self, callback: Callable[[str], None]) -> str:
        """Register callback untuk error events"""
        return self.callback_manager.register('error', callback)
    
    def on_reset(self, callback: Callable[[], None]) -> str:
        """Register callback untuk reset events"""
        return self.callback_manager.register('reset', callback)
    
    def remove_callback(self, callback_id: str):
        """Remove specific callback"""
        self.callback_manager.unregister(callback_id)
    
    def show(self, operation: str = None, steps: List[str] = None, 
             step_weights: Dict[str, int] = None, level: ProgressLevel = None):
        """Show progress tracker dengan dynamic configuration"""
        if operation:
            self.config.operation = operation
        if steps:
            self.config.steps = steps
        if step_weights:
            self.config.step_weights = step_weights
        if level and level != self.config.level:
            self.config.level = level
            self.bar_configs = self.config.get_level_configs()
            self.active_levels = [config.name for config in self.bar_configs if config.visible]
        
        self.ui_manager.update_header(self.config.operation)
        
        if not self.config.steps and self.config.level == ProgressLevel.TRIPLE:
            self.config.steps = ["Step 1", "Step 2", "Step 3"]
            self.config.step_weights = self.config.get_default_weights()
        
        self.ui_manager.show()
        self.tqdm_manager.initialize_bars(self.bar_configs)
        self._update_step_info()
        
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        
        self.ui_manager.update_status("🚀 Starting operation...", 'info')
    
    def update(self, level_name: str, progress: int, message: str = "", 
               color: str = None, trigger_callbacks: bool = True):
        """Update specific progress level dengan tqdm dan callback support"""
        if level_name not in self.active_levels:
            return
        
        if not self.ui_manager.is_visible:
            self.ui_manager.show()
        
        progress = max(0, min(100, progress))
        self.tqdm_manager.update_bar(level_name, progress, message, self.bar_configs)
        
        if trigger_callbacks:
            self.callback_manager.trigger('progress_update', level_name, progress, message)
            
            if (self.config.level == ProgressLevel.TRIPLE and 
                level_name == 'step' and progress >= 100):
                self.callback_manager.trigger('step_complete', 
                                            self.config.steps[self.current_step_index], 
                                            self.current_step_index)
    
    def update_primary(self, progress: int, message: str = "", color: str = None):
        """Update primary progress (SINGLE level)"""
        self.update('primary', progress, message, color)
    
    def update_overall(self, progress: int, message: str = "", color: str = None):
        """Update overall progress (DUAL/TRIPLE level)"""
        if self.config.level == ProgressLevel.TRIPLE:
            progress = self._calculate_weighted_overall_progress(progress)
        self.update('overall', progress, message, color)
    
    def update_step(self, progress: int, message: str = "", color: str = None):
        """Update step progress (TRIPLE level only)"""
        if self.config.level == ProgressLevel.TRIPLE:
            self.update('step', progress, message, color)
    
    def update_current(self, progress: int, message: str = "", color: str = None):
        """Update current operation progress (DUAL/TRIPLE level)"""
        self.update('current', progress, message, color)
    
    def complete(self, message: str = "Operation completed successfully!"):
        """Complete operation dengan callback triggering"""
        if self.is_complete:
            return
        
        self.is_complete = True
        self.tqdm_manager.set_all_complete(message, self.bar_configs)
        self.ui_manager.update_status(f"✅ {message}", 'success')
        
        if self.config.level == ProgressLevel.TRIPLE:
            self.current_step_index = len(self.config.steps)
            self._update_step_info()
        
        self.callback_manager.trigger('complete')
    
    def error(self, message: str = "Operation failed"):
        """Set error state dengan callback triggering"""
        if self.is_error:
            return
        
        self.is_error = True
        self.tqdm_manager.set_all_error(message)
        self.ui_manager.update_status(f"❌ {message}", 'error')
        self.callback_manager.trigger('error', message)
    
    def reset(self):
        """Reset tracker dengan complete cleanup"""
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        
        self.tqdm_manager.reset()
        self.hide()
        self.callback_manager.trigger('reset')
    
    def hide(self):
        """Hide progress container"""
        self.ui_manager.hide()
    
    def _calculate_weighted_overall_progress(self, step_progress: int) -> int:
        """Calculate weighted overall progress untuk TRIPLE level"""
        if not self.config.steps or not self.config.step_weights:
            return step_progress
        
        completed_weight = sum(
            self.config.step_weights.get(step, 0) 
            for step in self.config.steps[:self.current_step_index]
        )
        
        current_step = (self.config.steps[self.current_step_index] 
                       if self.current_step_index < len(self.config.steps) else '')
        current_weight = self.config.step_weights.get(current_step, 0)
        current_contribution = (step_progress / 100) * current_weight
        
        total_weight = sum(self.config.step_weights.values())
        if total_weight == 0:
            return step_progress
        
        return int((completed_weight + current_contribution) / total_weight * 100)
    
    def _auto_advance_step(self, step_name: str, step_index: int):
        """Auto advance ke step berikutnya"""
        if self.current_step_index < len(self.config.steps) - 1:
            self.current_step_index += 1
            self._update_step_info()
            self.update('step', 0, f"Starting {self.config.steps[self.current_step_index]}", 
                      trigger_callbacks=False)
    
    def _update_step_info(self):
        """Update step information display untuk TRIPLE level"""
        self.ui_manager.update_step_info(
            self.current_step_index, 
            self.config.steps, 
            self.config.step_weights
        )
    
    def _delayed_hide(self):
        """Hide container after delay - immediate hide untuk avoid threading"""
        if self.is_complete and not self.is_error:
            self.hide()
    
    def _sync_progress_state(self, level_name: str, progress: int, message: str):
        """Sync progress state untuk internal tracking"""
        if progress > 0 and message:
            style = 'success' if progress >= 100 else 'info'
            clean_message = self.tqdm_manager._clean_message(message)
            self.ui_manager.update_status(clean_message, style)