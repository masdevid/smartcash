"""
File: smartcash/ui/components/progress_tracker/progress_tracker.py
Deskripsi: Main progress tracker tanpa step info dan dengan auto hide 1 jam
"""

import time
from typing import Dict, List, Optional, Callable
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.callback_manager import CallbackManager
from smartcash.ui.components.progress_tracker.tqdm_manager import TqdmManager
from smartcash.ui.components.progress_tracker.ui_components import UIComponentsManager

class ProgressTracker:
    """Main progress tracker tanpa step info dengan auto hide dan layout [message][bar][percentage]"""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        self.config = config or ProgressConfig()
        self.callback_manager = CallbackManager()
        self.ui_manager = UIComponentsManager(self.config)
        self.tqdm_manager = TqdmManager(self.ui_manager)  # Pass ui_manager instead of single output
        
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
        """Dummy property untuk backward compatibility - always None"""
        return None
    
    @property
    def progress_bars(self):
        """Expose tqdm_bars untuk backward compatibility"""
        if not hasattr(self.tqdm_manager, 'tqdm_bars'):
            return {'main': None}
            
        if self.config.level == ProgressLevel.SINGLE and self.tqdm_manager.tqdm_bars:
            first_key = next(iter(self.tqdm_manager.tqdm_bars))
            return {'main': self.tqdm_manager.tqdm_bars.get(first_key)}
        
        return self.tqdm_manager.tqdm_bars
    
    def _register_default_callbacks(self):
        """Register default callbacks tanpa step logic"""
        self.on_complete(lambda: self._delayed_hide())
        self.on_progress_update(self._sync_progress_state)
    
    def on_progress_update(self, callback: Callable[[str, int, str], None]) -> str:
        """Register callback untuk progress updates"""
        return self.callback_manager.register('progress_update', callback)
    
    def on_step_complete(self, callback: Callable[[str, int], None]) -> str:
        """Register callback untuk step completion (untuk compatibility)"""
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
             step_weights: Dict[str, int] = None, level: ProgressLevel = None,
             auto_hide: bool = None):
        """Show progress tracker dengan optional auto hide override"""
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
        if auto_hide is not None:
            self.config.auto_hide = auto_hide
        
        self.ui_manager.update_header(self.config.operation)
        
        self.ui_manager.show()
        self.tqdm_manager.initialize_bars(self.bar_configs)
        
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        
        self.ui_manager.update_status("🚀 Starting operation...", 'info')
    
    def update(self, level_name: str, progress: int, message: str = "", 
               color: str = None, trigger_callbacks: bool = True):
        """Update progress dengan format [message][bar][percentage]"""
        if level_name not in self.active_levels:
            return
        
        if not self.ui_manager.is_visible:
            self.ui_manager.show()
        
        progress = max(0, min(100, progress))
        self.tqdm_manager.update_bar(level_name, progress, message, self.bar_configs)
        
        if trigger_callbacks:
            self.callback_manager.trigger('progress_update', level_name, progress, message)
    
    def update_primary(self, progress: int, message: str = "", color: str = None):
        """Update primary progress (SINGLE level)"""
        self.update('primary', progress, message, color)
    
    def update_overall(self, progress: int, message: str = "", color: str = None):
        """Update overall progress (DUAL/TRIPLE level)"""
        self.update('overall', progress, message, color)
    
    def update_step(self, progress: int, message: str = "", color: str = None):
        """Update step progress (TRIPLE level only)"""
        if self.config.level == ProgressLevel.TRIPLE:
            self.update('step', progress, message, color)
    
    def update_current(self, progress: int, message: str = "", color: str = None):
        """Update current operation progress (DUAL/TRIPLE level)"""
        self.update('current', progress, message, color)
    
    def complete(self, message: str = "Operation completed successfully!"):
        """Complete operation dengan auto hide timer"""
        if self.is_complete:
            return
        
        self.is_complete = True
        self.tqdm_manager.set_all_complete(message, self.bar_configs)
        self.ui_manager.update_status(f"✅ {message}", 'success')
        
        self.callback_manager.trigger('complete')
    
    def error(self, message: str = "Operation failed"):
        """Set error state dan cancel auto hide"""
        if self.is_error:
            return
        
        self.is_error = True
        self.tqdm_manager.set_all_error(message)
        self.ui_manager.update_status(f"❌ {message}", 'error')
        # Cancel auto hide saat error
        self.ui_manager._cancel_auto_hide_timer()
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
        """Hide progress container dan cancel auto hide timer"""
        self.ui_manager.hide()
    
    def _delayed_hide(self):
        """Hide sesuai dengan auto_hide setting"""
        if self.is_complete and not self.is_error and not self.config.auto_hide:
            # Jika auto_hide disabled, tetap tampilkan sampai manual hide
            return
        # Auto hide akan dihandle oleh UIComponentsManager
    
    def _sync_progress_state(self, level_name: str, progress: int, message: str):
        """Sync progress state untuk internal tracking"""
        if progress > 0 and message:
            style = 'success' if progress >= 100 else 'info'
            clean_message = self.tqdm_manager._clean_message(message)
            self.ui_manager.update_status(clean_message, style)