"""
File: smartcash/ui/components/progress_tracker.py
Deskripsi: Fixed progress tracker dengan auto show/hide dan clean progress display
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Set, Callable, List, Union, Tuple
import time
import threading
from enum import Enum
from dataclasses import dataclass, field

class ProgressLevel(Enum):
    """Progress level enum untuk type safety"""
    SINGLE = 1
    DUAL = 2
    TRIPLE = 3

@dataclass
class ProgressConfig:
    """Configuration untuk progress tracking"""
    level: ProgressLevel = ProgressLevel.SINGLE
    operation: str = "Process"
    steps: List[str] = field(default_factory=list)
    step_weights: Dict[str, int] = field(default_factory=dict)
    auto_advance: bool = True
    auto_hide_delay: float = 3.0
    animation_speed: float = 0.1
    width_adjustment: int = 0

@dataclass
class ProgressBarConfig:
    """Configuration untuk individual progress bar"""
    name: str
    description: str
    emoji: str
    color: str
    position: int
    visible: bool = True

class CallbackManager:
    """Manager untuk handling callbacks dengan type safety"""
    
    def __init__(self):
        self.callbacks: Dict[str, List[Callable]] = {}
        self.one_time_callbacks: Set[str] = set()
    
    def register(self, event: str, callback: Callable, one_time: bool = False) -> str:
        """Register callback dengan automatic cleanup"""
        callback_id = f"{event}_{id(callback)}_{time.time()}"
        self.callbacks.setdefault(event, []).append((callback_id, callback))
        one_time and self.one_time_callbacks.add(callback_id)
        return callback_id
    
    def unregister(self, callback_id: str):
        """Unregister specific callback dengan cleanup"""
        for event, callback_list in self.callbacks.items():
            self.callbacks[event] = [cb for cb in callback_list if cb[0] != callback_id]
        self.one_time_callbacks.discard(callback_id)
    
    def trigger(self, event: str, *args, **kwargs):
        """Trigger callbacks dengan error handling"""
        if event not in self.callbacks:
            return
        
        callbacks_to_remove = []
        for callback_id, callback in self.callbacks[event][:]:
            try:
                callback(*args, **kwargs)
                if callback_id in self.one_time_callbacks:
                    callbacks_to_remove.append(callback_id)
            except Exception as e:
                print(f"Callback error for {event}: {e}")
                callbacks_to_remove.append(callback_id)
        
        # Cleanup one-time callbacks
        for callback_id in callbacks_to_remove:
            self.unregister(callback_id)

class ProgressTracker:
    """Fixed progress tracker dengan auto show/hide dan clean display"""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        self.config = config or ProgressConfig()
        self.callback_manager = CallbackManager()
        self.progress_bars: Dict[str, widgets.HTML] = {}
        self.active_levels: List[str] = []
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        self.is_visible = False
        
        # Progress state tracking
        self.progress_values: Dict[str, int] = {}
        self.progress_messages: Dict[str, str] = {}
        
        self._setup_level_configuration()
        self._create_ui_components()
        self._register_default_callbacks()
    
    def _setup_level_configuration(self):
        """Setup level configuration berdasarkan ProgressLevel"""
        level_configs = {
            ProgressLevel.SINGLE: [
                ProgressBarConfig("primary", "Progress", "📊", "#28a745", 0)
            ],
            ProgressLevel.DUAL: [
                ProgressBarConfig("overall", "Overall Progress", "📊", "#28a745", 0),
                ProgressBarConfig("current", "Current Operation", "⚡", "#ffc107", 1)
            ],
            ProgressLevel.TRIPLE: [
                ProgressBarConfig("overall", "Overall Progress", "📊", "#28a745", 0),
                ProgressBarConfig("step", "Step Progress", "🔄", "#17a2b8", 1),
                ProgressBarConfig("current", "Current Operation", "⚡", "#ffc107", 2)
            ]
        }
        
        self.bar_configs = level_configs[self.config.level]
        self.active_levels = [config.name for config in self.bar_configs if config.visible]
    
    def _create_ui_components(self):
        """Create UI components dengan responsive design"""
        # Header dengan operation name (tanpa level indicator)
        self.header_widget = widgets.HTML(
            "",  # Empty by default
            layout=widgets.Layout(margin='0 0 10px 0', width='100%')
        )
        
        self.status_widget = widgets.HTML(
            "", layout=widgets.Layout(margin='0 0 8px 0', width='100%')
        )
        
        # Step info hanya untuk TRIPLE level
        self.step_info_widget = widgets.HTML(
            "", layout=widgets.Layout(
                margin='0 0 5px 0', width='100%',
                display='block' if self.config.level == ProgressLevel.TRIPLE else 'none'
            )
        )
        
        # Initialize progress bars sebagai HTML widgets
        self._initialize_progress_bars()
        
        # Dynamic container height berdasarkan level
        container_heights = {
            ProgressLevel.SINGLE: '100px',
            ProgressLevel.DUAL: '150px',
            ProgressLevel.TRIPLE: '200px'
        }
        
        # Combine all progress bars
        progress_bars_list = [self.progress_bars[level] for level in self.active_levels]
        
        # Container hidden by default
        self.container = widgets.VBox(
            [self.header_widget, self.status_widget, self.step_info_widget] + progress_bars_list,
            layout=widgets.Layout(
                display='none', flex_flow='column nowrap', align_items='stretch',
                margin='10px 0', padding='15px', border='1px solid #28a745',
                border_radius='8px', background_color='#f8fff8', width='100%',
                min_height=container_heights[self.config.level],
                max_height='300px', overflow='hidden', box_sizing='border-box'
            )
        )
    
    def _register_default_callbacks(self):
        """Register default callbacks untuk common operations"""
        # Auto-advance callback untuk TRIPLE level
        if self.config.level == ProgressLevel.TRIPLE and self.config.auto_advance:
            self.on_step_complete(self._auto_advance_step)
        
        # Auto-hide callback
        self.on_complete(lambda: self._delayed_hide())
        
        # Progress sync callbacks
        self.on_progress_update(self._sync_progress_state)
    
    # Callback registration methods
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
    
    # Main interface methods
    def show(self, operation: str = None, steps: List[str] = None, 
             step_weights: Dict[str, int] = None, level: ProgressLevel = None):
        """Show progress tracker dengan dynamic configuration"""
        # Update configuration jika provided
        if operation:
            self.config.operation = operation
        if steps:
            self.config.steps = steps
        if step_weights:
            self.config.step_weights = step_weights
        if level:
            self.config.level = level
            self._setup_level_configuration()
            self._create_ui_components()
        
        # Update header dengan operation name (tanpa duplikasi icon)
        self.header_widget.value = f"""<h4 style='color: #333; margin: 0; font-size: 16px; font-weight: 600;'>
        📊 {self.config.operation}</h4>"""
        
        # Initialize steps dan weights dari config
        if not self.config.steps and self.config.level == ProgressLevel.TRIPLE:
            self.config.steps = ["Step 1", "Step 2", "Step 3"]
            self.config.step_weights = self._get_default_weights()
        
        # Show container dan initialize progress bars
        self.container.layout.display = 'flex'
        self.container.layout.visibility = 'visible'
        self.is_visible = True
        self._initialize_progress_bars()
        self._update_step_info()
        
        # Reset state
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        
        self._update_status("🚀 Starting operation...", 'info')
    
    def update(self, level_name: str, progress: int, message: str = "", 
               color: str = None, trigger_callbacks: bool = True):
        """Update specific progress level dengan auto show dan callback support"""
        if level_name not in self.active_levels:
            return
        
        # Auto show jika belum visible
        if not self.is_visible:
            self.container.layout.display = 'flex'
            self.container.layout.visibility = 'visible'
            self.is_visible = True
        
        # Normalize progress value
        progress = max(0, min(100, progress))
        
        # Update progress bar
        if level_name in self.progress_bars:
            self._update_progress_bar(level_name, progress, message, color)
        
        # Update state tracking
        self.progress_values[level_name] = progress
        if message:
            self.progress_messages[level_name] = message
        
        # Trigger callbacks
        if trigger_callbacks:
            self.callback_manager.trigger('progress_update', level_name, progress, message)
            
            # Check for step completion (TRIPLE level only)
            if (self.config.level == ProgressLevel.TRIPLE and 
                level_name == 'step' and progress >= 100):
                self.callback_manager.trigger('step_complete', 
                                            self.config.steps[self.current_step_index], 
                                            self.current_step_index)
    
    # Convenience methods untuk different levels
    def update_primary(self, progress: int, message: str = "", color: str = None):
        """Update primary progress (SINGLE level)"""
        self.update('primary', progress, message, color)
    
    def update_overall(self, progress: int, message: str = "", color: str = None):
        """Update overall progress (DUAL/TRIPLE level)"""
        if self.config.level == ProgressLevel.TRIPLE:
            # Calculate overall progress berdasarkan step weights
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
        self._set_all_bars_complete(message)
        self._update_status(f"✅ {message}", 'success')
        
        if self.config.level == ProgressLevel.TRIPLE:
            self.current_step_index = len(self.config.steps)
            self._update_step_info()
        
        self.callback_manager.trigger('complete')
    
    def error(self, message: str = "Operation failed"):
        """Set error state dengan callback triggering"""
        if self.is_error:
            return
        
        self.is_error = True
        self._set_all_bars_error(message)
        self._update_status(f"❌ {message}", 'error')
        self.callback_manager.trigger('error', message)
    
    def reset(self):
        """Reset tracker dengan complete cleanup"""
        self.progress_values.clear()
        self.progress_messages.clear()
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        self._initialize_progress_bars()
        self.hide()
        self.callback_manager.trigger('reset')
    
    def hide(self):
        """Hide progress container"""
        self.container.layout.display = 'none'
        self.container.layout.visibility = 'hidden'
        self.is_visible = False
    
    # Internal helper methods
    def _initialize_progress_bars(self):
        """Initialize progress bars sebagai HTML widgets"""
        for bar_config in self.bar_configs:
            if bar_config.visible:
                self.progress_bars[bar_config.name] = widgets.HTML(
                    value="", 
                    layout=widgets.Layout(width='100%', margin='2px 0')
                )
                # Initialize dengan 0 progress (tanpa duplikasi icon)
                self._update_progress_bar(bar_config.name, 0, bar_config.description)
    
    def _update_progress_bar(self, level_name: str, value: int, message: str = "", color: str = None):
        """Update individual progress bar dengan HTML/CSS (tanpa duplikasi icon)"""
        if level_name not in self.progress_bars:
            return
            
        # Get config untuk level
        config = next((c for c in self.bar_configs if c.name == level_name), None)
        if not config:
            return
        
        # Use provided color atau default
        bar_color = color or config.color
        
        # Clean message - hilangkan emoji yang duplikat
        display_message = self._clean_message(message or config.description)
        
        # Generate HTML untuk progress bar (tanpa emoji duplikat)
        bar_html = f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 14px; font-weight: 500; color: #333;">
                    {config.emoji} {self._truncate_message(display_message, 40)}
                </span>
                <span style="font-size: 12px; color: #666;">{value}%</span>
            </div>
            <div style="background: #e9ecef; border-radius: 10px; overflow: hidden; height: 16px;">
                <div style="background: {bar_color}; height: 100%; width: {value}%; 
                           transition: width 0.3s ease; border-radius: 10px;"></div>
            </div>
        </div>
        """
        
        self.progress_bars[level_name].value = bar_html
    
    def _clean_message(self, message: str) -> str:
        """Clean message dari emoji duplikat"""
        import re
        # Remove leading emoji yang mungkin duplikat
        cleaned = re.sub(r'^[📊🔄⚡🔍📥☁️✅❌]+\s*', '', message)
        return cleaned.strip() or message
    
    def _calculate_weighted_overall_progress(self, step_progress: int) -> int:
        """Calculate weighted overall progress untuk TRIPLE level"""
        if not self.config.steps or not self.config.step_weights:
            return step_progress
        
        # Calculate completed steps weight
        completed_weight = sum(
            self.config.step_weights.get(step, 0) 
            for step in self.config.steps[:self.current_step_index]
        )
        
        # Calculate current step contribution
        current_step = (self.config.steps[self.current_step_index] 
                       if self.current_step_index < len(self.config.steps) else '')
        current_weight = self.config.step_weights.get(current_step, 0)
        current_contribution = (step_progress / 100) * current_weight
        
        # Calculate total progress
        total_weight = sum(self.config.step_weights.values())
        if total_weight == 0:
            return step_progress
        
        return int((completed_weight + current_contribution) / total_weight * 100)
    
    def _auto_advance_step(self, step_name: str, step_index: int):
        """Auto advance ke step berikutnya"""
        if self.current_step_index < len(self.config.steps) - 1:
            self.current_step_index += 1
            self._update_step_info()
            # Reset step progress
            self.update('step', 0, f"Starting {self.config.steps[self.current_step_index]}", 
                      trigger_callbacks=False)
    
    def _update_step_info(self):
        """Update step information display untuk TRIPLE level"""
        if (self.config.level != ProgressLevel.TRIPLE or 
            not self.config.steps or 
            self.current_step_index >= len(self.config.steps)):
            return
        
        current_step = self.config.steps[self.current_step_index]
        weight = self.config.step_weights.get(current_step, 0)
        
        step_info = f"""
        <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin: 2px 0;">
            <small style="color: #1976d2;">
                <strong>Step {self.current_step_index + 1}/{len(self.config.steps)}:</strong> 
                {current_step.title()} 
                <span style="color: #666;">(Weight: {weight}%)</span>
            </small>
        </div>
        """
        self.step_info_widget.value = step_info
    
    def _set_all_bars_complete(self, message: str):
        """Set all bars ke complete state"""
        for level_name in self.progress_bars:
            self._update_progress_bar(level_name, 100, f"✅ {self._truncate_message(message, 35)}", '#28a745')
    
    def _set_all_bars_error(self, message: str):
        """Set all bars ke error state"""
        for level_name in self.progress_bars:
            self._update_progress_bar(level_name, self.progress_values.get(level_name, 0), 
                                    f"❌ {self._truncate_message(message, 35)}", '#dc3545')
    
    def _delayed_hide(self):
        """Hide container after delay"""
        def hide_after_delay():
            time.sleep(self.config.auto_hide_delay)
            if self.is_complete and not self.is_error:
                self.hide()
        
        threading.Thread(target=hide_after_delay, daemon=True).start()
    
    def _sync_progress_state(self, level_name: str, progress: int, message: str):
        """Sync progress state untuk internal tracking"""
        # Update status message jika significant progress
        if progress > 0 and message:
            style = 'success' if progress >= 100 else 'info'
            self._update_status(message, style)
    
    def _update_status(self, message: str, style: str = None):
        """Update status message dengan styling"""
        color_map = {
            'success': '#28a745', 'info': '#007bff', 
            'warning': '#ffc107', 'error': '#dc3545'
        }
        color = color_map.get(style, '#495057')
        
        self.status_widget.value = f"""
        <div style="color: {color}; font-size: 13px; font-weight: 500; margin: 0; 
                    padding: 8px 12px; background: rgba(233, 236, 239, 0.5); 
                    border-radius: 6px; border-left: 3px solid {color}; 
                    width: 100%; box-sizing: border-box; word-wrap: break-word; 
                    overflow-wrap: break-word; line-height: 1.4; 
                    display: flex; align-items: center;">
            {message}
        </div>
        """
    
    def _get_default_weights(self) -> Dict[str, int]:
        """Generate equal weights untuk semua steps"""
        if not self.config.steps:
            return {}
        
        num_steps = len(self.config.steps)
        base_weight = 100 // num_steps
        remainder = 100 % num_steps
        
        weights = {}
        for i, step in enumerate(self.config.steps):
            weights[step] = base_weight + (1 if i < remainder else 0)
        
        return weights
    
    @staticmethod
    def _truncate_message(message: str, max_length: int) -> str:
        """Truncate message dengan ellipsis"""
        return message if len(message) <= max_length else f"{message[:max_length-3]}..."

# Factory functions untuk different use cases
def create_single_progress_tracker(operation: str = "Process") -> ProgressTracker:
    """Create single-level progress tracker"""
    config = ProgressConfig(level=ProgressLevel.SINGLE, operation=operation)
    return ProgressTracker(config)

def create_dual_progress_tracker(operation: str = "Process") -> ProgressTracker:
    """Create dual-level progress tracker"""
    config = ProgressConfig(level=ProgressLevel.DUAL, operation=operation)
    return ProgressTracker(config)

def create_triple_progress_tracker(operation: str = "Process", 
                                 steps: List[str] = None,
                                 step_weights: Dict[str, int] = None) -> ProgressTracker:
    """Create triple-level progress tracker"""
    steps = steps or ["Initialization", "Processing", "Completion"]
    
    config = ProgressConfig(
        level=ProgressLevel.TRIPLE, operation=operation,
        steps=steps, step_weights=step_weights or {}
    )
    return ProgressTracker(config)

def create_flexible_tracker(config: ProgressConfig) -> ProgressTracker:
    """Create tracker dengan custom configuration"""
    return ProgressTracker(config)

# Backward compatibility
def create_three_progress_tracker() -> Dict[str, Any]:
    """Backward compatibility untuk existing code"""
    tracker = create_triple_progress_tracker()
    return {
        'container': tracker.container,
        'progress_container': tracker.container,
        'status_widget': tracker.status_widget,
        'step_info_widget': tracker.step_info_widget,
        'tracker': tracker,
        'show_container': tracker.show,
        'hide_container': tracker.hide,
        'show_for_operation': tracker.show,
        'update_overall': tracker.update_overall,
        'update_step': tracker.update_step,
        'update_current': tracker.update_current,
        'update_progress': tracker.update,
        'complete_operation': tracker.complete,
        'error_operation': tracker.error,
        'reset_all': tracker.reset
    }