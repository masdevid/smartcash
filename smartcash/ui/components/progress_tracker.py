"""
File: smartcash/ui/components/progress_tracker.py
Deskripsi: Fixed progress tracker dengan tqdm compatibility untuk Colab environment
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Set, Callable, List, Union, Tuple
import time
import threading
from enum import Enum
from dataclasses import dataclass, field

# Fixed tqdm import untuk Colab compatibility
try:
    from tqdm.notebook import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        TQDM_AVAILABLE = False
        print("⚠️ tqdm tidak tersedia, menggunakan fallback progress display")

class ProgressLevel(Enum):
    """Progress level enum untuk type safety"""
    SINGLE = 1
    DUAL = 2
    TRIPLE = 3

@dataclass
class ProgressConfig:
    """Configuration untuk progress tracking"""
    level: ProgressLevel = ProgressLevel.TRIPLE
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

class FallbackProgressBar:
    """Fallback progress bar jika tqdm tidak tersedia"""
    
    def __init__(self, total=100, desc="Progress", colour="#28a745", **kwargs):
        self.total = total
        self.desc = desc
        self.colour = colour
        self.n = 0
        self.widget = widgets.HTML(self._generate_html())
    
    def _generate_html(self):
        percentage = (self.n / self.total) * 100 if self.total > 0 else 0
        return f"""
        <div style="margin: 5px 0;">
            <div style="font-size: 13px; margin-bottom: 3px; color: #333;">{self.desc}</div>
            <div style="background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                <div style="background: {self.colour}; height: 100%; width: {percentage}%; 
                           transition: width 0.3s ease; border-radius: 4px; 
                           display: flex; align-items: center; justify-content: center;">
                    <span style="color: white; font-size: 11px; font-weight: bold;">
                        {percentage:.0f}%
                    </span>
                </div>
            </div>
        </div>
        """
    
    def update(self, n=1):
        self.n = min(self.total, self.n + n)
        self.widget.value = self._generate_html()
    
    def set_description(self, desc):
        self.desc = desc
        self.widget.value = self._generate_html()
    
    def reset(self, total=None):
        if total is not None:
            self.total = total
        self.n = 0
        self.widget.value = self._generate_html()
    
    def close(self):
        pass
    
    def refresh(self):
        self.widget.value = self._generate_html()

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
                print(f"⚠️ Callback error untuk {event}: {e}")
                callbacks_to_remove.append(callback_id)
        
        # Cleanup one-time callbacks
        for callback_id in callbacks_to_remove:
            self.unregister(callback_id)

class ProgressTracker:
    """Flexible progress tracker dengan dynamic level configuration dan callback system"""
    
    def __init__(self, config: Optional[ProgressConfig] = None):
        self.config = config or ProgressConfig()
        self.callback_manager = CallbackManager()
        self.progress_bars: Dict[str, Union[tqdm, FallbackProgressBar]] = {}
        self.progress_widgets: Dict[str, widgets.Widget] = {}
        self.active_levels: List[str] = []
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        
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
                ProgressBarConfig("primary", f"📊 {self.config.operation} Progress", "📊", "#28a745", 0)
            ],
            ProgressLevel.DUAL: [
                ProgressBarConfig("overall", f"📊 Overall Progress", "📊", "#28a745", 0),
                ProgressBarConfig("current", f"⚡ Current Operation", "⚡", "#ffc107", 1)
            ],
            ProgressLevel.TRIPLE: [
                ProgressBarConfig("overall", f"📊 Overall Progress ({self.config.operation})", "📊", "#28a745", 0),
                ProgressBarConfig("step", f"🔄 Step Progress", "🔄", "#17a2b8", 1),
                ProgressBarConfig("current", f"⚡ Current Operation", "⚡", "#ffc107", 2)
            ]
        }
        
        self.bar_configs = level_configs[self.config.level]
        self.active_levels = [config.name for config in self.bar_configs if config.visible]
    
    def _create_ui_components(self):
        """Create UI components dengan responsive design"""
        # Header dengan level indicator
        level_indicator = {
            ProgressLevel.SINGLE: "Single Progress",
            ProgressLevel.DUAL: "Dual Progress", 
            ProgressLevel.TRIPLE: "Triple Progress"
        }[self.config.level]
        
        self.header_widget = widgets.HTML(
            f"<h4 style='color: #333; margin: 0; font-size: 16px; font-weight: 600;'>"
            f"📊 {level_indicator} Tracker</h4>",
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
        
        # Container untuk progress bars - gunakan VBox untuk fallback compatibility
        self.tqdm_container = widgets.VBox(
            [], layout=widgets.Layout(
                margin='0', width='100%', max_width='100%', 
                overflow='visible', flex='1 1 auto'
            )
        )
        
        # Dynamic container height berdasarkan level
        container_heights = {
            ProgressLevel.SINGLE: '120px',
            ProgressLevel.DUAL: '180px',
            ProgressLevel.TRIPLE: '250px'
        }
        
        self.container = widgets.VBox(
            [self.header_widget, self.status_widget, self.step_info_widget, self.tqdm_container],
            layout=widgets.Layout(
                display='flex', flex_flow='column nowrap', align_items='stretch',
                margin='10px 0', padding='15px', border='1px solid #28a745',
                border_radius='8px', background_color='#f8fff8', width='100%',
                min_height=container_heights[self.config.level],
                max_height='400px', overflow='visible', box_sizing='border-box'
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
        
        # Initialize steps dan weights dari config untuk TRIPLE level
        if self.config.level == ProgressLevel.TRIPLE and not self.config.steps:
            # Provide default steps jika tidak ada
            self.config.steps = ["Initialization", "Processing", "Finalization"]
        
        # Show container dan initialize progress bars
        self.container.layout.display = 'flex'
        self.container.layout.visibility = 'visible'
        self._initialize_progress_bars()
        self._update_step_info()
        
        # Reset state
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        
        self._update_status("🚀 Starting operation...", 'info')
    
    def update(self, level_name: str, progress: int, message: str = "", 
               color: str = None, trigger_callbacks: bool = True):
        """Update specific progress level dengan callback support"""
        if level_name not in self.active_levels:
            return
        
        # Normalize progress value
        progress = max(0, min(100, progress))
        
        # Update progress bar
        if level_name in self.progress_bars:
            self._update_progress_bar(
                self.progress_bars[level_name], progress, level_name, message, color
            )
        
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
        self._cleanup_progress_bars()
        self.progress_values.clear()
        self.progress_messages.clear()
        self.current_step_index = 0
        self.is_complete = False
        self.is_error = False
        self.hide()
        self.callback_manager.trigger('reset')
    
    def hide(self):
        """Hide progress container"""
        self.container.layout.display = 'none'
        self.container.layout.visibility = 'hidden'
        self._cleanup_progress_bars()
    
    # Internal helper methods
    def _initialize_progress_bars(self):
        """Initialize progress bars berdasarkan current configuration"""
        self._cleanup_progress_bars()
        
        for bar_config in self.bar_configs:
            if bar_config.visible:
                self._create_progress_bar(bar_config)
    
    def _create_progress_bar(self, config: ProgressBarConfig):
        """Create individual progress bar dengan tqdm atau fallback"""
        try:
            if TQDM_AVAILABLE:
                # Gunakan tqdm dengan parameter yang safe untuk Colab
                bar = tqdm(
                    total=100, 
                    desc=config.description,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
                    colour=config.color, 
                    ascii=False, 
                    leave=True,
                    disable=False  # Explicitly set disable untuk avoid AttributeError
                )
                self.progress_bars[config.name] = bar
                
                # Untuk tqdm, kita perlu output container
                if not hasattr(self, '_tqdm_output_created'):
                    self.tqdm_container.children = []
                    self._tqdm_output_created = True
                    
            else:
                # Gunakan fallback progress bar
                bar = FallbackProgressBar(
                    total=100,
                    desc=config.description,
                    colour=config.color
                )
                self.progress_bars[config.name] = bar
                self.progress_widgets[config.name] = bar.widget
                
                # Update container children
                children = list(self.tqdm_container.children)
                children.append(bar.widget)
                self.tqdm_container.children = children
                
        except Exception as e:
            print(f"⚠️ Error creating progress bar {config.name}: {e}")
            # Fallback ke FallbackProgressBar
            bar = FallbackProgressBar(
                total=100,
                desc=config.description,
                colour=config.color
            )
            self.progress_bars[config.name] = bar
            self.progress_widgets[config.name] = bar.widget
            
            children = list(self.tqdm_container.children)
            children.append(bar.widget)
            self.tqdm_container.children = children
    
    def _update_progress_bar(self, bar: Union[tqdm, FallbackProgressBar], value: int, 
                           level_name: str, message: str, color: str = None):
        """Update individual progress bar dengan smooth animation"""
        try:
            # Update progress value
            if hasattr(bar, 'n'):
                diff = value - bar.n
                if diff > 0:
                    bar.update(diff)
                elif diff < 0:
                    bar.reset(total=100)
                    bar.update(value)
            
            # Update color jika provided
            if color and hasattr(bar, 'colour'):
                bar.colour = self._normalize_color(color)
                if hasattr(bar, 'refresh'):
                    bar.refresh()
            
            # Update description dengan message
            if message and hasattr(bar, 'set_description'):
                emoji = next((config.emoji for config in self.bar_configs 
                             if config.name == level_name), "📊")
                truncated_message = self._truncate_message(message, 40)
                bar.set_description(f"{emoji} {truncated_message}")
                
        except Exception as e:
            print(f"⚠️ Error updating progress bar {level_name}: {e}")
    
    def _calculate_weighted_overall_progress(self, step_progress: int) -> int:
        """Calculate weighted overall progress untuk TRIPLE level"""
        if not self.config.steps:
            return step_progress
        
        # Generate equal weights jika tidak ada step_weights
        if not self.config.step_weights:
            self.config.step_weights = self._get_default_weights()
        
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
            if 'step' in self.progress_bars:
                self.update('step', 0, f"Starting {self.config.steps[self.current_step_index]}", 
                          trigger_callbacks=False)
    
    def _update_step_info(self):
        """Update step information display untuk TRIPLE level"""
        if (self.config.level != ProgressLevel.TRIPLE or 
            not self.config.steps or 
            self.current_step_index >= len(self.config.steps)):
            return
        
        current_step = self.config.steps[self.current_step_index]
        weight = self.config.step_weights.get(current_step, 0) if self.config.step_weights else 0
        
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
        for level_name, bar in self.progress_bars.items():
            try:
                if hasattr(bar, 'n'):
                    bar.n = 100
                if hasattr(bar, 'colour'):
                    bar.colour = '#28a745'
                if hasattr(bar, 'refresh'):
                    bar.refresh()
                if hasattr(bar, 'set_description'):
                    emoji = next((config.emoji for config in self.bar_configs 
                                 if config.name == level_name), "📊")
                    bar.set_description(f"✅ {emoji} {self._truncate_message(message, 35)}")
            except Exception as e:
                print(f"⚠️ Error setting complete state for {level_name}: {e}")
    
    def _set_all_bars_error(self, message: str):
        """Set all bars ke error state"""
        for level_name, bar in self.progress_bars.items():
            try:
                if hasattr(bar, 'colour'):
                    bar.colour = '#dc3545'
                if hasattr(bar, 'refresh'):
                    bar.refresh()
                if hasattr(bar, 'set_description'):
                    emoji = next((config.emoji for config in self.bar_configs 
                                 if config.name == level_name), "📊")
                    bar.set_description(f"❌ {emoji} {self._truncate_message(message, 35)}")
            except Exception as e:
                print(f"⚠️ Error setting error state for {level_name}: {e}")
    
    def _cleanup_progress_bars(self):
        """Cleanup all progress bars"""
        for bar in self.progress_bars.values():
            try:
                if hasattr(bar, 'close'):
                    bar.close()
            except Exception:
                pass
        
        self.progress_bars.clear()
        self.progress_widgets.clear()
        
        # Clear container
        try:
            if hasattr(self.tqdm_container, 'clear_output'):
                self.tqdm_container.clear_output(wait=True)
            else:
                self.tqdm_container.children = []
        except Exception:
            pass
    
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
    def _normalize_color(color: str) -> str:
        """Normalize color values"""
        color_map = {
            'success': '#28a745', 'info': '#007bff',
            'warning': '#ffc107', 'error': '#dc3545'
        }
        return color_map.get(color, color)
    
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
    config = ProgressConfig(
        level=ProgressLevel.TRIPLE, operation=operation,
        steps=steps or ["Initialization", "Processing", "Finalization"], 
        step_weights=step_weights or {}
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
        'tqdm_container': tracker.tqdm_container,
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