"""
File: smartcash/ui/components/enhanced_progress_tracker.py
Deskripsi: Enhanced configurable progress tracker dengan dynamic level switching dan one-liner optimizations untuk mengganti implementasi lama
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import threading
import time

class ProgressMode(Enum):
    """Progress mode enum untuk type safety dan configurability"""
    SINGLE = 1
    DUAL = 2  
    TRIPLE = 3

class EnhancedProgressTracker:
    """Enhanced configurable progress tracker dengan dynamic level switching dan DRY optimizations"""
    
    def __init__(self, ui_components: Dict[str, Any], mode: ProgressMode = ProgressMode.SINGLE, 
                 auto_hide_delay: float = 3.0, operation_name: str = "Process"):
        self.ui_components, self.mode, self.auto_hide_delay, self.operation_name = ui_components, mode, auto_hide_delay, operation_name
        self.is_visible, self.auto_hide_timer = False, None
        self.progress_values = {'overall': 0, 'step': 0, 'current': 0}
        self.progress_messages = {'overall': '', 'step': '', 'current': ''}
        self.step_weights, self.current_step_index, self.total_steps = {}, 0, 0
        self.progress_bars = {}
        self.hidden_bars = {}
        self._create_ui_components()
    
    def _create_ui_components(self):
        """Create UI components dengan responsive design dan mode-aware visibility"""
        mode_configs = {
            ProgressMode.SINGLE: {'bars': ['overall'], 'title': 'Progress', 'height': '120px'},
            ProgressMode.DUAL: {'bars': ['overall', 'step'], 'title': 'Dual Progress', 'height': '160px'},
            ProgressMode.TRIPLE: {'bars': ['overall', 'step', 'current'], 'title': 'Triple Progress', 'height': '200px'}
        }
        config = mode_configs[self.mode]
        
        # Initialize progress bars untuk semua levels
        all_levels = ['overall', 'step', 'current']
        self.progress_bars = {bar: self._create_progress_bar(bar, bar in config['bars']) for bar in all_levels}
        
        # Header dengan mode indicator
        self.header = widgets.HTML(f"""<h4 style='color: #333; margin: 0 0 10px 0; font-size: 16px; font-weight: 600;'>
            📊 {config['title']} - {self.operation_name}</h4>""", layout=widgets.Layout(margin='0 0 10px 0'))
        
        # Status message
        self.status_message = widgets.HTML("", layout=widgets.Layout(width='100%', margin='5px 0'))
        
        # Container dengan only visible bars
        visible_bars = [self.progress_bars[bar] for bar in config['bars']]
        self.container = widgets.VBox([self.header, self.status_message] + visible_bars, 
            layout=widgets.Layout(width='100%', visibility='hidden', padding='15px', margin='10px 0',
                                 border='1px solid #28a745', border_radius='8px', background_color='#f8fff8',
                                 min_height=config['height'], max_height='300px'))
    
    def _create_progress_bar(self, level: str, visible: bool = True) -> widgets.HTML:
        """Create individual progress bar dengan level-specific styling"""
        level_configs = {
            'overall': {'icon': '📊', 'name': 'Overall Progress', 'color': '#28a745'},
            'step': {'icon': '🔄', 'name': 'Step Progress', 'color': '#17a2b8'},
            'current': {'icon': '⚡', 'name': 'Current Operation', 'color': '#ffc107'}
        }
        config = level_configs.get(level, {'icon': '📊', 'name': level.title(), 'color': '#007bff'})
        
        # Initialize dengan empty progress bar
        initial_html = f"""
        <div style="margin-bottom: 8px; {'' if visible else 'display: none;'}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 14px; font-weight: 500; color: #333;">{config['icon']} {config['name']}</span>
                <span style="font-size: 12px; color: #666;">0%</span>
            </div>
            <div style="background: #e9ecef; border-radius: 10px; overflow: hidden; height: 16px;">
                <div style="background: {config['color']}; height: 100%; width: 0%; transition: width 0.3s ease; border-radius: 10px;"></div>
            </div>
        </div>"""
        
        return widgets.HTML(initial_html, layout=widgets.Layout(width='100%', margin='2px 0'))
    
    def configure_mode(self, mode: ProgressMode, operation_name: str = None, steps: List[str] = None, 
                      step_weights: Dict[str, int] = None) -> None:
        """Configure tracker mode dengan dynamic UI rebuild"""
        self.mode = mode
        operation_name and setattr(self, 'operation_name', operation_name)
        
        # Setup step configuration untuk DUAL/TRIPLE mode
        if steps and mode in [ProgressMode.DUAL, ProgressMode.TRIPLE]:
            self.total_steps, self.step_weights, self.current_step_index = len(steps), step_weights or self._calculate_equal_weights(steps), 0
        
        # Rebuild UI dengan new mode
        self._create_ui_components()
        self._reset_all_progress()
    
    def show(self, operation_name: str = None, mode: ProgressMode = None, steps: List[str] = None,
             step_weights: Dict[str, int] = None) -> None:
        """Show progress tracker dengan optional reconfiguration"""
        # Reconfigure jika ada parameter baru
        if mode or operation_name or steps:
            self.configure_mode(mode or self.mode, operation_name, steps, step_weights)
        
        self.is_visible = True
        self.container.layout.visibility = 'visible'
        self.container.layout.display = 'flex'
        self._reset_all_progress()
        self._update_status(f"🚀 Memulai {self.operation_name}...", 'info')
    
    def update_progress(self, level: str, value: int, message: str = "", color: str = None) -> None:
        """Update progress level dengan mode-aware validation dan auto-calculation"""
        if not self.is_visible or level not in self.progress_bars:
            return
        
        value = max(0, min(100, value))
        self.progress_values[level] = value
        message and (self.progress_messages.__setitem__(level, message))
        
        # Auto-calculate overall progress untuk DUAL/TRIPLE mode
        if level == 'step' and self.mode in [ProgressMode.DUAL, ProgressMode.TRIPLE]:
            overall_value = self._calculate_weighted_overall_progress(value)
            self._update_progress_bar('overall', overall_value, f"Overall: {overall_value}%", '#28a745')
        
        self._update_progress_bar(level, value, message, color)
        message and self._update_status(message, 'info')
    
    def _calculate_weighted_overall_progress(self, step_progress: int) -> int:
        """Calculate weighted overall progress untuk DUAL/TRIPLE mode dengan one-liner optimization"""
        if not self.step_weights or self.total_steps == 0:
            return step_progress
        
        # One-liner calculation
        completed_weight = sum(weight for i, weight in enumerate(self.step_weights.values()) if i < self.current_step_index)
        current_weight = list(self.step_weights.values())[self.current_step_index] if self.current_step_index < len(self.step_weights) else 0
        total_weight = sum(self.step_weights.values())
        
        return int((completed_weight + (step_progress / 100) * current_weight) / total_weight * 100) if total_weight > 0 else step_progress
    
    def _calculate_equal_weights(self, steps: List[str]) -> Dict[str, int]:
        """Calculate equal weights untuk steps dengan one-liner distribution"""
        base_weight, remainder = divmod(100, len(steps))
        return {step: base_weight + (1 if i < remainder else 0) for i, step in enumerate(steps)}
    
    def advance_step(self, new_step_message: str = "") -> None:
        """Advance ke step berikutnya untuk DUAL/TRIPLE mode"""
        if self.mode not in [ProgressMode.DUAL, ProgressMode.TRIPLE] or self.current_step_index >= self.total_steps - 1:
            return
        
        self.current_step_index += 1
        self.update_progress('step', 0, new_step_message or f"Step {self.current_step_index + 1}/{self.total_steps}")
    
    def complete_operation(self, message: str = "Operation completed successfully!") -> None:
        """Complete operation dengan success styling dan auto-hide"""
        if not self.is_visible:
            return
        
        # Set all visible bars ke 100% dengan one-liner
        [self._update_progress_bar(level, 100, "Selesai", '#28a745') for level in self.progress_bars.keys()]
        [self.progress_values.update({level: 100}) for level in self.progress_values]
        
        self._update_status(f"✅ {message}", 'success')
        self._schedule_auto_hide()
    
    def error_operation(self, message: str = "Operation failed") -> None:
        """Set error state dengan error styling dan cancel auto-hide"""
        if not self.is_visible:
            return
        
        # Set all visible bars ke error color dengan one-liner
        [self._update_progress_bar(level, self.progress_values[level], "Error", '#dc3545') for level in self.progress_bars.keys()]
        
        self._update_status(f"❌ {message}", 'error')
        self._cancel_auto_hide()
    
    def reset_all(self) -> None:
        """Reset semua progress dan state"""
        self._reset_all_progress()
        self.current_step_index = 0
        self.hide()
    
    def hide(self) -> None:
        """Hide progress container dengan cleanup"""
        self.is_visible = False
        self.container.layout.visibility = 'hidden'
        self.container.layout.display = 'none'
        self._cancel_auto_hide()
    
    def _update_progress_bar(self, level: str, value: int, message: str = "", color: str = '#007bff') -> None:
        """Update individual progress bar dengan optimized HTML generation"""
        level_configs = {'overall': ('📊', 'Overall Progress'), 'step': ('🔄', 'Step Progress'), 'current': ('⚡', 'Current Operation')}
        icon, name = level_configs.get(level, ('📊', level.title()))
        display_message = message or self.progress_messages.get(level, '')
        
        # One-liner HTML generation
        self.progress_bars[level].value = f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 14px; font-weight: 500; color: #333;">{icon} {name}</span>
                <span style="font-size: 12px; color: #666;">{value}%</span>
            </div>
            <div style="background: #e9ecef; border-radius: 10px; overflow: hidden; height: 16px;">
                <div style="background: {color}; height: 100%; width: {value}%; transition: width 0.3s ease; border-radius: 10px;"></div>
            </div>
            {f'<div style="font-size: 12px; color: #555; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{display_message}</div>' if display_message else ''}
        </div>"""
    
    def _update_status(self, message: str, style: str = 'info') -> None:
        """Update status message dengan consistent styling"""
        color_map = {'success': '#28a745', 'info': '#007bff', 'warning': '#ffc107', 'error': '#dc3545'}
        color = color_map.get(style, '#495057')
        
        self.status_message.value = f"""
        <div style="color: {color}; font-size: 13px; font-weight: 500; margin: 8px 0; 
                    padding: 8px 12px; background: rgba(233, 236, 239, 0.5); 
                    border-radius: 6px; border-left: 3px solid {color}; word-wrap: break-word;">
            {message}
        </div>"""
    
    def _reset_all_progress(self) -> None:
        """Reset all progress bars dengan one-liner state clearing"""
        [self.progress_values.update({level: 0}) for level in self.progress_values]
        [self.progress_messages.update({level: ""}) for level in self.progress_messages]
        [self._update_progress_bar(level, 0) for level in self.progress_bars.keys()]
        self.status_message.value = ""
    
    def _schedule_auto_hide(self) -> None:
        """Schedule auto-hide dengan threading timer"""
        self._cancel_auto_hide()
        if self.auto_hide_delay > 0:
            self.auto_hide_timer = threading.Timer(self.auto_hide_delay, self.hide)
            self.auto_hide_timer.start()
    
    def _cancel_auto_hide(self) -> None:
        """Cancel pending auto-hide timer"""
        self.auto_hide_timer and self.auto_hide_timer.cancel()
        self.auto_hide_timer = None
    
    # Convenience methods dengan mode-aware delegation
    def update_overall(self, progress: int, message: str = "", color: str = None):
        """Update overall progress - available in all modes"""
        self.update_progress('overall', progress, message, color)
    
    def update_step(self, progress: int, message: str = "", color: str = None):
        """Update step progress - available in DUAL/TRIPLE modes"""
        self.mode in [ProgressMode.DUAL, ProgressMode.TRIPLE] and self.update_progress('step', progress, message, color)
    
    def update_current(self, progress: int, message: str = "", color: str = None):
        """Update current operation progress - available in TRIPLE mode only"""
        self.mode == ProgressMode.TRIPLE and self.update_progress('current', progress, message, color)


# Factory functions dengan enhanced configurability
def create_enhanced_progress_tracker(ui_components: Dict[str, Any], mode: ProgressMode = ProgressMode.SINGLE,
                                   auto_hide_delay: float = 3.0, operation_name: str = "Process") -> Dict[str, Any]:
    """Create enhanced progress tracker dengan full configurability"""
    tracker = EnhancedProgressTracker(ui_components, mode, auto_hide_delay, operation_name)
    
    return {
        'container': tracker.container,
        'progress_container': tracker.container,
        'tracker': tracker,
        'show': tracker.show,
        'show_for_operation': lambda op_name: tracker.show(op_name),
        'hide': tracker.hide,
        'hide_container': tracker.hide,
        'configure_mode': tracker.configure_mode,
        'update_progress': tracker.update_progress,
        'update_overall': tracker.update_overall,
        'update_step': tracker.update_step,
        'update_current': tracker.update_current,
        'advance_step': tracker.advance_step,
        'complete_operation': tracker.complete_operation,
        'error_operation': tracker.error_operation,
        'reset_all': tracker.reset_all
    }

def create_single_progress(ui_components: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Create single progress tracker dengan one-liner"""
    return create_enhanced_progress_tracker(ui_components, ProgressMode.SINGLE, **kwargs)

def create_dual_progress(ui_components: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Create dual progress tracker dengan one-liner"""
    return create_enhanced_progress_tracker(ui_components, ProgressMode.DUAL, **kwargs)

def create_triple_progress(ui_components: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Create triple progress tracker dengan one-liner"""
    return create_enhanced_progress_tracker(ui_components, ProgressMode.TRIPLE, **kwargs)

# Backward compatibility wrapper untuk existing code
def create_simple_progress_tracker(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility wrapper untuk existing code"""
    return create_enhanced_progress_tracker(ui_components, ProgressMode.SINGLE)