"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Enhanced progress tracking dengan three-level progress system (Overall, Step, Current)
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Set
from tqdm.auto import tqdm
import time
import threading

class ProgressTracker:
    """Three-level progress tracker dengan overall/step/current progress system dan one-liner operations"""
    
    def __init__(self):
        self.overall_bar = self.step_bar = self.current_bar = None
        self.active_bars: Set[str] = set()
        self.operation_type = None
        self.operation_steps = []
        self.current_step_index = 0
        self.step_weights = {}
        self._create_ui_components()
        
    def _create_ui_components(self):
        """Create UI components dengan enhanced three-level layout - one-liner style"""
        self.header_widget = widgets.HTML("<h4 style='color: #333; margin: 0; font-size: 16px; font-weight: 600;'>📊 Progress Tracking</h4>",
                                         layout=widgets.Layout(margin='0 0 10px 0', width='100%', flex='0 0 auto'))
        self.status_widget = widgets.HTML("", layout=widgets.Layout(margin='0 0 8px 0', width='100%', flex='0 0 auto'))
        self.step_info_widget = widgets.HTML("", layout=widgets.Layout(margin='0 0 5px 0', width='100%', flex='0 0 auto'))
        self.tqdm_container = widgets.Output(layout=widgets.Layout(margin='0', width='100%', max_width='100%', flex='1 1 auto', overflow='hidden'))
        self.container = widgets.VBox([self.header_widget, self.status_widget, self.step_info_widget, self.tqdm_container],
                                     layout=widgets.Layout(display='flex', flex_flow='column nowrap', align_items='stretch', margin='10px 0',
                                                          padding='15px', border='1px solid #28a745', border_radius='8px', background_color='#f8fff8',
                                                          width='100%', max_width='100%', min_height='150px', max_height='350px', overflow='hidden', box_sizing='border-box'))
    
    def show(self, operation: str = None, steps: list = None, step_weights: Dict[str, int] = None):
        """Show progress container dengan three-level initialization"""
        setattr(self.container.layout, 'display', 'flex'), setattr(self.container.layout, 'visibility', 'visible')
        
        if operation:
            self.operation_type = operation
            self.operation_steps = steps or self._get_default_steps(operation)
            self.step_weights = step_weights or self._get_default_weights(operation)
            self.current_step_index = 0
            self._initialize_three_level_bars(operation)
            self._update_step_info()
    
    def _get_default_steps(self, operation: str) -> list:
        """Get default steps untuk different operations dengan one-liner mapping"""
        operation_steps = {
            'download': ['validate', 'connect', 'metadata', 'download', 'extract', 'organize'],
            'check': ['validate', 'connect', 'metadata', 'local_check', 'report'],
            'cleanup': ['scan', 'confirm', 'cleanup', 'verify'],
            'preprocessing': ['validate', 'process', 'save'],
            'augmentation': ['validate', 'augment', 'save']
        }
        return operation_steps.get(operation, ['validate', 'process', 'complete'])
    
    def _get_default_weights(self, operation: str) -> Dict[str, int]:
        """Get default step weights untuk progress calculation dengan one-liner mapping"""
        weight_maps = {
            'download': {'validate': 5, 'connect': 10, 'metadata': 10, 'download': 50, 'extract': 15, 'organize': 10},
            'check': {'validate': 10, 'connect': 20, 'metadata': 30, 'local_check': 30, 'report': 10},
            'cleanup': {'scan': 20, 'confirm': 5, 'cleanup': 70, 'verify': 5},
            'preprocessing': {'validate': 20, 'process': 60, 'save': 20},
            'augmentation': {'validate': 10, 'augment': 80, 'save': 10}
        }
        return weight_maps.get(operation, {step: 100//len(self.operation_steps) for step in self.operation_steps})
    
    def _initialize_three_level_bars(self, operation: str):
        """Initialize three-level progress bars dengan proper configuration"""
        self._cleanup_bars()
        optimal_width = self._calculate_optimal_width(3)  # Always 3 bars
        
        with self.tqdm_container:
            # Three-level bar configuration dengan descriptive names
            bar_configs = [
                ('overall', f'📊 Overall Progress ({operation.title()})', '#28a745', 0),
                ('step', f'🔄 Step Progress', '#17a2b8', 1),
                ('current', f'⚡ Current Operation', '#ffc107', 2)
            ]
            
            [self._create_progress_bar(bar_type, desc, color, position, optimal_width) 
             for bar_type, desc, color, position in bar_configs]
    
    def _create_progress_bar(self, bar_type: str, desc: str, color: str, position: int, width: int):
        """Create progress bar dengan detailed formatting - one-liner creation"""
        bar = tqdm(total=100, desc=desc, 
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]', 
                  colour=color, position=position, ncols=width, ascii=False, 
                  mininterval=0.1, maxinterval=0.5, smoothing=0.3, dynamic_ncols=True, leave=True)
        setattr(self, f'{bar_type}_bar', bar), self.active_bars.add(bar_type)
    
    def update_overall(self, progress: int, message: str = ""):
        """Update overall progress dengan step-based calculation"""
        # Calculate overall progress berdasarkan current step dan step weights
        if self.operation_steps and self.step_weights:
            completed_weight = sum(self.step_weights.get(step, 0) 
                                 for step in self.operation_steps[:self.current_step_index])
            current_step_weight = self.step_weights.get(self.operation_steps[self.current_step_index] if self.current_step_index < len(self.operation_steps) else '', 0)
            current_contribution = (progress / 100) * current_step_weight
            total_weight = sum(self.step_weights.values())
            
            overall_progress = int((completed_weight + current_contribution) / total_weight * 100) if total_weight > 0 else progress
        else:
            overall_progress = progress
        
        self.update('overall', overall_progress, message)
        self._update_step_info()
    
    def update_step(self, progress: int, message: str = ""):
        """Update step progress untuk current step"""
        self.update('step', progress, message)
        # Auto-advance ke step berikutnya jika completed
        progress >= 100 and self._advance_to_next_step()
    
    def update_current(self, progress: int, message: str = ""):
        """Update current operation progress"""
        self.update('current', progress, message)
    
    def update(self, progress_type: str, value: int, message: str = "", color: str = None):
        """Update specific progress bar dengan message dan color - one-liner validation"""
        progress_type not in ['overall', 'step', 'current'] and None or (
            setattr(self, '_temp_value', max(0, min(100, value))),
            bar := getattr(self, f'{progress_type}_bar', None),
            bar and self._update_progress_bar(bar, value, progress_type, message, color),
            message and self._update_status(message, self._get_status_style(color, value))
        )
    
    def _advance_to_next_step(self):
        """Advance ke step berikutnya dengan automatic progression"""
        if self.current_step_index < len(self.operation_steps) - 1:
            self.current_step_index += 1
            self._update_step_info()
            # Reset step progress untuk step baru
            self.step_bar and self._update_progress_bar(self.step_bar, 0, 'step', f"Starting {self.operation_steps[self.current_step_index]}")
    
    def _update_step_info(self):
        """Update step information display dengan one-liner HTML generation"""
        if self.operation_steps and self.current_step_index < len(self.operation_steps):
            current_step = self.operation_steps[self.current_step_index]
            step_info = f"""
            <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin: 2px 0;">
                <small style="color: #1976d2;">
                    <strong>Step {self.current_step_index + 1}/{len(self.operation_steps)}:</strong> 
                    {current_step.title()} 
                    <span style="color: #666;">(Weight: {self.step_weights.get(current_step, 0)}%)</span>
                </small>
            </div>
            """
            setattr(self.step_info_widget, 'value', step_info)
    
    def complete(self, message: str = "Selesai"):
        """Complete operation dengan three-level completion - one-liner state setting"""
        self._set_bars_state(100, '#28a745', '✅', message), self._update_status(f"✅ {message}", 'success')
        self.current_step_index = len(self.operation_steps)  # Mark all steps completed
        self._update_step_info()
        threading.Thread(target=lambda: (time.sleep(3), self._cleanup_bars()), daemon=True).start()
    
    def error(self, message: str = "Error"):
        """Set error state dengan proper error display - one-liner error state"""
        self._set_bars_state(None, '#dc3545', '❌', message), self._update_status(f"❌ {message}", 'error')
    
    def reset(self):
        """Reset progress dengan complete cleanup - one-liner reset"""
        self._cleanup_bars(), self.hide(), setattr(self, 'operation_type', None)
        self.operation_steps = []
        self.current_step_index = 0
        self.step_weights = {}
    
    def hide(self):
        """Hide progress container dengan one-liner cleanup"""
        setattr(self.container.layout, 'display', 'none'), setattr(self.container.layout, 'visibility', 'hidden'), self._cleanup_bars()
    
    def _update_progress_bar(self, bar: tqdm, value: int, bar_type: str, message: str, color: str = None):
        """Update single bar dengan smooth animation - one-liner updates"""
        diff = value - bar.n
        (diff > 0 and bar.update(diff)) or (diff < 0 and (bar.reset(total=100), bar.update(value)))
        color and (setattr(bar, 'colour', self._normalize_color(color)), bar.refresh())
        message and bar.set_description(f"{'📊' if bar_type == 'overall' else '🔄' if bar_type == 'step' else '⚡'} {self._truncate_message(message, 35)}")
    
    def _set_bars_state(self, progress: Optional[int], color: str, prefix: str, message: str):
        """Set state untuk all active bars dengan one-liner updates"""
        emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
        [bar and (progress is not None and setattr(bar, 'n', progress), setattr(bar, 'colour', color), bar.refresh(),
                 bar.set_description(f"{prefix} {emoji_map.get(bar_type, '📊')} {self._truncate_message(message, 30)}"))
         for bar_type in self.active_bars if (bar := getattr(self, f'{bar_type}_bar', None))]
    
    def _cleanup_bars(self):
        """Cleanup all progress bars dengan one-liner removal"""
        [bar and (bar.close() if True else None, setattr(self, f'{bar_type}_bar', None)) 
         for bar_type in ['overall', 'step', 'current'] if (bar := getattr(self, f'{bar_type}_bar', None))]
        self.active_bars.clear(), self.tqdm_container.clear_output(wait=True)
    
    def _update_status(self, message: str, style: str = None):
        """Update status message dengan one-liner styling"""
        color_map = {'success': '#28a745', 'info': '#007bff', 'warning': '#ffc107', 'error': '#dc3545'}
        color = color_map.get(style, '#495057')
        setattr(self.status_widget, 'value', f"""<div style="color: {color}; font-size: 13px; font-weight: 500; margin: 0; padding: 8px 12px; background: rgba(233, 236, 239, 0.5); border-radius: 6px; border-left: 3px solid {color}; width: 100%; box-sizing: border-box; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.4; display: flex; align-items: center;">{message}</div>""")
        setattr(self.status_widget.layout, 'visibility', 'visible')
    
    def _get_status_style(self, color: str, progress: int) -> str:
        """Determine status style dengan one-liner mapping"""
        return ({'success': 'success', '#28a745': 'success', 'info': 'info', '#007bff': 'info', 'warning': 'warning', '#ffc107': 'warning', 'error': 'error', '#dc3545': 'error'}.get(color) 
                if color else ('success' if progress >= 100 else ('info' if progress > 0 else None)))
    
    def _normalize_color(self, color: str) -> str:
        """Normalize color dengan one-liner mapping"""
        return {'success': '#28a745', 'info': '#007bff', 'warning': '#ffc107', 'error': '#dc3545'}.get(color, color)
    
    @staticmethod
    def _calculate_optimal_width(num_bars: int) -> int:
        """Calculate optimal width dengan one-liner logic"""
        return 100 + (20 if num_bars == 1 else (10 if num_bars == 2 else 0))
    
    @staticmethod
    def _truncate_message(message: str, max_length: int) -> str:
        """Truncate message dengan one-liner"""
        return message if len(message) <= max_length else f"{message[:max_length-3]}..."

def create_three_level_progress_tracking() -> Dict[str, Any]:
    """Factory function untuk create three-level progress tracker"""
    tracker = ThreeLevelProgressTracker()
    return {'container': tracker.container, 'progress_container': tracker.container, 'status_widget': tracker.status_widget,
            'step_info_widget': tracker.step_info_widget, 'tqdm_container': tracker.tqdm_container, 'tracker': tracker, 
            'show_container': tracker.show, 'hide_container': tracker.hide, 'show_for_operation': tracker.show, 
            'update_overall': tracker.update_overall, 'update_step': tracker.update_step, 'update_current': tracker.update_current,
            'update_progress': tracker.update, 'complete_operation': tracker.complete, 'error_operation': tracker.error, 'reset_all': tracker.reset}

# Backward compatibility
def create_progress_tracking_container() -> Dict[str, Any]:
    """Backward compatibility alias untuk existing code"""
    return create_three_level_progress_tracking()

def create_progress_tracking() -> Dict[str, Any]:
    """Backward compatibility alias untuk existing code"""
    return create_three_level_progress_tracking()