"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Enhanced progress tracking component dengan one-liner style dan consolidated methods
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Set
from tqdm.auto import tqdm
import time
import threading

class ProgressTracker:
    """Enhanced progress tracker dengan one-liner style dan consolidated methods"""
    
    def __init__(self):
        self.overall_bar = self.step_bar = self.current_bar = None
        self.active_bars: Set[str] = set()
        self.operation_type = None
        self._create_ui_components()
        
    def _create_ui_components(self):
        """Create UI components dengan one-liner flexbox layout"""
        self.header_widget = widgets.HTML("<h4 style='color: #333; margin: 0; font-size: 16px; font-weight: 600;'>📊 Progress</h4>",
                                         layout=widgets.Layout(margin='0 0 10px 0', width='100%', flex='0 0 auto'))
        self.status_widget = widgets.HTML("", layout=widgets.Layout(margin='0 0 8px 0', width='100%', flex='0 0 auto'))
        self.tqdm_container = widgets.Output(layout=widgets.Layout(margin='0', width='100%', max_width='100%', flex='1 1 auto', overflow='hidden'))
        self.container = widgets.VBox([self.header_widget, self.status_widget, self.tqdm_container],
                                     layout=widgets.Layout(display='flex', flex_flow='column nowrap', align_items='stretch', margin='10px 0',
                                                          padding='15px', border='1px solid #28a745', border_radius='8px', background_color='#f8fff8',
                                                          width='100%', max_width='100%', min_height='120px', max_height='300px', overflow='hidden', box_sizing='border-box'))
    
    def show(self, operation: str = None):
        """Show progress container dengan one-liner initialization"""
        setattr(self.container.layout, 'display', 'flex'), setattr(self.container.layout, 'visibility', 'visible')
        operation and (setattr(self, 'operation_type', operation), self._initialize_bars_for_operation(operation))
    
    def hide(self):
        """Hide progress container dengan one-liner cleanup"""
        setattr(self.container.layout, 'display', 'none'), setattr(self.container.layout, 'visibility', 'hidden'), self._cleanup_bars()
    
    def update(self, progress_type: str, value: int, message: str = "", color: str = None):
        """Update progress bar dengan one-liner message dan optional color"""
        progress_type not in ['overall', 'step', 'current'] and None or (
            setattr(self, 'value', max(0, min(100, value))),
            bar := getattr(self, f'{progress_type}_bar', None),
            bar and self._update_bar(bar, value, progress_type, message, color),
            message and self._update_status(message, self._get_status_style(color, value))
        )
    
    def complete(self, message: str = "Selesai"):
        """Complete operation dengan one-liner success state"""
        self._set_bars_state(100, '#28a745', '✅', message), self._update_status(f"✅ {message}", 'success')
        threading.Thread(target=lambda: (time.sleep(3), self._cleanup_bars()), daemon=True).start()
    
    def error(self, message: str = "Error"):
        """Set error state dengan one-liner untuk all active bars"""
        self._set_bars_state(None, '#dc3545', '❌', message), self._update_status(f"❌ {message}", 'error')
    
    def reset(self):
        """Reset progress dengan one-liner"""
        self._cleanup_bars(), self.hide(), setattr(self, 'operation_type', None)
    
    def _initialize_bars_for_operation(self, operation: str):
        """Initialize bars dengan one-liner configuration"""
        self._cleanup_bars()
        configs = {'check': ['overall'], 'cleanup': ['overall'], 'preprocessing': ['overall', 'step'], 
                  'download': ['overall', 'step', 'current'], 'augmentation': ['overall', 'step'], 'training': ['overall', 'step', 'current']}
        bars_needed = configs.get(operation, ['overall'])
        optimal_width = self._calculate_optimal_width(len(bars_needed))
        
        with self.tqdm_container:
            bar_configs = [('overall', '📊 Overall Progress', '#28a745', 0), ('step', '🔄 Step Progress', '#17a2b8', 1), ('current', '⚡ Current Operation', '#ffc107', 2)]
            [self._create_bar(bar_type, desc, color, position, optimal_width) for bar_type, desc, color, position in bar_configs if bar_type in bars_needed]
    
    def _create_bar(self, bar_type: str, desc: str, color: str, position: int, width: int):
        """Create single progress bar dengan one-liner full width"""
        bar = tqdm(total=100, desc=desc, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}', colour=color, position=position,
                  ncols=width, ascii=False, mininterval=0.1, maxinterval=0.5, smoothing=0.3, dynamic_ncols=True, leave=True)
        setattr(self, f'{bar_type}_bar', bar), self.active_bars.add(bar_type)
    
    def _update_bar(self, bar: tqdm, value: int, bar_type: str, message: str, color: str = None):
        """Update single bar dengan one-liner smooth animation"""
        diff = value - bar.n
        (diff > 0 and bar.update(diff)) or (diff < 0 and (bar.reset(total=100), bar.update(value)))
        color and (setattr(bar, 'colour', self._normalize_color(color)), bar.refresh())
        message and bar.set_description(f"{'📊' if bar_type == 'overall' else '🔄' if bar_type == 'step' else '⚡'} {self._truncate_message(message, 30)}")
    
    def _set_bars_state(self, progress: Optional[int], color: str, prefix: str, message: str):
        """Set state untuk all active bars dengan one-liner"""
        emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
        [bar and (progress is not None and setattr(bar, 'n', progress), setattr(bar, 'colour', color), bar.refresh(),
                 bar.set_description(f"{prefix} {emoji_map.get(bar_type, '📊')} {self._truncate_message(message, 25)}"))
         for bar_type in self.active_bars if (bar := getattr(self, f'{bar_type}_bar', None))]
    
    def _cleanup_bars(self):
        """Cleanup all progress bars dengan one-liner"""
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

def create_progress_tracking_container() -> Dict[str, Any]:
    """Factory function untuk create progress tracker dengan one-liner"""
    tracker = ProgressTracker()
    return {'container': tracker.container, 'progress_container': tracker.container, 'status_widget': tracker.status_widget,
            'tqdm_container': tracker.tqdm_container, 'tracker': tracker, 'show_container': tracker.show,
            'hide_container': tracker.hide, 'show_for_operation': tracker.show, 'update_progress': tracker.update,
            'complete_operation': tracker.complete, 'error_operation': tracker.error, 'reset_all': tracker.reset}

def create_progress_tracking() -> Dict[str, Any]:
    """Alias untuk create_progress_tracking_container dengan one-liner kompatibilitas"""
    return create_progress_tracking_container()