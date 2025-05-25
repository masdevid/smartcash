"""
File: smartcash/ui/components/progress_tracking.py
Deskripsi: Enhanced progress tracking dengan flexbox layout, DRY principles, dan consolidated functions
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Set
from tqdm.auto import tqdm
import time
import threading

class ProgressTracker:
    """Enhanced progress tracker dengan flexbox layout dan consolidated methods."""
    
    def __init__(self):
        self.overall_bar = None
        self.step_bar = None
        self.current_bar = None
        self.active_bars: Set[str] = set()
        self.operation_type = None
        
        # Create UI components
        self._create_ui_components()
        
    def _create_ui_components(self):
        """Create UI components dengan flexbox layout."""
        # Header
        self.header_widget = widgets.HTML(
            "<h4 style='color: #333; margin: 0; font-size: 16px; font-weight: 600;'>📊 Progress</h4>",
            layout=widgets.Layout(margin='0 0 10px 0', width='100%', flex='0 0 auto')
        )
        
        # Status widget
        self.status_widget = widgets.HTML(
            value="",
            layout=widgets.Layout(margin='0 0 8px 0', width='100%', flex='0 0 auto')
        )
        
        # Progress bars container
        self.tqdm_container = widgets.Output(
            layout=widgets.Layout(
                margin='0', 
                width='100%', 
                max_width='100%',
                flex='1 1 auto', 
                overflow='hidden'
            )
        )
        
        # Main container
        self.container = widgets.VBox([
            self.header_widget,
            self.status_widget,
            self.tqdm_container
        ], layout=widgets.Layout(
            display='flex',
            flex_flow='column nowrap',
            align_items='stretch',
            margin='10px 0',
            padding='15px',
            border='1px solid #28a745',
            border_radius='8px',
            background_color='#f8fff8',
            width='100%',
            max_width='100%',
            min_height='120px',
            max_height='300px',
            overflow='hidden',
            box_sizing='border-box'
        ))
    
    def show(self, operation: str = None):
        """Show progress container dan initialize bars untuk operation."""
        self.container.layout.display = 'flex'
        self.container.layout.visibility = 'visible'
        
        if operation:
            self.operation_type = operation
            self._initialize_bars_for_operation(operation)
    
    def hide(self):
        """Hide progress container dan cleanup bars."""
        self.container.layout.display = 'none'
        self.container.layout.visibility = 'hidden'
        self._cleanup_bars()
    
    def update(self, progress_type: str, value: int, message: str = "", color: str = None):
        """Update progress bar dengan message dan optional color."""
        if progress_type not in ['overall', 'step', 'current']:
            return
            
        value = max(0, min(100, value))
        bar = getattr(self, f'{progress_type}_bar', None)
        
        if bar is not None:
            self._update_bar(bar, value, progress_type, message, color)
        
        if message:
            # Determine status style based on color or progress
            status_style = self._get_status_style(color, value)
            self._update_status(message, status_style)
    
    def complete(self, message: str = "Selesai"):
        """Complete operation dengan success state."""
        self._set_bars_state(100, '#28a745', '✅', message)
        self._update_status(f"✅ {message}", 'success')
        
        # Auto cleanup after delay
        threading.Thread(
            target=lambda: (time.sleep(3), self._cleanup_bars()),
            daemon=True
        ).start()
    
    def error(self, message: str = "Error"):
        """Set error state untuk all active bars."""
        self._set_bars_state(None, '#dc3545', '❌', message)
        self._update_status(f"❌ {message}", 'error')
    
    def reset(self):
        """Reset progress dan hide container."""
        self._cleanup_bars()
        self.hide()
        self.operation_type = None
    
    def _initialize_bars_for_operation(self, operation: str):
        """Initialize bars based on operation type."""
        self._cleanup_bars()
        
        # Operation configurations
        configs = {
            'check': ['overall'],
            'cleanup': ['overall'],
            'preprocessing': ['overall', 'step'],
            'download': ['overall', 'step', 'current'],
            'augmentation': ['overall', 'step'],
            'training': ['overall', 'step', 'current'],
        }
        
        bars_needed = configs.get(operation, ['overall'])
        optimal_width = self._calculate_optimal_width(len(bars_needed))
        
        with self.tqdm_container:
            bar_configs = [
                ('overall', '📊 Overall Progress', '#28a745', 0),
                ('step', '🔄 Step Progress', '#17a2b8', 1),
                ('current', '⚡ Current Operation', '#ffc107', 2)
            ]
            
            for bar_type, desc, color, position in bar_configs:
                if bar_type in bars_needed:
                    self._create_bar(bar_type, desc, color, position, optimal_width)
    
    def _create_bar(self, bar_type: str, desc: str, color: str, position: int, width: int):
        """Create single progress bar dengan full width."""
        bar = tqdm(
            total=100,
            desc=desc,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}',
            colour=color,
            position=position,
            ncols=width,
            ascii=False,
            mininterval=0.1,
            maxinterval=0.5,
            smoothing=0.3,
            dynamic_ncols=True,  # Enable dynamic width
            leave=True
        )
        
        setattr(self, f'{bar_type}_bar', bar)
        self.active_bars.add(bar_type)
    
    def _update_bar(self, bar: tqdm, value: int, bar_type: str, message: str, color: str = None):
        """Update single bar dengan smooth animation dan optional color."""
        diff = value - bar.n
        if diff > 0:
            bar.update(diff)
        elif diff < 0:
            bar.reset(total=100)
            bar.update(value)
        
        # Update color if provided
        if color:
            bar.colour = self._normalize_color(color)
            bar.refresh()
        
        if message:
            emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
            emoji = emoji_map.get(bar_type, '📊')
            truncated_msg = self._truncate_message(message, 30)
            bar.set_description(f"{emoji} {truncated_msg}")
    
    def _set_bars_state(self, progress: Optional[int], color: str, prefix: str, message: str):
        """Set state untuk all active bars."""
        emoji_map = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
        
        for bar_type in self.active_bars:
            bar = getattr(self, f'{bar_type}_bar', None)
            if bar:
                if progress is not None:
                    bar.n = progress
                bar.colour = color
                bar.refresh()
                
                emoji = emoji_map.get(bar_type, '📊')
                truncated_msg = self._truncate_message(message, 25)
                bar.set_description(f"{prefix} {emoji} {truncated_msg}")
    
    def _cleanup_bars(self):
        """Cleanup all progress bars."""
        for bar_type in ['overall', 'step', 'current']:
            bar = getattr(self, f'{bar_type}_bar', None)
            if bar:
                try:
                    bar.close()
                except Exception:
                    pass
                setattr(self, f'{bar_type}_bar', None)
        
        self.active_bars.clear()
        self.tqdm_container.clear_output(wait=True)
    
    def _update_status(self, message: str, style: str = None):
        """Update status message dengan styling."""
        color_map = {
            'success': '#28a745',
            'info': '#007bff',
            'warning': '#ffc107',
            'error': '#dc3545'
        }
        
        color = color_map.get(style, '#495057')
        
        html_content = f"""
        <div style="
            color: {color}; 
            font-size: 13px; 
            font-weight: 500; 
            margin: 0; 
            padding: 8px 12px; 
            background: rgba(233, 236, 239, 0.5); 
            border-radius: 6px; 
            border-left: 3px solid {color};
            width: 100%; 
            box-sizing: border-box;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.4;
            display: flex;
            align-items: center;
        ">
            {message}
        </div>
        """
        
        self.status_widget.value = html_content
        self.status_widget.layout.visibility = 'visible'
    
    def _get_status_style(self, color: str, progress: int) -> str:
        """Determine status style based on color or progress."""
        if color:
            color_style_map = {
                'success': 'success',
                '#28a745': 'success',
                'info': 'info',
                '#007bff': 'info',
                'warning': 'warning',
                '#ffc107': 'warning',
                'error': 'error',
                '#dc3545': 'error'
            }
            return color_style_map.get(color, 'info')
        
        # Default based on progress
        if progress >= 100:
            return 'success'
        elif progress > 0:
            return 'info'
        else:
            return None
    
    def _normalize_color(self, color: str) -> str:
        """Normalize color string to hex format."""
        color_map = {
            'success': '#28a745',
            'info': '#007bff',
            'warning': '#ffc107',
            'error': '#dc3545'
        }
        return color_map.get(color, color)
    
    @staticmethod
    def _calculate_optimal_width(num_bars: int) -> int:
        """Calculate optimal width untuk progress bars - use larger values for full width."""
        # Increased base width for better container utilization
        base_width = 100
        
        if num_bars == 1:
            return base_width + 20  # Single bar gets more space
        elif num_bars == 2:
            return base_width + 10
        else:
            return base_width  # 3+ bars use base width
    
    @staticmethod
    def _truncate_message(message: str, max_length: int) -> str:
        """Truncate message to prevent overflow."""
        if len(message) <= max_length:
            return message
        return message[:max_length-3] + "..."

def create_progress_tracking_container() -> Dict[str, Any]:
    """Factory function untuk create progress tracker."""
    tracker = ProgressTracker()
    
    return {
        'container': tracker.container,
        'progress_container': tracker.container,
        'status_widget': tracker.status_widget,
        'tqdm_container': tracker.tqdm_container,
        'tracker': tracker,
        # Main API methods
        'show_container': tracker.show,
        'hide_container': tracker.hide,
        'show_for_operation': tracker.show,
        'update_progress': tracker.update,
        'complete_operation': tracker.complete,
        'error_operation': tracker.error,
        'reset_all': tracker.reset
    }