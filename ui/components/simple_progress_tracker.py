"""
File: smartcash/ui/components/simple_progress_tracker.py
Deskripsi: Simplified progress tracker tanpa tqdm untuk menghindari weak reference error
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable
import threading
import time

class SimpleProgressTracker:
    """Simplified progress tracker tanpa tqdm dependencies untuk avoid weak reference error"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.is_visible = False
        self.current_operation = None
        self._create_ui_components()
    
    def _create_ui_components(self):
        """Create simple progress UI components"""
        # Progress bars menggunakan HTML dan CSS
        self.overall_bar = widgets.HTML(value="", layout=widgets.Layout(width='100%', margin='2px 0'))
        self.step_bar = widgets.HTML(value="", layout=widgets.Layout(width='100%', margin='2px 0'))
        self.current_bar = widgets.HTML(value="", layout=widgets.Layout(width='100%', margin='2px 0'))
        
        # Status message
        self.status_message = widgets.HTML(value="", layout=widgets.Layout(width='100%', margin='5px 0'))
        
        # Container
        self.container = widgets.VBox([
            self.overall_bar, self.step_bar, self.current_bar, self.status_message
        ], layout=widgets.Layout(
            width='100%', visibility='hidden', padding='15px', margin='10px 0',
            border='1px solid #28a745', border_radius='8px', background_color='#f8fff8'
        ))
        
        # Progress state
        self.progress_values = {'overall': 0, 'step': 0, 'current': 0}
        self.progress_messages = {'overall': '', 'step': '', 'current': ''}
    
    def show_for_operation(self, operation_name: str) -> None:
        """Show progress tracker untuk operation"""
        self.current_operation = operation_name
        self.is_visible = True
        self.container.layout.visibility = 'visible'
        self._reset_all_progress()
        self._update_status(f"🚀 Memulai {operation_name}...", 'info')
    
    def update_progress(self, level: str, value: int, message: str = "") -> None:
        """Update progress level dengan message"""
        if not self.is_visible or level not in self.progress_values:
            return
        
        # Normalize value
        value = max(0, min(100, value))
        self.progress_values[level] = value
        
        if message:
            self.progress_messages[level] = message
        
        self._update_progress_bar(level, value, message)
        
        # Update status untuk significant progress
        if value > 0 and message:
            self._update_status(message, 'info')
    
    def complete_operation(self, message: str = "Operation completed successfully!") -> None:
        """Complete operation dengan success styling"""
        if not self.is_visible:
            return
        
        # Set all bars ke 100%
        for level in self.progress_values:
            self.progress_values[level] = 100
            self._update_progress_bar(level, 100, "Selesai", color='#28a745')
        
        self._update_status(f"✅ {message}", 'success')
        
        # Auto hide setelah 3 detik
        threading.Timer(3.0, self.hide).start()
    
    def error_operation(self, message: str = "Operation failed") -> None:
        """Set error state dengan error styling"""
        if not self.is_visible:
            return
        
        # Set all bars ke error color
        for level in self.progress_values:
            self._update_progress_bar(level, self.progress_values[level], "Error", color='#dc3545')
        
        self._update_status(f"❌ {message}", 'error')
    
    def reset_all(self) -> None:
        """Reset semua progress dan hide"""
        self._reset_all_progress()
        self.hide()
    
    def hide(self) -> None:
        """Hide progress container"""
        self.is_visible = False
        self.current_operation = None
        self.container.layout.visibility = 'hidden'
    
    def _update_progress_bar(self, level: str, value: int, message: str = "", color: str = '#007bff') -> None:
        """Update individual progress bar dengan HTML/CSS"""
        level_icons = {'overall': '📊', 'step': '🔄', 'current': '⚡'}
        level_names = {'overall': 'Overall Progress', 'step': 'Step Progress', 'current': 'Current Operation'}
        
        icon = level_icons.get(level, '📊')
        name = level_names.get(level, level.title())
        display_message = message or self.progress_messages.get(level, '')
        
        bar_html = f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 14px; font-weight: 500; color: #333;">
                    {icon} {name}
                </span>
                <span style="font-size: 12px; color: #666;">{value}%</span>
            </div>
            <div style="background: #e9ecef; border-radius: 10px; overflow: hidden; height: 16px;">
                <div style="background: {color}; height: 100%; width: {value}%; 
                           transition: width 0.3s ease; border-radius: 10px;"></div>
            </div>
            {f'<div style="font-size: 12px; color: #555; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{display_message}</div>' if display_message else ''}
        </div>
        """
        
        if level == 'overall':
            self.overall_bar.value = bar_html
        elif level == 'step':
            self.step_bar.value = bar_html
        elif level == 'current':
            self.current_bar.value = bar_html
    
    def _update_status(self, message: str, style: str = 'info') -> None:
        """Update status message dengan styling"""
        color_map = {
            'success': '#28a745', 'info': '#007bff', 
            'warning': '#ffc107', 'error': '#dc3545'
        }
        color = color_map.get(style, '#495057')
        
        self.status_message.value = f"""
        <div style="color: {color}; font-size: 13px; font-weight: 500; margin: 8px 0; 
                    padding: 8px 12px; background: rgba(233, 236, 239, 0.5); 
                    border-radius: 6px; border-left: 3px solid {color}; 
                    word-wrap: break-word;">
            {message}
        </div>
        """
    
    def _reset_all_progress(self) -> None:
        """Reset all progress bars ke 0"""
        for level in self.progress_values:
            self.progress_values[level] = 0
            self.progress_messages[level] = ""
            self._update_progress_bar(level, 0)
        self.status_message.value = ""

# Factory function untuk backward compatibility
def create_simple_progress_tracker(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create simple progress tracker dengan interface compatibility"""
    tracker = SimpleProgressTracker(ui_components)
    
    return {
        'container': tracker.container,
        'progress_container': tracker.container,
        'tracker': tracker,
        'show_for_operation': tracker.show_for_operation,
        'update_progress': tracker.update_progress,
        'complete_operation': tracker.complete_operation,
        'error_operation': tracker.error_operation,
        'reset_all': tracker.reset_all,
        'hide_container': tracker.hide,
        # Compatibility methods untuk existing code
        'update_overall': lambda progress, message="", color=None: tracker.update_progress('overall', progress, message),
        'update_step': lambda progress, message="", color=None: tracker.update_progress('step', progress, message),
        'update_current': lambda progress, message="", color=None: tracker.update_progress('current', progress, message)
    }