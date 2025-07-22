"""
File: smartcash/ui/components/progress_tracker/ui_components.py
Deskripsi: Manager UI components tanpa step info dan auto hide support
"""

import threading
import time
import ipywidgets as widgets
from typing import Optional, Dict, Any

from .types import ProgressConfig

class UIComponentsManager:
    """Manager untuk UI components tanpa step info dengan auto hide support"""
    
    def __init__(self, config: ProgressConfig):
        self.config = config
        self.header_widget = None
        self.status_widget = None
        self.progress_output = None
        self.container = None
        self.is_visible = False
        self._auto_hide_timer = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create UI widgets dengan layout vertikal untuk setiap progress bar"""
        # Header dengan modern typography
        self.header_widget = widgets.HTML("", layout=widgets.Layout(
            width='100%', 
            margin='0 0 8px 0',
            padding='0'
        ))
        
        # Status message dengan modern design
        self.status_widget = widgets.HTML("", layout=widgets.Layout(
            width='100%', 
            margin='0 0 8px 0',
            padding='0'
        ))
        
        # Progress bars dengan compact styling
        progress_style = {
            'width': '100%',
            'margin': '4px 0',
            'min_height': '24px',
            'max_height': '24px',
            'padding': '0',
            'overflow': 'visible'
        }
        
        self.overall_output = widgets.Output(layout=widgets.Layout(**progress_style))
        self.step_output = widgets.Output(layout=widgets.Layout(**progress_style))
        self.current_output = widgets.Output(layout=widgets.Layout(**progress_style))
        
        self._create_container()
    
    def _create_container(self):
        """Create main container dengan separate progress outputs"""
        progress_widgets = []
        
        # Add progress outputs berdasarkan level
        if self.config.level.value >= 2:  # DUAL atau TRIPLE
            progress_widgets.append(self.overall_output)
        
        if self.config.level.value >= 3:  # TRIPLE
            progress_widgets.append(self.step_output)
            progress_widgets.append(self.current_output)
        elif self.config.level.value == 2:  # DUAL
            progress_widgets.append(self.current_output)
        elif self.config.level.value == 1:  # SINGLE
            progress_widgets.append(self.overall_output)  # Use overall for single
        
        # Modern container with glassmorphism effect (no padding)
        self.container = widgets.VBox(
            [self.header_widget, self.status_widget] + progress_widgets,
            layout=widgets.Layout(
                display='none',
                width='100%',
                margin='4px 0',
                padding='0px',
                border='1px solid rgba(255, 255, 255, 0.18)',
                border_radius='12px',
                background_color='rgba(255, 255, 255, 0.95)',
                box_shadow='0 4px 16px rgba(0, 0, 0, 0.08), 0 1px 4px rgba(0, 0, 0, 0.04)',
                backdrop_filter='blur(10px)',
                min_height=self.config.get_container_height(),
                max_height='none',
                overflow='visible',
                box_sizing='border-box'
            )
        )
    
    def show(self):
        """Show container dengan auto hide timer jika enabled"""
        self.container.layout.display = 'flex'
        self.container.layout.visibility = 'visible'
        self.is_visible = True
        
        # Start auto hide timer jika enabled
        if self.config.auto_hide and self.config.auto_hide_delay > 0:
            self._start_auto_hide_timer()
    
    def hide(self):
        """Hide container dan cancel auto hide timer"""
        self.container.layout.display = 'none'
        self.container.layout.visibility = 'hidden'
        self.is_visible = False
        self._cancel_auto_hide_timer()
    
    def _start_auto_hide_timer(self):
        """Start auto hide timer untuk 1 jam"""
        self._cancel_auto_hide_timer()  # Cancel existing timer
        
        def auto_hide_task():
            time.sleep(self.config.auto_hide_delay)
            if self.is_visible:  # Check masih visible
                self.hide()
        
        self._auto_hide_timer = threading.Thread(target=auto_hide_task, daemon=True)
        self._auto_hide_timer.start()
    
    def _cancel_auto_hide_timer(self):
        """Cancel auto hide timer jika ada"""
        if self._auto_hide_timer and self._auto_hide_timer.is_alive():
            # Thread akan terminate karena daemon=True
            self._auto_hide_timer = None
    
    def update_header(self, operation: str):
        """Update header dengan operation name"""
        # Modern header with better visual hierarchy
        self.header_widget.value = f"""
        <div style='
            display: flex;
            align-items: center;
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        '>
            <div style='
                width: 4px;
                height: 32px;
                background: linear-gradient(180deg, #007bff 0%, #28a745 100%);
                border-radius: 2px;
                margin-right: 16px;
            '></div>
            <h3 style='
                font-size: 18px;
                font-weight: 700;
                margin: 0;
                padding: 0;
                line-height: 1.3;
                color: #1a1a1a;
                letter-spacing: -0.01em;
            '>
                {operation}
            </h3>
        </div>
        """
    
    def update_status(self, message: str, style: str = None):
        """Update status message dengan modern design dan adaptive logging level"""
        color_map = {
            'success': '#16a34a', 'info': '#0ea5e9', 
            'warning': '#f59e0b', 'error': '#ef4444'
        }
        color = color_map.get(style, '#64748b')
        
        # Clean message dari emoji duplikat jika ada
        clean_message = self._clean_status_message(message)
        
        # Adaptive background based on logging level
        bg_colors = {
            'success': 'linear-gradient(135deg, rgba(22, 163, 74, 0.08) 0%, rgba(22, 163, 74, 0.04) 100%)',
            'info': 'linear-gradient(135deg, rgba(14, 165, 233, 0.08) 0%, rgba(14, 165, 233, 0.04) 100%)',
            'warning': 'linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.04) 100%)',
            'error': 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.04) 100%)'
        }
        bg_color = bg_colors.get(style, 'linear-gradient(135deg, rgba(100, 116, 139, 0.05) 0%, rgba(100, 116, 139, 0.02) 100%)')
        
        # Status icon based on style
        icons = {
            'success': '✅',
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌'
        }
        icon = icons.get(style, '📋')
        
        self.status_widget.value = f"""
        <div style="
            display: flex;
            align-items: center;
            width: 100%;
            color: {color};
            font-size: 14px;
            font-weight: 600;
            margin: 0;
            padding: 12px 16px;
            background: {bg_color};
            border-radius: 8px;
            border: 1px solid rgba({self._hex_to_rgb(color)}, 0.2);
            box-sizing: border-box;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.4;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 8px rgba({self._hex_to_rgb(color)}, 0.1);
        ">
            <span style="
                font-size: 16px;
                margin-right: 12px;
                opacity: 0.9;
            ">{icon}</span>
            <span style="flex: 1;">{clean_message}</span>
        </div>
        """
    
    @staticmethod
    def _clean_status_message(message: str) -> str:
        """Clean status message dari emoji duplikat"""
        import re
        # Remove existing status emojis to prevent duplication
        cleaned = re.sub(r'^[✅❌⚠️ℹ️🚀📋]\s*', '', message)
        return cleaned.strip()
        
    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        """Convert hex color to RGB values for CSS"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"
        return "100, 116, 139"  # Default gray