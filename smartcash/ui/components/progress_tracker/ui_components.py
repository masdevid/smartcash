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
        # Header dengan padding dan font yang lebih baik
        self.header_widget = widgets.HTML("", layout=widgets.Layout(
            width='100%', 
            margin='0 0 12px 0',
            padding='0 4px'
        ))
        
        # Status message dengan spacing yang lebih baik
        self.status_widget = widgets.HTML("", layout=widgets.Layout(
            width='100%', 
            margin='0 0 16px 0',
            padding='0 4px'
        ))
        
        # Progress bars dengan style yang konsisten
        progress_style = {
            'width': '100%',
            'margin': '12px 0',
            'min_height': '24px',
            'max_height': '28px',
            'padding': '0 4px'
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
        
        # Container dengan shadow dan border yang lebih halus
        self.container = widgets.VBox(
            [self.header_widget, self.status_widget] + progress_widgets,
            layout=widgets.Layout(
                display='none',
                width='100%',
                margin='12px 0',
                padding='20px',
                border='1px solid #e0e0e0',
                border_radius='10px',
                background_color='#ffffff',
                box_shadow='0 2px 8px rgba(0,0,0,0.05)',
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
        # Header dengan typography yang lebih baik
        self.header_widget.value = f"""
        <div style='color: #2c3e50; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;'>
            <h3 style='font-size: 17px; font-weight: 600; margin: 0 0 4px 0; padding: 0; line-height: 1.4; color: #2c3e50;'>
                {operation}
            </h3>
        </div>
        """
    
    def update_status(self, message: str, style: str = None):
        """Update status message dengan flexbox layout dan emoji handling"""
        color_map = {
            'success': '#28a745', 'info': '#007bff', 
            'warning': '#ffc107', 'error': '#dc3545'
        }
        color = color_map.get(style, '#495057')
        
        # Clean message dari emoji duplikat jika ada
        clean_message = self._clean_status_message(message)
        
        # Status message dengan style yang lebih modern
        bg_color = {
            'success': 'rgba(40, 167, 69, 0.08)',
            'info': 'rgba(0, 123, 255, 0.08)',
            'warning': 'rgba(255, 193, 7, 0.08)',
            'error': 'rgba(220, 53, 69, 0.08)'
        }.get(style, 'rgba(233, 236, 239, 0.5)')
        
        self.status_widget.value = f"""
        <div style="
            display: flex; 
            align-items: center; 
            width: 100%;
            color: {color};
            font-size: 13.5px;
            font-weight: 500;
            margin: 0;
            padding: 12px 16px;
            background: {bg_color};
            border-radius: 8px;
            border-left: 3px solid {color};
            box-sizing: border-box;
            word-wrap: break-word;
            overflow-wrap: break-word;
            line-height: 1.5;
            transition: all 0.2s ease;
        ">
            {clean_message}
        </div>
        """
    
    @staticmethod
    def _clean_status_message(message: str) -> str:
        """Clean status message dari emoji duplikat"""
        import re
        # Check jika sudah ada emoji status di awal
        if re.match(r'^[✅❌⚠️ℹ️🚀]\s', message):
            return message
        # Jika tidak ada, return message as is (emoji akan ditambah di level yang memanggil)
        return message