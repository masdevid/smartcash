"""
File: smartcash/ui/components/progress_tracker/ui_components.py
Deskripsi: Manager UI components tanpa step info dan auto hide support
"""

import threading
import time
import ipywidgets as widgets
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel

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
        """Create UI widgets dengan flexbox layout dan full width"""
        self.header_widget = widgets.HTML("", layout=widgets.Layout(
            width='100%', margin='0 0 10px 0'
        ))
        
        self.status_widget = widgets.HTML("", layout=widgets.Layout(
            width='100%', margin='0 0 8px 0'
        ))
        
        self.progress_output = widgets.Output(layout=widgets.Layout(
            width='100%', margin='5px 0', border='1px solid #ddd',
            border_radius='4px', padding='8px', min_height='60px',
            display='flex', flex_direction='column'
        ))
        
        self._create_container()
    
    def _create_container(self):
        """Create main container dengan flexbox layout"""
        self.container = widgets.VBox(
            [self.header_widget, self.status_widget, self.progress_output],
            layout=widgets.Layout(
                display='none', width='100%', margin='10px 0', padding='15px',
                border='1px solid #28a745', border_radius='8px', 
                background_color='#f8fff8', min_height=self.config.get_container_height(),
                max_height='300px', overflow='hidden', box_sizing='border-box'
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
        self.header_widget.value = f"""
        <h4 style='color: #333; margin: 0; font-size: 16px; font-weight: 600;'>
        📊 {operation}</h4>
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
        
        self.status_widget.value = f"""
        <div style="display: flex; align-items: center; width: 100%; 
                    color: {color}; font-size: 13px; font-weight: 500; margin: 0; 
                    padding: 8px 12px; background: rgba(233, 236, 239, 0.5); 
                    border-radius: 6px; border-left: 3px solid {color}; 
                    box-sizing: border-box; word-wrap: break-word; 
                    overflow-wrap: break-word; line-height: 1.4;">
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