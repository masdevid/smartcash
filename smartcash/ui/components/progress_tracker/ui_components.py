"""
File: smartcash/ui/components/progress_tracker/ui_components.py
Deskripsi: Manager untuk UI components dengan widget management
"""

import ipywidgets as widgets
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel

class UIComponentsManager:
    """Manager untuk UI components dengan widget management"""
    
    def __init__(self, config: ProgressConfig):
        self.config = config
        self.header_widget = None
        self.status_widget = None
        self.step_info_widget = None
        self.progress_output = None
        self.container = None
        self.is_visible = False
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create semua UI widgets"""
        self.header_widget = widgets.HTML("", layout=widgets.Layout(
            margin='0 0 10px 0', width='100%'
        ))
        
        self.status_widget = widgets.HTML("", layout=widgets.Layout(
            margin='0 0 8px 0', width='100%'
        ))
        
        self.step_info_widget = widgets.HTML("", layout=widgets.Layout(
            margin='0 0 5px 0', width='100%',
            display='block' if self.config.level == ProgressLevel.TRIPLE else 'none'
        ))
        
        self.progress_output = widgets.Output(layout=widgets.Layout(
            width='100%', margin='5px 0', border='1px solid #ddd',
            border_radius='4px', padding='5px'
        ))
        
        self._create_container()
    
    def _create_container(self):
        """Create main container widget"""
        progress_components = [self.progress_output]
        
        self.container = widgets.VBox(
            [self.header_widget, self.status_widget, self.step_info_widget] + progress_components,
            layout=widgets.Layout(
                display='none', flex_flow='column nowrap', align_items='stretch',
                margin='10px 0', padding='15px', border='1px solid #28a745',
                border_radius='8px', background_color='#f8fff8', width='100%',
                min_height=self.config.get_container_height(),
                max_height='400px', overflow='hidden', box_sizing='border-box'
            )
        )
    
    def show(self):
        """Show container"""
        self.container.layout.display = 'flex'
        self.container.layout.visibility = 'visible'
        self.is_visible = True
    
    def hide(self):
        """Hide container"""
        self.container.layout.display = 'none'
        self.container.layout.visibility = 'hidden'
        self.is_visible = False
    
    def update_header(self, operation: str):
        """Update header dengan operation name"""
        self.header_widget.value = f"""
        <h4 style='color: #333; margin: 0; font-size: 16px; font-weight: 600;'>
        📊 {operation}</h4>
        """
    
    def update_status(self, message: str, style: str = None):
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
    
    def update_step_info(self, current_step_index: int, steps: list, step_weights: dict):
        """Update step information untuk TRIPLE level"""
        if (self.config.level != ProgressLevel.TRIPLE or 
            not steps or current_step_index >= len(steps)):
            return
        
        current_step = steps[current_step_index]
        weight = step_weights.get(current_step, 0)
        
        step_info = f"""
        <div style="padding: 8px; background: #e3f2fd; border-radius: 4px; margin: 2px 0;">
            <small style="color: #1976d2;">
                <strong>Step {current_step_index + 1}/{len(steps)}:</strong> 
                {current_step.title()} 
                <span style="color: #666;">(Weight: {weight}%)</span>
            </small>
        </div>
        """
        self.step_info_widget.value = step_info