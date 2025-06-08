
"""
File: smartcash/ui/components/log_accordion.py
Deskripsi: Komponen log accordion dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from IPython.display import display
from smartcash.ui.utils.constants import ICONS, COLORS
import datetime

def create_log_accordion(module_name: str = 'process', height: str = '200px', width: str = '100%', 
                        output_widget: Optional[widgets.Output] = None) -> Dict[str, widgets.Widget]:
    """Membuat komponen log accordion dengan FIXED overflow issues."""
    output_widget = output_widget or widgets.Output(layout=widgets.Layout(
        max_height=height, 
        width='100%',
        overflow='hidden', 
        border='1px solid #ddd', 
        padding='8px',
        box_sizing='border-box',
        word_wrap='break-word'
    ))
    
    def append_log(message: str, level: str = 'info', namespace: str = None, module: str = None) -> None:
        """Menambahkan log dengan format yang rapi dan FIXED overflow."""
        level_to_color = {'debug': '#6c757d', 'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545', 'critical': '#dc3545'}
        now = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = f"<span style='color: #6610f2;'>[{(namespace or module or '').split('.')[-1]}]</span> " if namespace or module else ""
        level_color = level_to_color.get(level, '#007bff')
        level_display = f"<span style='color: {level_color};'>{level.upper()}</span>"
        
        # FIXED: Add proper text wrapping and overflow handling
        formatted_message = f"""
        <div style='margin: 2px 0; padding: 4px; word-wrap: break-word; overflow-wrap: break-word; 
                    white-space: pre-wrap; max-width: 100%; overflow: hidden;'>
            <span style='color: #666; font-size: 12px;'>{now}</span> 
            {level_display} 
            {prefix}
            <span style='word-wrap: break-word; overflow-wrap: break-word;'>{message}</span>
        </div>"""
        
        with output_widget: display(widgets.HTML(formatted_message))
    
    setattr(output_widget, 'append_log', append_log)
    log_accordion = widgets.Accordion(children=[output_widget], layout=widgets.Layout(
        width=width, 
        margin='10px 0',
        overflow='hidden',
        box_sizing='border-box',
        display='flex',
        flex_flow='column',
        justify_content='space-between'
    ))
    log_accordion.set_title(0, f"{ICONS.get('log', '📋')} Log {module_name.capitalize()}")
    log_accordion.selected_index = 0  # Open by default
    return {'log_output': output_widget, 'log_accordion': log_accordion}

def update_log(ui_components: Dict[str, Any], message: str, expand: bool = False, clear: bool = False) -> None:
    """Update log dengan one-liner style."""
    if 'log_output' not in ui_components: return
    with ui_components['log_output']:
        clear and __import__('IPython.display', fromlist=['clear_output']).clear_output(wait=True)
        display(widgets.HTML(f"<p>{message}</p>"))
    expand and 'log_accordion' in ui_components and setattr(ui_components['log_accordion'], 'selected_index', 0)