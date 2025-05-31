
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
    """Membuat komponen log accordion dengan one-liner style."""
    output_widget = output_widget or widgets.Output(layout=widgets.Layout(max_height=height, overflow='auto', border='1px solid #ddd', padding='10px'))
    
    def append_log(message: str, level: str = 'info', namespace: str = None, module: str = None) -> None:
        """Menambahkan log dengan format yang rapi dalam one-liner."""
        level_to_color = {'debug': '#6c757d', 'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545', 'critical': '#dc3545'}
        now = datetime.datetime.now().strftime("%H:%M:%S")
        prefix = f"<span style='color: #6610f2;'>[{(namespace or module or '').split('.')[-1]}]</span> " if namespace or module else ""
        level_color = level_to_color.get(level, '#007bff')
        level_display = f"<span style='color: {level_color};'>{level.upper()}</span>"
        formatted_message = f"<span style='color: #666;'>{now}</span> {level_display} {prefix}{message}"
        with output_widget: display(widgets.HTML(formatted_message))
    
    setattr(output_widget, 'append_log', append_log)
    log_accordion = widgets.Accordion(children=[output_widget], layout=widgets.Layout(width=width, margin='10px 0'))
    log_accordion.set_title(0, f"{ICONS.get('log', '📋')} Log {module_name.capitalize()}")
    return {'log_output': output_widget, 'log_accordion': log_accordion}

def update_log(ui_components: Dict[str, Any], message: str, expand: bool = False, clear: bool = False) -> None:
    """Update log dengan one-liner style."""
    if 'log_output' not in ui_components: return
    with ui_components['log_output']:
        clear and __import__('IPython.display', fromlist=['clear_output']).clear_output(wait=True)
        display(widgets.HTML(f"<p>{message}</p>"))
    expand and 'log_accordion' in ui_components and setattr(ui_components['log_accordion'], 'selected_index', 0)

