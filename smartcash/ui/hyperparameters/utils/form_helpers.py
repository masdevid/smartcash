"""
File: smartcash/ui/hyperparameters/utils/form_helpers.py
Deskripsi: Helper functions untuk form widgets dengan parameter order yang benar
"""

from typing import Dict, Any, List
import ipywidgets as widgets


def create_slider_widget(value: float, min_val: float, max_val: float, step: float, 
                        description: str, readout_format: str = '.3f') -> widgets.FloatSlider:
    """Create float slider dengan standard layout"""
    return widgets.FloatSlider(
        value=value, min=min_val, max=max_val, step=step, description=description,
        readout_format=readout_format, style={'description_width': '140px'},
        layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_int_slider_widget(value: int, min_val: int, max_val: int, 
                           description: str, step: int = 1) -> widgets.IntSlider:
    """Create int slider dengan standard layout"""
    return widgets.IntSlider(
        value=value, min=min_val, max=max_val, step=step, description=description,
        style={'description_width': '140px'}, layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_dropdown_widget(value: str, options: List[str], description: str) -> widgets.Dropdown:
    """Create dropdown dengan parameter order yang benar: value, options, description"""
    # Pastikan value ada di options, fallback ke first option
    safe_value = value if value in options else (options[0] if options else value)
    
    return widgets.Dropdown(
        value=safe_value, options=options, description=description,
        style={'description_width': '140px'}, layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_checkbox_widget(value: bool, description: str) -> widgets.Checkbox:
    """Create checkbox dengan standard layout"""
    return widgets.Checkbox(
        value=value, description=description,
        style={'description_width': 'initial'}, layout=widgets.Layout(width='100%', margin='2px 0')
    )


def create_summary_cards_widget(data: Dict[str, str] = None) -> widgets.HBox:
    """Create summary cards untuk menampilkan config overview"""
    if not data:
        data = {
            'Training': 'Epochs: 100, Batch: 16, LR: 0.0100',
            'Optimizer': 'SGD (decay: 0.0005)',
            'Loss': 'Box: 0.05, Cls: 0.5, Obj: 1.0',
            'Control': 'Early Stop: On, Save Best: On'
        }
    
    cards = [_create_summary_card(title, content) for title, content in data.items()]
    
    return widgets.HBox(cards, layout=widgets.Layout(
        width='100%', justify_content='space-between', flex_wrap='wrap'
    ))


def _create_summary_card(title: str, content: str) -> widgets.HTML:
    """Create individual summary card"""
    return widgets.HTML(f"""
        <div style='background: #f8f9fa; border: 1px solid #dee2e6; 
                    border-radius: 6px; padding: 8px; margin: 2px; min-width: 180px;'>
            <strong style='color: #495057;'>{title}:</strong><br>
            <span style='color: #6c757d; font-size: 0.9em;'>{content}</span>
        </div>
    """)


def create_section_card(title: str, widgets_list: List, color: str = '#2196f3') -> widgets.VBox:
    """Create section card dengan header dan widgets"""
    header = widgets.HTML(f"""
        <div style='background: linear-gradient(135deg, {color} 0%, {color}aa 100%); 
                    color: white; padding: 8px 12px; border-radius: 6px; 
                    font-weight: bold; margin-bottom: 8px;'>
            {title}
        </div>
    """)
    
    return widgets.VBox([header] + widgets_list, layout=widgets.Layout(
        padding='8px', border='1px solid #dee2e6', border_radius='6px',
        background_color='#fafafa', margin='4px 0'
    ))


def create_responsive_grid_layout(cards: List[widgets.Widget]) -> widgets.HBox:
    """Create responsive grid layout untuk cards dengan flexbox"""
    return widgets.HBox(
        cards,
        layout=widgets.Layout(
            width='100%', display='flex', flex_flow='row wrap',
            justify_content='space-between', align_items='stretch',
            gap='10px', overflow='hidden'
        )
    )