"""
File: smartcash/ui/components/split_config.py
Deskripsi: Komponen shared untuk konfigurasi split dataset dengan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import ICONS, COLORS

def create_split_config(title: str = "Konfigurasi Split Dataset", description: str = "Tentukan pembagian dataset untuk training, validation, dan testing",
                       train_value: float = 0.7, val_value: float = 0.2, test_value: float = 0.1,
                       min_value: float = 0.0, max_value: float = 1.0, step: float = 0.05,
                       width: str = "100%", icon: str = "split") -> Dict[str, Any]:
    """Buat komponen konfigurasi split dataset dengan one-liner style."""
    display_title = f"{ICONS.get(icon, '')} {title}" if icon and icon in ICONS else title
    header = widgets.HTML(f"<h4 style='margin-top: 5px; margin-bottom: 10px; color: {COLORS.get('dark', '#333')};'>{display_title}</h4>")
    description_widget = widgets.HTML(f"<div style='margin-bottom: 10px; color: {COLORS.get('secondary', '#666')};'>{description}</div>") if description else None
    
    train_slider = widgets.FloatSlider(value=train_value, min=min_value, max=max_value, step=step, description='Training:',
                                      style={'description_width': 'initial'}, layout=widgets.Layout(width=width), readout_format='.0%')
    val_slider = widgets.FloatSlider(value=val_value, min=min_value, max=max_value, step=step, description='Validation:',
                                    style={'description_width': 'initial'}, layout=widgets.Layout(width=width), readout_format='.0%')
    test_slider = widgets.FloatSlider(value=test_value, min=min_value, max=max_value, step=step, description='Testing:',
                                     style={'description_width': 'initial'}, layout=widgets.Layout(width=width), readout_format='.0%')
    
    total_output = widgets.HTML(f"<div style='margin-top: 10px;'><b>Total:</b> {(train_value + val_value + test_value) * 100:.0f}%</div>")
    
    def update_total(*args):
        total = train_slider.value + val_slider.value + test_slider.value
        color = COLORS.get('success', 'green') if abs(total - 1.0) < 0.01 else COLORS.get('error', 'red')
        setattr(total_output, 'value', f"<div style='margin-top: 10px;'><b>Total:</b> <span style='color: {color};'>{total * 100:.0f}%</span></div>")
    
    [slider.observe(update_total, names='value') for slider in [train_slider, val_slider, test_slider]]
    
    widgets_list = [header] + ([description_widget] if description_widget else []) + [train_slider, val_slider, test_slider, total_output]
    container = widgets.VBox(widgets_list, layout=widgets.Layout(margin='10px 0px', padding='10px', border='1px solid #eee', border_radius='4px'))
    
    return {'container': container, 'train_slider': train_slider, 'val_slider': val_slider, 'test_slider': test_slider, 'total_output': total_output, 'header': header}
