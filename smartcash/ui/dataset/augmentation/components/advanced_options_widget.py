# File: smartcash/ui/dataset/augmentation/components/advanced_options_widget.py

import ipywidgets as widgets
from typing import Dict, Any
def create_advanced_options_widget() -> Dict[str, Any]:
    """
    Buat widget UI murni untuk opsi lanjutan (tanpa logika bisnis).
    
    Returns:
        Dictionary berisi container dan mapping widget individual
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Parameter posisi
    fliplr = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Flip Horizontal:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%')
    )
    
    degrees = widgets.IntSlider(
        value=15,
        min=0,
        max=45,
        step=5,
        description='Rotasi (°):',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        layout=widgets.Layout(width='95%')
    )
    
    translate = widgets.FloatSlider(
        value=0.15,
        min=0.0,
        max=0.5,
        step=0.05,
        description='Translasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%')
    )
    
    scale = widgets.FloatSlider(
        value=0.15,
        min=0.0,
        max=0.5,
        step=0.05,
        description='Skala:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%')
    )
    
    # Parameter pencahayaan
    hsv_h = widgets.FloatSlider(
        value=0.025,
        min=0.0,
        max=0.1,
        step=0.005,
        description='HSV Hue:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout=widgets.Layout(width='95%')
    )
    
    hsv_s = widgets.FloatSlider(
        value=0.7,
        min=0.0,
        max=1.0,
        step=0.1,
        description='HSV Saturation:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    brightness = widgets.FloatSlider(
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.1,
        description='Brightness:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    contrast = widgets.FloatSlider(
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.1,
        description='Contrast:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    # Tab layout
    position_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>Parameter Posisi</h6>"),
        fliplr, degrees, translate, scale
    ])
    
    lighting_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS['dark']}; margin: 5px 0;'>Parameter Pencahayaan</h6>"),
        hsv_h, hsv_s, brightness, contrast
    ])
    
    tabs = widgets.Tab(children=[position_tab, lighting_tab])
    tabs.set_title(0, "Posisi")
    tabs.set_title(1, "Pencahayaan")
    
    container = widgets.VBox([tabs], layout=widgets.Layout(margin='10px 0'))
    
    return {
        'container': container,
        'widgets': {
            'fliplr': fliplr,
            'degrees': degrees,
            'translate': translate,
            'scale': scale,
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'brightness': brightness,
            'contrast': contrast
        }
    }
