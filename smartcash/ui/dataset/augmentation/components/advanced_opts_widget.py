"""
File: smartcash/ui/dataset/augmentation/components/advanced_opts_widget.py
Deskripsi: Advanced options widget dengan parameter info berwarna sesuai border
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_advanced_options_widget() -> Dict[str, Any]:
    """
    Create advanced options dengan colored parameter info dan tab layout
    
    Returns:
        Dictionary berisi container dan widget mapping
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Position Parameters Tab
    fliplr = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.05,
        description='Flip Horizontal:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    degrees = widgets.IntSlider(
        value=10, min=0, max=30, step=2,
        description='Rotasi (°):',
        continuous_update=False,
        readout=True,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    translate = widgets.FloatSlider(
        value=0.1, min=0.0, max=0.25, step=0.02,
        description='Translasi:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    scale = widgets.FloatSlider(
        value=0.1, min=0.0, max=0.25, step=0.02,
        description='Skala:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Lighting Parameters Tab
    hsv_h = widgets.FloatSlider(
        value=0.015, min=0.0, max=0.05, step=0.002,
        description='HSV Hue:',
        continuous_update=False,
        readout=True, readout_format='.3f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    hsv_s = widgets.FloatSlider(
        value=0.7, min=0.0, max=1.0, step=0.05,
        description='HSV Saturation:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    brightness = widgets.FloatSlider(
        value=0.2, min=0.0, max=0.4, step=0.05,
        description='Brightness:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    contrast = widgets.FloatSlider(
        value=0.2, min=0.0, max=0.4, step=0.05,
        description='Contrast:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Position parameter info - green theme
    position_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: #4caf5015; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #4caf5040;">
            <strong style="color: #2e7d32;">{ICONS.get('info', 'ℹ️')} Parameter Posisi:</strong><br>
            • <strong style="color: #2e7d32;">Rotasi:</strong> 5-15° optimal untuk uang kertas<br>
            • <strong style="color: #2e7d32;">Translasi & Skala:</strong> 0.05-0.15 untuk menjaga proporsi<br>
            • <strong style="color: #388e3c;">Flip:</strong> 0.5 = 50% kemungkinan horizontal flip
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Lighting parameter info - purple theme
    lighting_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: #9c27b015; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #9c27b040;">
            <strong style="color: #7b1fa2;">{ICONS.get('info', 'ℹ️')} Parameter Pencahayaan:</strong><br>
            • <strong style="color: #7b1fa2;">HSV Hue:</strong> 0.01-0.03 untuk variasi warna natural<br>
            • <strong style="color: #7b1fa2;">Brightness/Contrast:</strong> 0.1-0.3 untuk pencahayaan realistis<br>
            • <strong style="color: #8e24aa;">HSV Saturation:</strong> 0.5-0.9 untuk saturasi optimal
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Tab content dengan organized layout
    position_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: #2e7d32; margin: 5px 0;'>📍 Parameter Posisi</h6>"),
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Transformasi geometri untuk variasi posisi uang kertas</p>"),
        fliplr, degrees, translate, scale,
        position_info
    ], layout=widgets.Layout(padding='5px'))
    
    lighting_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: #7b1fa2; margin: 5px 0;'>💡 Parameter Pencahayaan</h6>"),
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Variasi pencahayaan untuk kondisi lingkungan berbeda</p>"),
        hsv_h, hsv_s, brightness, contrast,
        lighting_info
    ], layout=widgets.Layout(padding='5px'))
    
    # Tab widget
    tabs = widgets.Tab(children=[position_tab, lighting_tab])
    tabs.set_title(0, "📍 Posisi")
    tabs.set_title(1, "💡 Pencahayaan")
    
    # Main container
    container = widgets.VBox([tabs], layout=widgets.Layout(margin='10px 0', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            # Position parameters
            'fliplr': fliplr,
            'degrees': degrees,
            'translate': translate,
            'scale': scale,
            # Lighting parameters
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'brightness': brightness,
            'contrast': contrast
        },
        # Parameter mapping untuk easy access
        'parameter_groups': {
            'position': ['fliplr', 'degrees', 'translate', 'scale'],
            'lighting': ['hsv_h', 'hsv_s', 'brightness', 'contrast']
        },
        # Validation ranges
        'validation': {
            'ranges': {
                'fliplr': (0.0, 1.0),
                'degrees': (0, 30),
                'translate': (0.0, 0.25),
                'scale': (0.0, 0.25),
                'hsv_h': (0.0, 0.05),
                'hsv_s': (0.0, 1.0),
                'brightness': (0.0, 0.4),
                'contrast': (0.0, 0.4)
            }
        }
    }