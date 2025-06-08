"""
File: smartcash/ui/dataset/augmentation/components/advanced_opts_widget.py
Deskripsi: Advanced options widget dengan tab layout dan parameter yang moderat
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_advanced_options_widget() -> Dict[str, Any]:
    """
    Create advanced options dengan tab layout dan range validation
    
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
    
    # Parameter guidance info
    parameter_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: {COLORS.get('bg_light', '#f8f9fa')}; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;">
            <strong>{ICONS.get('info', 'ℹ️')} Parameter Guidance:</strong><br>
            • <strong>Rotasi:</strong> 5-15° optimal untuk uang kertas<br>
            • <strong>Translasi & Skala:</strong> 0.05-0.15 untuk menjaga proporsi<br>
            • <strong>HSV Hue:</strong> 0.01-0.03 untuk variasi warna natural<br>
            • <strong>Brightness/Contrast:</strong> 0.1-0.3 untuk pencahayaan realistis
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Tab content dengan organized layout
    position_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 5px 0;'>📍 Parameter Posisi</h6>"),
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Transformasi geometri untuk variasi posisi uang kertas</p>"),
        fliplr, degrees, translate, scale
    ], layout=widgets.Layout(padding='5px'))
    
    lighting_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {COLORS.get('dark', '#333')}; margin: 5px 0;'>💡 Parameter Pencahayaan</h6>"),
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Variasi pencahayaan untuk kondisi lingkungan berbeda</p>"),
        hsv_h, hsv_s, brightness, contrast
    ], layout=widgets.Layout(padding='5px'))
    
    # Tab widget
    tabs = widgets.Tab(children=[position_tab, lighting_tab])
    tabs.set_title(0, "📍 Posisi")
    tabs.set_title(1, "💡 Pencahayaan")
    
    # Main container
    container = widgets.VBox([
        tabs,
        parameter_info
    ], layout=widgets.Layout(margin='10px 0', width='100%'))
    
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