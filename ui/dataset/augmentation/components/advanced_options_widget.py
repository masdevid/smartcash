"""
File: smartcash/ui/dataset/augmentation/components/advanced_options_widget.py
Deskripsi: Advanced options widget dengan nilai yang moderat untuk pipeline research
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_advanced_options_widget() -> Dict[str, Any]:
    """
    Buat widget UI untuk opsi lanjutan dengan nilai moderat untuk penelitian.
    
    Returns:
        Dictionary berisi container dan mapping widget individual
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Parameter posisi dengan nilai moderat untuk penelitian
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
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    degrees = widgets.IntSlider(
        value=10,  # Reduced dari 15 - lebih konservatif untuk uang kertas
        min=0,
        max=30,    # Reduced dari 45 - range yang lebih realistic
        step=2,    # Step lebih kecil untuk fine-tuning
        description='Rotasi (°):',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    translate = widgets.FloatSlider(
        value=0.1,  # Reduced dari 0.15 - translasi minimal
        min=0.0,
        max=0.25,   # Reduced dari 0.5 - range lebih kecil
        step=0.02,  # Step lebih halus
        description='Translasi:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    scale = widgets.FloatSlider(
        value=0.1,  # Reduced dari 0.15 - scaling minimal
        min=0.0,
        max=0.25,   # Reduced dari 0.5 - tidak terlalu ekstrim
        step=0.02,  # Step lebih halus
        description='Skala:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Parameter pencahayaan dengan nilai penelitian yang optimal
    hsv_h = widgets.FloatSlider(
        value=0.015,  # Reduced dari 0.025 - hue shift minimal
        min=0.0,
        max=0.05,     # Reduced dari 0.1 - range lebih kecil
        step=0.002,   # Step lebih halus
        description='HSV Hue:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.3f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    hsv_s = widgets.FloatSlider(
        value=0.7,
        min=0.0,
        max=1.0,
        step=0.05,  # Step lebih halus
        description='HSV Saturation:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    brightness = widgets.FloatSlider(
        value=0.2,  # Reduced dari 0.3 - brightness moderat
        min=0.0,
        max=0.4,    # Reduced dari 1.0 - range lebih realistic
        step=0.05,  # Step lebih halus
        description='Brightness:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    contrast = widgets.FloatSlider(
        value=0.2,  # Reduced dari 0.3 - contrast moderat
        min=0.0,
        max=0.4,    # Reduced dari 1.0 - range lebih realistic
        step=0.05,  # Step lebih halus
        description='Contrast:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Info panel untuk parameter guidance
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
    
    # Tab layout dengan info yang lebih informatif
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
    
    tabs = widgets.Tab(children=[position_tab, lighting_tab])
    tabs.set_title(0, "📍 Posisi")
    tabs.set_title(1, "💡 Pencahayaan")
    
    container = widgets.VBox([
        tabs,
        parameter_info
    ], layout=widgets.Layout(margin='10px 0', width='100%'))
    
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
        },
        # Research pipeline mapping untuk easy access
        'research_params': {
            'position': ['fliplr', 'degrees', 'translate', 'scale'],
            'lighting': ['hsv_h', 'hsv_s', 'brightness', 'contrast'],
            'combined': ['fliplr', 'degrees', 'translate', 'scale', 'hsv_h', 'hsv_s', 'brightness', 'contrast']
        }
    }