"""
File: smartcash/ui/dataset/augmentation/components/advanced_opts_widget.py
Deskripsi: Advanced options dengan warna info konsisten antara posisi dan pencahayaan
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_advanced_options_widget() -> Dict[str, Any]:
    """Create advanced options dengan consistent info colors"""
    
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Position Parameters
    fliplr = widgets.FloatSlider(
        value=0.5, min=0.0, max=1.0, step=0.05,
        description='Flip Horizontal:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    degrees = widgets.IntSlider(
        value=10, min=0, max=30, step=1,
        description='Rotasi (°):',
        continuous_update=False,
        readout=True,
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    translate = widgets.FloatSlider(
        value=0.1, min=0.0, max=0.25, step=0.01,
        description='Translasi:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    scale = widgets.FloatSlider(
        value=0.1, min=0.0, max=0.25, step=0.01,
        description='Skala:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # Lighting Parameters
    hsv_h = widgets.FloatSlider(
        value=0.015, min=0.0, max=0.05, step=0.001,
        description='HSV Hue:',
        continuous_update=False,
        readout=True, readout_format='.3f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    hsv_s = widgets.FloatSlider(
        value=0.7, min=0.0, max=1.0, step=0.02,
        description='HSV Saturation:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    brightness = widgets.FloatSlider(
        value=0.2, min=0.0, max=0.4, step=0.02,
        description='Brightness:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    contrast = widgets.FloatSlider(
        value=0.2, min=0.0, max=0.4, step=0.02,
        description='Contrast:',
        continuous_update=False,
        readout=True, readout_format='.2f',
        layout=widgets.Layout(width='95%'),
        style={'description_width': '120px'}
    )
    
    # FIXED: Purple colors matching header box
    info_color = "#9c27b0"  # Purple matching header
    info_bg = "#9c27b015"   # Purple background
    info_border = "#9c27b0" # Purple border
    
    position_info = widgets.HTML(
        f"""
        <div style="padding: 6px; background-color: {info_bg}; 
                    border-radius: 4px; margin: 5px 0; font-size: 10px;
                    border: 1px solid {info_border}40; line-height: 1.2;">
            <strong style="color: {info_color};">{ICONS.get('info', 'ℹ️')} Parameter Posisi:</strong><br>
            • <strong style="color: {info_color};">Rotasi:</strong> 0-30° (optimal: 8-15°)<br>
            • <strong style="color: {info_color};">Translasi & Skala:</strong> 0.0-0.25<br>
            • <strong style="color: {info_color};">Flip:</strong> 0.0-1.0 (50% probabilitas)<br>
            • <strong style="color: {info_color};">Backend:</strong> Albumentations compatible
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    lighting_info = widgets.HTML(
        f"""
        <div style="padding: 6px; background-color: {info_bg}; 
                    border-radius: 4px; margin: 5px 0; font-size: 10px;
                    border: 1px solid {info_border}40; line-height: 1.2;">
            <strong style="color: {info_color};">{ICONS.get('info', 'ℹ️')} Parameter Pencahayaan:</strong><br>
            • <strong style="color: {info_color};">HSV Hue:</strong> 0.0-0.05 (precision: 0.001)<br>
            • <strong style="color: {info_color};">HSV Saturation:</strong> 0.0-1.0<br>
            • <strong style="color: {info_color};">Brightness/Contrast:</strong> 0.0-0.4<br>
            • <strong style="color: {info_color};">Backend:</strong> OpenCV HSV compatible
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='3px 0')
    )
    
    # Tabs dengan consistent styling
    position_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {info_color}; margin: 5px 0;'>📍 Parameter Posisi</h6>"),
        widgets.HTML(f"<p style='font-size: 10px; color: #666; margin: 2px 0;'>Transformasi geometri dengan Albumentations</p>"),
        fliplr, degrees, translate, scale,
        position_info
    ], layout=widgets.Layout(padding='5px'))
    
    lighting_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: {info_color}; margin: 5px 0;'>💡 Parameter Pencahayaan</h6>"),
        widgets.HTML(f"<p style='font-size: 10px; color: #666; margin: 2px 0;'>Variasi pencahayaan dengan OpenCV HSV</p>"),
        hsv_h, hsv_s, brightness, contrast,
        lighting_info
    ], layout=widgets.Layout(padding='5px'))
    
    # Tabs
    tabs = widgets.Tab(children=[position_tab, lighting_tab])
    tabs.set_title(0, "📍 Posisi")
    tabs.set_title(1, "💡 Pencahayaan")
    
    # Main container
    container = widgets.VBox([tabs], layout=widgets.Layout(margin='10px 0', width='100%'))
    
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
        'backend_mapping': {
            'position': {
                'horizontal_flip': 'fliplr',
                'rotation_limit': 'degrees',
                'translate_limit': 'translate', 
                'scale_limit': 'scale'
            },
            'lighting': {
                'hsv_h_limit': 'hsv_h',
                'hsv_s_limit': 'hsv_s',
                'brightness_limit': 'brightness',
                'contrast_limit': 'contrast'
            }
        },
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