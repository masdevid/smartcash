"""
File: smartcash/ui/dataset/augmentation/components/advanced_opts_widget.py
Deskripsi: Advanced options widget dengan HSV parameters dan styling terkonsolidasi
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.style_utils import (
    tabbed_container, create_info_content, info_panel
)

def create_advanced_options_widget() -> Dict[str, Any]:
    """Create advanced options dengan HSV parameters dan tabbed layout"""
    # Position parameters dengan overflow-safe styling
    position_widgets = {
        'fliplr': widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.05, description='Flip Horizontal:',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'degrees': widgets.IntSlider(
            value=12, min=0, max=30, step=1, description='Rotasi (°):',
            continuous_update=False, readout=True,
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'translate': widgets.FloatSlider(
            value=0.08, min=0.0, max=0.25, step=0.01, description='Translasi:',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'scale': widgets.FloatSlider(
            value=0.04, min=0.0, max=0.25, step=0.01, description='Skala:',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        )
    }
    
    # Lighting parameters dengan HSV support
    lighting_widgets = {
        'brightness': widgets.FloatSlider(
            value=0.2, min=0.0, max=0.4, step=0.02, description='Brightness:',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'contrast': widgets.FloatSlider(
            value=0.15, min=0.0, max=0.4, step=0.02, description='Contrast:',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'hsv_h': widgets.IntSlider(
            value=10, min=0, max=30, step=1, description='HSV Hue:',
            continuous_update=False, readout=True,
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'hsv_s': widgets.IntSlider(
            value=15, min=0, max=50, step=1, description='HSV Saturation:',
            continuous_update=False, readout=True,
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        )
    }
    
    # Create info content untuk setiap tab
    position_info = create_info_content([
        ('Parameter Posisi', ''),
        ('Rotasi', '0-30° (optimal: 8-15°)'),
        ('Translasi & Skala', '0.0-0.25'),
        ('Flip', '0.0-1.0 (50% probabilitas)'),
        ('Backend', 'Albumentations compatible')
    ], theme='advanced')
    
    lighting_info = create_info_content([
        ('Parameter Pencahayaan', ''),
        ('Brightness/Contrast', '0.0-0.4'),
        ('HSV Hue', '0-30 (color shift)'),
        ('HSV Saturation', '0-50 (saturation shift)'),
        ('Backend', 'OpenCV HSV compatible')
    ], theme='advanced')
    
    # Create tab content
    position_content = widgets.VBox([
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Transformasi geometri dengan Albumentations</p>"),
        *position_widgets.values(),
        info_panel(position_info, theme='advanced')
    ])
    
    lighting_content = widgets.VBox([
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Variasi pencahayaan dengan OpenCV HSV</p>"),
        *lighting_widgets.values(),
        info_panel(lighting_info, theme='advanced')
    ])
    
    # Create tabbed container
    tabs_config = [
        ('📍 Posisi', position_content),
        ('💡 Pencahayaan', lighting_content)
    ]
    
    container = tabbed_container(tabs_config, theme='advanced')
    
    # Combine all widgets
    all_widgets = {**position_widgets, **lighting_widgets}
    
    return {
        'container': container,
        'widgets': all_widgets,
        'backend_mapping': {
            'position': {
                'horizontal_flip': 'fliplr',
                'rotation_limit': 'degrees',
                'translate_limit': 'translate',
                'scale_limit': 'scale'
            },
            'lighting': {
                'brightness_limit': 'brightness',
                'contrast_limit': 'contrast',
                'hsv_hue': 'hsv_h',
                'hsv_saturation': 'hsv_s'
            }
        },
        'validation': {
            'ranges': {
                'fliplr': (0.0, 1.0),
                'degrees': (0, 30),
                'translate': (0.0, 0.25),
                'scale': (0.0, 0.25),
                'brightness': (0.0, 0.4),
                'contrast': (0.0, 0.4),
                'hsv_h': (0, 30),
                'hsv_s': (0, 50)
            }
        }
    }