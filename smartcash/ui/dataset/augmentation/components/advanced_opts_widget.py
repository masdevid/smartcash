"""
File: smartcash/ui/dataset/augmentation/components/advanced_opts_widget.py
Deskripsi: Advanced options widget dengan HSV parameters dan styling terkonsolidasi
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_advanced_options_widget() -> Dict[str, Any]:
    """Create advanced options dengan HSV parameters dan tabbed layout"""
    # Position parameters dengan overflow-safe styling
    position_widgets = {
        'fliplr': widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.05, description='Balik Horizontal (0.0-1.0):',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'degrees': widgets.IntSlider(
            value=12, min=0, max=30, step=1, description='Rotasi (0-30°):',
            continuous_update=False, readout=True,
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'translate': widgets.FloatSlider(
            value=0.08, min=0.0, max=0.25, step=0.01, description='Translasi (0.0-0.25):',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'scale': widgets.FloatSlider(
            value=0.04, min=0.0, max=0.25, step=0.01, description='Skala (0.0-0.25):',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        )
    }
    
    # Lighting parameters dengan HSV support
    lighting_widgets = {
        'brightness': widgets.FloatSlider(
            value=0.2, min=0.0, max=0.4, step=0.02, description='Kecerahan (0.0-0.4):',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'contrast': widgets.FloatSlider(
            value=0.15, min=0.0, max=0.4, step=0.02, description='Kontras (0.0-0.4):',
            continuous_update=False, readout=True, readout_format='.2f',
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'hsv_h': widgets.IntSlider(
            value=10, min=0, max=30, step=1, description='Warna HSV (0-30):',
            continuous_update=False, readout=True,
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        ),
        'hsv_s': widgets.IntSlider(
            value=15, min=0, max=50, step=1, description='Saturasi HSV (0-50):',
            continuous_update=False, readout=True,
            style={'description_width': '100px'}, layout=widgets.Layout(width='100%', max_width='100%')
        )
    }
    
    # Create info content with simple HTML
    position_info = widgets.HTML("""
    <div style='background: #f0f8ff; padding: 8px; border-radius: 4px; margin: 8px 0; font-size: 12px;'>
        <strong>Parameter Posisi:</strong><br>
        • Rotasi: 0-30° (optimal: 8-15°)<br>
        • Translasi & Skala: 0.0-0.25<br>
        • Flip: 0.0-1.0 (50% probabilitas)<br>
        • Backend: Kompatibel dengan Albumentations
    </div>
    """)
    
    lighting_info = widgets.HTML("""
    <div style='background: #fff8f0; padding: 8px; border-radius: 4px; margin: 8px 0; font-size: 12px;'>
        <strong>Parameter Pencahayaan:</strong><br>
        • Kecerahan/Kontras: 0.0-0.4<br>
        • Warna HSV: 0-30 (perubahan warna)<br>
        • Saturasi HSV: 0-50 (perubahan saturasi)<br>
        • Backend: Kompatibel dengan OpenCV HSV
    </div>
    """)
    
    # Create tab content
    position_content = widgets.VBox([
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Transformasi geometri menggunakan Albumentations</p>"),
        *position_widgets.values(),
        position_info
    ])
    
    lighting_content = widgets.VBox([
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Variasi pencahayaan menggunakan OpenCV HSV</p>"),
        *lighting_widgets.values(),
        lighting_info
    ])
    
    # Create simple tab container
    container = widgets.Tab([position_content, lighting_content])
    container.set_title(0, '📍 Posisi')
    container.set_title(1, '💡 Pencahayaan')
    
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