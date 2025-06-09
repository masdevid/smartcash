"""
File: smartcash/ui/dataset/augmentation/components/advanced_opts_widget.py
Deskripsi: Enhanced advanced options dengan backend mapping, validation ranges dan HSV support
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_advanced_options_widget() -> Dict[str, Any]:
    """
    Create advanced options dengan backend mapping dan enhanced validation
    
    Returns:
        Dictionary berisi container dan widget mapping untuk backend integration
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Position Parameters dengan backend-compatible ranges
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
    
    # Enhanced Lighting Parameters dengan HSV support
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
    
    # Enhanced info panels dengan backend compatibility info
    position_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: #e8f5e8; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #4caf5040;">
            <strong style="color: #2e7d32;">{ICONS.get('info', 'ℹ️')} Parameter Posisi (Backend Compatible):</strong><br>
            • <strong style="color: #2e7d32;">Rotasi:</strong> 0-30° (optimal: 8-15° untuk mata uang)<br>
            • <strong style="color: #2e7d32;">Translasi & Skala:</strong> 0.0-0.25 (step: 0.01 untuk presisi)<br>
            • <strong style="color: #2e7d32;">Flip:</strong> 0.0-1.0 (0.5 = 50% probabilitas horizontal flip)<br>
            • <strong style="color: #ff9800;">📡 Backend:</strong> Albumentations-compatible parameter mapping
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    lighting_info = widgets.HTML(
        f"""
        <div style="padding: 8px; background-color: #f3e5f5; 
                    border-radius: 4px; margin: 5px 0; font-size: 11px;
                    border: 1px solid #9c27b040;">
            <strong style="color: #7b1fa2;">{ICONS.get('info', 'ℹ️')} Parameter Pencahayaan (HSV Enhanced):</strong><br>
            • <strong style="color: #7b1fa2;">HSV Hue:</strong> 0.0-0.05 (step: 0.001 untuk color precision)<br>
            • <strong style="color: #7b1fa2;">HSV Saturation:</strong> 0.0-1.0 (step: 0.02 untuk smooth transitions)<br>
            • <strong style="color: #7b1fa2;">Brightness/Contrast:</strong> 0.0-0.4 (step: 0.02 untuk realistic lighting)<br>
            • <strong style="color: #ff9800;">📡 Backend:</strong> OpenCV HSV space + gamma correction support
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Enhanced tabs dengan backend indicators
    position_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: #2e7d32; margin: 5px 0;'>📍 Parameter Posisi</h6>"),
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Transformasi geometri dengan Albumentations backend compatibility</p>"),
        fliplr, degrees, translate, scale,
        position_info
    ], layout=widgets.Layout(padding='5px'))
    
    lighting_tab = widgets.VBox([
        widgets.HTML(f"<h6 style='color: #7b1fa2; margin: 5px 0;'>💡 Parameter Pencahayaan</h6>"),
        widgets.HTML("<p style='font-size: 10px; color: #666; margin: 2px 0;'>Variasi pencahayaan dengan OpenCV HSV space precision</p>"),
        hsv_h, hsv_s, brightness, contrast,
        lighting_info
    ], layout=widgets.Layout(padding='5px'))
    
    # Enhanced tabs dengan backend status
    tabs = widgets.Tab(children=[position_tab, lighting_tab])
    tabs.set_title(0, "📍 Posisi (Albumentations)")
    tabs.set_title(1, "💡 Pencahayaan (OpenCV HSV)")
    
    # Main container dengan backend compatibility indicator
    backend_status = widgets.HTML(
        f"""
        <div style="padding: 4px 8px; background-color: #e3f2fd; 
                    border-radius: 3px; margin: 5px 0; font-size: 10px;
                    border-left: 3px solid #2196f3;">
            <strong style="color: #1976d2;">🔗 Backend Integration:</strong> 
            Albumentations v1.3+ | OpenCV 4.8+ | NumPy array compatibility
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='0')
    )
    
    container = widgets.VBox([
        backend_status,
        tabs
    ], layout=widgets.Layout(margin='10px 0', width='100%'))
    
    return {
        'container': container,
        'widgets': {
            # Position parameters dengan backend mapping
            'fliplr': fliplr,
            'degrees': degrees, 
            'translate': translate,
            'scale': scale,
            # Enhanced lighting parameters
            'hsv_h': hsv_h,
            'hsv_s': hsv_s,
            'brightness': brightness,
            'contrast': contrast
        },
        
        # Backend mapping untuk service integration
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
        
        # Enhanced validation dengan backend constraints
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
            },
            'steps': {
                'fliplr': 0.05,
                'degrees': 1,
                'translate': 0.01,
                'scale': 0.01,
                'hsv_h': 0.001,
                'hsv_s': 0.02,
                'brightness': 0.02,
                'contrast': 0.02
            },
            'backend_constraints': {
                'albumentations_compatible': ['fliplr', 'degrees', 'translate', 'scale'],
                'opencv_compatible': ['hsv_h', 'hsv_s', 'brightness', 'contrast'],
                'requires_float32': ['fliplr', 'translate', 'scale', 'hsv_h', 'hsv_s', 'brightness', 'contrast'],
                'requires_int': ['degrees']
            }
        },
        
        # Parameter groups untuk batch operations
        'parameter_groups': {
            'position': ['fliplr', 'degrees', 'translate', 'scale'],
            'lighting': ['hsv_h', 'hsv_s', 'brightness', 'contrast'],
            'geometric': ['degrees', 'translate', 'scale'],
            'color': ['hsv_h', 'hsv_s', 'brightness', 'contrast']
        },
        
        # Preset configurations untuk common use cases
        'presets': {
            'conservative': {
                'fliplr': 0.3, 'degrees': 5, 'translate': 0.05, 'scale': 0.05,
                'hsv_h': 0.01, 'hsv_s': 0.5, 'brightness': 0.1, 'contrast': 0.1
            },
            'moderate': {
                'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2
            },
            'aggressive': {
                'fliplr': 0.7, 'degrees': 20, 'translate': 0.2, 'scale': 0.2,
                'hsv_h': 0.03, 'hsv_s': 0.9, 'brightness': 0.3, 'contrast': 0.3
            }
        }
    }