"""
File: smartcash/ui/dataset/augmentation/components/advanced_options.py
Deskripsi: Komponen UI untuk opsi lanjutan augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, List, Optional

def create_advanced_options(config: Dict[str, Any] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi lanjutan augmentasi dataset.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi lanjutan
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.common.config.manager import get_config_manager
    
    # Dapatkan konfigurasi augmentasi
    config_manager = get_config_manager()
    aug_config = config_manager.get_module_config('augmentation') or {}
    
    # Pastikan struktur konfigurasi yang benar
    if not isinstance(aug_config, dict):
        aug_config = {}
    
    # Dapatkan konfigurasi augmentasi, pastikan ada struktur yang benar
    augmentation_config = aug_config.get('augmentation', {})
    if not isinstance(augmentation_config, dict):
        augmentation_config = {}
    
    # Parameter posisi
    position_params = augmentation_config.get('position', {}) 
    if not isinstance(position_params, dict):
        position_params = {}
    
    fliplr = widgets.FloatSlider(
        value=position_params.get('fliplr', 0.5),
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
        value=position_params.get('degrees', 15),
        min=0,
        max=45,
        step=5,
        description='Rotasi (°):',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    translate = widgets.FloatSlider(
        value=position_params.get('translate', 0.15),
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
        value=position_params.get('scale', 0.15),
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
    
    shear_max = widgets.IntSlider(
        value=position_params.get('shear_max', 10),
        min=0,
        max=30,
        step=5,
        description='Shear Max (°):',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d',
        layout=widgets.Layout(width='95%')
    )
    
    # Parameter pencahayaan
    lighting_params = augmentation_config.get('lighting', {})
    if not isinstance(lighting_params, dict):
        lighting_params = {}
    
    hsv_h = widgets.FloatSlider(
        value=lighting_params.get('hsv_h', 0.025),
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
        value=lighting_params.get('hsv_s', 0.7),
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
    
    hsv_v = widgets.FloatSlider(
        value=lighting_params.get('hsv_v', 0.4),
        min=0.0,
        max=1.0,
        step=0.1,
        description='HSV Value:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    # Contrast range
    contrast_min = widgets.FloatSlider(
        value=lighting_params.get('contrast', [0.7, 1.3])[0],
        min=0.5,
        max=1.0,
        step=0.1,
        description='Contrast Min:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    contrast_max = widgets.FloatSlider(
        value=lighting_params.get('contrast', [0.7, 1.3])[1],
        min=1.0,
        max=1.5,
        step=0.1,
        description='Contrast Max:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    # Brightness range
    brightness_min = widgets.FloatSlider(
        value=lighting_params.get('brightness', [0.7, 1.3])[0],
        min=0.5,
        max=1.0,
        step=0.1,
        description='Brightness Min:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    brightness_max = widgets.FloatSlider(
        value=lighting_params.get('brightness', [0.7, 1.3])[1],
        min=1.0,
        max=1.5,
        step=0.1,
        description='Brightness Max:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    blur = widgets.FloatSlider(
        value=lighting_params.get('blur', 0.2),
        min=0.0,
        max=1.0,
        step=0.1,
        description='Blur:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout=widgets.Layout(width='95%')
    )
    
    noise = widgets.FloatSlider(
        value=lighting_params.get('noise', 0.1),
        min=0.0,
        max=0.5,
        step=0.05,
        description='Noise:',
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
        layout=widgets.Layout(width='95%')
    )
    
    # Proses bounding boxes
    process_bboxes = widgets.Checkbox(
        value=augmentation_config.get('process_bboxes', True),
        description='Proses Bounding Boxes',
        indent=False,
        layout=widgets.Layout(width='auto')
    )
    
    # Layout parameter posisi
    position_box = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['position']} Parameter Posisi</h5>"),
        fliplr,
        degrees,
        translate,
        scale,
        shear_max
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    # Layout parameter pencahayaan
    lighting_box = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['lighting']} Parameter Pencahayaan</h5>"),
        hsv_h,
        hsv_s,
        hsv_v,
        widgets.HBox([contrast_min, contrast_max], layout=widgets.Layout(width='95%')),
        widgets.HBox([brightness_min, brightness_max], layout=widgets.Layout(width='95%')),
        blur,
        noise
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    # Layout parameter tambahan
    additional_box = widgets.VBox([
        widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0;'>{ICONS['settings']} Parameter Tambahan</h5>"),
        process_bboxes
    ], layout=widgets.Layout(padding='10px', border='1px solid #ddd', width='100%'))
    
    # Tab untuk parameter posisi dan pencahayaan
    tabs = widgets.Tab(children=[position_box, lighting_box, additional_box])
    tabs.set_title(0, "Posisi")
    tabs.set_title(1, "Pencahayaan")
    tabs.set_title(2, "Tambahan")
    
    # Container utama
    container = widgets.VBox([
        tabs
    ], layout=widgets.Layout(margin='10px 0'))
    
    return container
