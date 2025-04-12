"""
File: smartcash/ui/dataset/augmentation/components/augmentation_options.py
Deskripsi: Komponen opsi augmentasi dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_augmentation_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi augmentasi.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox dengan opsi augmentasi
    """
    # Default values dari config jika tersedia
    aug_config = config.get('augmentation', {}) if config else {}
    
    # Nilai default
    default_types = aug_config.get('types', ['combined'])
    default_variations = aug_config.get('num_variations', 2)
    default_prefix = aug_config.get('output_prefix', 'aug')
    default_process_bboxes = aug_config.get('process_bboxes', True)
    default_validate = aug_config.get('validate_results', True)
    default_workers = aug_config.get('num_workers', 4)
    default_balance = aug_config.get('target_balance', True)
    
    # Map config types ke UI options
    type_map = {
        'combined': 'Combined (Recommended)', 
        'position': 'Position Variations', 
        'lighting': 'Lighting Variations', 
        'extreme_rotation': 'Extreme Rotation'
    }
    ui_types = [type_map.get(t, 'Combined (Recommended)') for t in default_types if t in type_map]
    if not ui_types:
        ui_types = ['Combined (Recommended)']
    
    # Buat komponen dengan nilai dari config
    aug_options = widgets.VBox([
        widgets.SelectMultiple(
            options=['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation'],
            value=ui_types,
            description='Types:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', height='100px')
        ),
        widgets.BoundedIntText(
            value=default_variations,
            min=1,
            max=10,
            description='Variations:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Text(
            value=default_prefix,
            description='Prefix:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=default_process_bboxes,
            description='Process bboxes',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=default_validate,
            description='Validate results',
            style={'description_width': 'initial'}
        ),
        widgets.IntSlider(
            value=default_workers,
            min=1,
            max=16,
            step=1,
            description='Workers:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=default_balance,
            description='Balance classes',
            style={'description_width': 'initial'}
        )
    ])
    
    return aug_options