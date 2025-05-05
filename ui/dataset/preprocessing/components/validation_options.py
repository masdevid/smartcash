"""
File: smartcash/ui/dataset/preprocessing/components/validation_options.py
Deskripsi: Komponen opsi validasi untuk preprocessing dataset
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.dataset.utils.dataset_constants import DEFAULT_INVALID_DIR

def create_validation_options(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    Buat komponen UI untuk opsi validasi dataset.
    
    Args:
        config: Konfigurasi aplikasi
        
    Returns:
        Widget VBox berisi opsi validasi
    """
    # Dapatkan nilai default dari config jika tersedia
    validate_enabled = True
    fix_issues = True
    move_invalid = True
    invalid_dir = DEFAULT_INVALID_DIR  # Default 'data/invalid'
    
    if config and 'preprocessing' in config and 'validate' in config['preprocessing']:
        validate_config = config['preprocessing']['validate']
        validate_enabled = validate_config.get('enabled', True)
        fix_issues = validate_config.get('fix_issues', True)
        move_invalid = validate_config.get('move_invalid', True)
        invalid_dir = validate_config.get('invalid_dir', DEFAULT_INVALID_DIR)
    
    # Buat komponen-komponen UI
    validate_checkbox = widgets.Checkbox(
        value=validate_enabled,
        description='Validate dataset integrity',
        style={'description_width': 'initial'}
    )
    
    fix_issues_checkbox = widgets.Checkbox(
        value=fix_issues,
        description='Fix issues automatically',
        style={'description_width': 'initial'}
    )
    
    move_invalid_checkbox = widgets.Checkbox(
        value=move_invalid,
        description='Move invalid files',
        style={'description_width': 'initial'}
    )
    
    invalid_dir_text = widgets.Text(
        value=invalid_dir,
        description='Invalid dir:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='60%')
    )
    
    # Gabungkan dalam container
    return widgets.VBox([
        validate_checkbox,
        fix_issues_checkbox,
        move_invalid_checkbox,
        invalid_dir_text
    ])