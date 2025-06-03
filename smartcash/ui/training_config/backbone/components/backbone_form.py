"""
File: smartcash/ui/training_config/backbone/components/backbone_form.py
Deskripsi: Form components untuk backbone configuration dengan reusable widgets
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.status_panel import create_status_panel

def create_backbone_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create form components untuk backbone configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary berisi form components
    """
    model_config = config.get('model', {})
    
    # One-liner widget creation
    components = {
        'backbone_dropdown': widgets.Dropdown(
            options=[('EfficientNet-B4', 'efficientnet_b4'), ('CSPDarknet-S', 'cspdarknet_s')], 
            value=model_config.get('backbone', 'efficientnet_b4'),
            description='Backbone:', 
            style={'description_width': '120px'}
        ),
        'model_type_dropdown': widgets.Dropdown(
            options=[
                ('EfficientNet Basic', 'efficient_basic'), 
                ('EfficientNet Optimized', 'efficient_optimized'),
                ('EfficientNet Advanced', 'efficient_advanced'),
                ('YOLOv5s', 'yolov5s')
            ], 
            value=model_config.get('model_type', 'efficient_optimized'),  # Default ke optimized
            description='Model Type:', 
            style={'description_width': '120px'}
        ),
        'use_attention_checkbox': widgets.Checkbox(
            value=model_config.get('use_attention', True), 
            description='FeatureAdapter (Attention)',
            style={'description_width': '200px'}
        ),
        'use_residual_checkbox': widgets.Checkbox(
            value=model_config.get('use_residual', True), 
            description='ResidualAdapter (Residual)',
            style={'description_width': '200px'}
        ),
        'use_ciou_checkbox': widgets.Checkbox(
            value=model_config.get('use_ciou', False), 
            description='CIoU Loss',
            style={'description_width': '200px'}
        )
    }
    
    # Add save/reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi backbone",
        reset_tooltip="Reset ke konfigurasi default",
        with_sync_info=True,
        sync_message="Konfigurasi akan disinkronkan dengan Google Drive."
    )
    
    # Add status panel
    status_panel = create_status_panel("Konfigurasi backbone siap", "info")
    
    # Merge components
    components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_container': save_reset_buttons['container'],
        'status_panel': status_panel
    })
    
    return components