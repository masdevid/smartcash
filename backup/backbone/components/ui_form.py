"""
File: smartcash/ui/backbone/components/ui_form.py
Deskripsi: Form components untuk backbone configuration dengan responsive checkbox layout dan status panel yang fixed
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.status_panel import create_status_panel

def create_backbone_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form components untuk backbone configuration dengan responsive checkbox layout"""
    from ..handlers.defaults import get_backbone_options, get_model_type_options
    
    # Extract config values
    backbones = config.get('backbones', {})
    model_types = config.get('model_types', {})
    feature_adapter = config.get('feature_adapter', {})
    selected_backbone = config.get('selected_backbone', 'efficientnet_b4')
    selected_model_type = config.get('selected_model_type', 'efficient_optimized')
    
    # One-liner widget creation dengan responsive layout
    components = {
        'backbone_dropdown': widgets.Dropdown(
            options=get_backbone_options(),
            value=selected_backbone,
            description='Backbone:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='100%', margin='5px 0')
        ),
        'model_type_dropdown': widgets.Dropdown(
            options=get_model_type_options(),
            value=selected_model_type,
            description='Model Type:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='100%', margin='5px 0')
        ),
        
        # Fixed checkbox layout untuk prevent overflow
        'use_attention_checkbox': widgets.Checkbox(
            value=feature_adapter.get('channel_attention', True),
            description='FeatureAdapter',
            style={'description_width': '140px'},
            layout=widgets.Layout(width='100%', margin='3px 0')
        ),
        'use_residual_checkbox': widgets.Checkbox(
            value=feature_adapter.get('use_residual', False),
            description='ResidualAdapter',
            style={'description_width': '140px'},
            layout=widgets.Layout(width='100%', margin='3px 0')
        ),
        'use_ciou_checkbox': widgets.Checkbox(
            value=model_types.get(selected_model_type, {}).get('use_ciou', False),
            description='CIoU Loss',
            style={'description_width': '140px'},
            layout=widgets.Layout(width='100%', margin='3px 0')
        )
    }
    
    # Add shared save/reset buttons tanpa sync message
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="💾 Simpan konfigurasi backbone",
        reset_tooltip="🔄 Reset ke konfigurasi default",
        with_sync_info=False
    )
    
    # Add status panel dengan single icon
    backbone_info = backbones.get(selected_backbone, {}).get('description', 'Tidak ada deskripsi')
    status_message = f"Backbone {selected_backbone} siap digunakan"
    status_panel = create_status_panel(status_message, "info")
    
    # Merge all components
    components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_container': save_reset_buttons['container'],
        'status_panel': status_panel
    })
    
    return components