"""
File: smartcash/ui/backbone/components/ui_form.py
Deskripsi: Form components untuk backbone configuration menggunakan reusable UI components
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.status_panel import create_status_panel

def create_backbone_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form components untuk backbone configuration menggunakan backbone_config.yaml"""
    from ..handlers.defaults import get_backbone_options, get_model_type_options
    
    # Extract config values
    backbones = config.get('backbones', {})
    model_types = config.get('model_types', {})
    feature_adapter = config.get('feature_adapter', {})
    selected_backbone = config.get('selected_backbone', 'efficientnet_b4')
    selected_model_type = config.get('selected_model_type', 'efficient_optimized')
    
    # One-liner widget creation dengan backbone_config.yaml values
    components = {
        'backbone_dropdown': widgets.Dropdown(
            options=get_backbone_options(),
            value=selected_backbone,
            description='Backbone:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='100%')
        ),
        'model_type_dropdown': widgets.Dropdown(
            options=get_model_type_options(),
            value=selected_model_type,
            description='Model Type:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='100%')
        ),
        'use_attention_checkbox': widgets.Checkbox(
            value=feature_adapter.get('channel_attention', True),
            description='FeatureAdapter (Channel Attention)',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='100%')
        ),
        'use_residual_checkbox': widgets.Checkbox(
            value=feature_adapter.get('use_residual', False),
            description='ResidualAdapter (Skip Connections)',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='100%')
        ),
        'use_ciou_checkbox': widgets.Checkbox(
            value=model_types.get(selected_model_type, {}).get('use_ciou', False),
            description='CIoU Loss (Improved Localization)',
            style={'description_width': '250px'},
            layout=widgets.Layout(width='100%')
        )
    }
    
    # Add shared save/reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="💾 Simpan konfigurasi backbone ke backbone_config.yaml",
        reset_tooltip="🔄 Reset ke konfigurasi default",
        with_sync_info=True,
        sync_message="✅ Konfigurasi akan disinkronkan dengan Google Drive."
    )
    
    # Add status panel dengan informasi backbone
    backbone_info = backbones.get(selected_backbone, {}).get('description', 'No description')
    model_info = model_types.get(selected_model_type, {}).get('description', 'No description')
    status_message = f"🧠 {backbone_info} | 🔧 {model_info}"
    status_panel = create_status_panel(status_message, "info")
    
    # Merge all components
    components.update({
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        'save_reset_container': save_reset_buttons['container'],
        'status_panel': status_panel
    })
    
    return components