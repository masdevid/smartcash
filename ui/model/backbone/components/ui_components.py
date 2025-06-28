"""
File: smartcash/ui/model/backbone/components/ui_components.py
Deskripsi: UI components untuk backbone model configuration
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.model.backbone.components.model_form import create_model_form
from smartcash.ui.model.backbone.components.config_summary import create_config_summary
from smartcash.ui.components import (
    create_action_buttons, create_save_reset_buttons,
    create_confirmation_area
)
from smartcash.ui.components.action_section import create_action_section

def create_backbone_child_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create child components untuk backbone configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of child components
    """
    config = config or {}
    child_components = {}
    
    # === MODEL CONFIGURATION SECTIONS ===
    
    # Model form (left column)
    model_form = create_model_form(config)
    child_components['model_form'] = model_form
    
    # Config summary (right column)
    config_summary = create_config_summary(config)
    child_components['config_summary'] = config_summary
    
    # Two-column layout dengan grid flexbox
    model_config_section = widgets.HBox([
        model_form,
        config_summary
    ], layout=widgets.Layout(
        display='flex',
        gap='20px',
        width='100%',
        align_items='stretch'
    ))
    
    # Store the main section without adding it to the components yet
    # to prevent duplicate display
    child_components['model_config_section'] = model_config_section
    
    # === CONFIG SECTION ===
    
    # Save/Reset buttons
    save_reset_components = create_save_reset_buttons(
        save_label="Simpan Konfigurasi",
        reset_label="Reset", 
        with_sync_info=True
    )
    child_components['save_reset_buttons'] = save_reset_components
    
    # Config section dengan save/reset buttons
    config_section = widgets.VBox([
        widgets.Box([save_reset_components['container']], 
            layout=widgets.Layout(
                display='flex', 
                justify_content='flex-end', 
                width='100%'
            ))
    ], layout=widgets.Layout(margin='8px 0'))
    child_components['config_section'] = config_section
    
    # === ACTION SECTION ===
    
    # Action buttons
    action_components = create_action_buttons(
        primary_button={
            "label": "🏗️ Build Model",
            "style": "success",
            "width": "160px"
        },
        secondary_buttons=[
            {
                "label": "📊 Validate Config",
                "style": "info",
                "width": "150px"
            },
            {
                "label": "🔍 Model Info",
                "style": "warning",
                "width": "120px"
            }
        ]
    )
    child_components['action_components'] = action_components
    
    # Confirmation area
    confirmation_area = create_confirmation_area(ui_components=child_components)
    child_components['confirmation_area'] = confirmation_area
    
    # Create action section
    action_section = create_action_section(
        action_buttons=action_components,
        confirmation_area=confirmation_area,
        title="🚀 Model Operations",
        status_label="📋 Status:",
        show_status=True
    )
    child_components['action_section'] = action_section
    
    # === EXTRACT INDIVIDUAL COMPONENTS ===
    
    # Extract form widgets dari model_form
    if hasattr(model_form, 'backbone_dropdown'):
        child_components['backbone_dropdown'] = model_form.backbone_dropdown
    if hasattr(model_form, 'detection_layers_select'):
        child_components['detection_layers_select'] = model_form.detection_layers_select
    if hasattr(model_form, 'layer_mode_dropdown'):
        child_components['layer_mode_dropdown'] = model_form.layer_mode_dropdown
    if hasattr(model_form, 'feature_optimization_checkbox'):
        child_components['feature_optimization_checkbox'] = model_form.feature_optimization_checkbox
    if hasattr(model_form, 'mixed_precision_checkbox'):
        child_components['mixed_precision_checkbox'] = model_form.mixed_precision_checkbox
    
    # Extract buttons
    child_components['build_btn'] = action_components.get('primary')
    child_components['validate_btn'] = action_components.get('secondary_0')
    child_components['info_btn'] = action_components.get('secondary_1')
    child_components['save_button'] = save_reset_components.get('save_button')
    child_components['reset_button'] = save_reset_components.get('reset_button')
    
    return child_components

def get_layout_sections(child_components: Dict[str, Any]) -> list:
    """Get ordered layout sections untuk main container
    
    Args:
        child_components: Dictionary of child components
        
    Returns:
        List of widgets in display order
    """
    sections = []
    
    # Add sections in order, but only if they're not already in a parent container
    if 'model_config_section' in child_components and not hasattr(child_components['model_config_section'], 'parent'):
        sections.append(child_components['model_config_section'])
    
    if 'config_section' in child_components and not hasattr(child_components['config_section'], 'parent'):
        sections.append(child_components['config_section'])
    
    if 'action_section' in child_components and not hasattr(child_components['action_section'], 'parent'):
        sections.append(child_components['action_section'])
    
    return sections