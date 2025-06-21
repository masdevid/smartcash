"""
File: smartcash/ui/hyperparameters/components/ui_layout.py
Deskripsi: Layout arrangement untuk hyperparameters dengan grid responsif simplified
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.hyperparameters.utils.form_helpers import create_section_card, create_responsive_grid_layout


def create_hyperparameters_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create layout dengan section cards simplified sesuai parameter yang ada"""
    
    # Training parameters section - essentials only
    training_section = create_section_card(
        "📊 Parameter Training", [
            form_components['epochs_slider'],
            form_components['batch_size_slider'],
            form_components['learning_rate_slider'],
            form_components['image_size_slider']
        ], '#2196f3'
    )
    
    # Optimizer & scheduler section - essentials only
    optimizer_section = create_section_card(
        "⚙️ Optimizer & Scheduler", [
            form_components['optimizer_dropdown'],
            form_components['weight_decay_slider'],
            widgets.HTML("<hr style='margin: 8px 0; border: 0; border-top: 1px solid #eee;'>"),
            form_components['scheduler_dropdown'],
            form_components['warmup_epochs_slider']
        ], '#9c27b0'
    )
    
    # Loss parameters section
    loss_section = create_section_card(
        "🎯 Loss Weights", [
            form_components['box_loss_gain_slider'],
            form_components['cls_loss_gain_slider'],
            form_components['obj_loss_gain_slider']
        ], '#ff9800'
    )
    
    # Control & checkpoint section
    control_section = create_section_card(
        "🛑 Control & Checkpoint", [
            form_components['early_stopping_checkbox'],
            form_components['patience_slider'],
            widgets.HTML("<hr style='margin: 8px 0; border: 0; border-top: 1px solid #eee;'>"),
            form_components['save_best_checkbox'],
            form_components['checkpoint_metric_dropdown']
        ], '#4caf50'
    )
    
    # Create responsive grid layout dengan 2x2 grid
    params_grid = create_responsive_grid_layout([training_section, optimizer_section])
    advanced_grid = create_responsive_grid_layout([loss_section, control_section])
    
    # Summary cards section
    summary_section = widgets.VBox([
        widgets.HTML("<h6 style='margin: 8px 0; color: #495057;'>📋 Ringkasan Konfigurasi</h6>"),
        form_components['summary_cards']
    ], layout=widgets.Layout(margin='16px 0 8px 0'))
    
    # Status dan action buttons
    action_section = widgets.VBox([
        form_components['status_panel'],
        form_components['button_container']
    ], layout=widgets.Layout(margin='8px 0'))
    
    # Main container dengan proper spacing
    main_container = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; border-bottom: 2px solid #2196f3; padding-bottom: 4px;'>🎛️ Konfigurasi Hyperparameters</h4>"),
        params_grid,
        advanced_grid,
        summary_section,
        action_section
    ], layout=widgets.Layout(
        width='100%', padding='12px', border='1px solid #dee2e6',
        border_radius='8px', background_color='#fafbfc'
    ))
    
    # Add button container to the layout
    button_container = widgets.HBox([
        form_components['save_button'],
        form_components['reset_button']
    ], layout=widgets.Layout(justify_content='flex-end', margin='10px 0'))
    
    # Add button container to the main container
    main_container.children = list(main_container.children) + [button_container]
    
    # Return components untuk akses dari handler
    return {
        'main_container': main_container,
        'button_container': button_container,
        'training_section': training_section,
        'optimizer_section': optimizer_section,
        'loss_section': loss_section,
        'control_section': control_section,
        'summary_section': summary_section,
        'action_section': action_section,
        **form_components  # Spread all form components
    }