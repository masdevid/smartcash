# File: smartcash/ui/hyperparameters/components/ui_layout.py
# Deskripsi: Layout UI responsive dengan flex dan color-coded groups

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.components import create_save_reset_buttons
from smartcash.ui.utils.responsive_styling import create_group_container, apply_mobile_breakpoints

logger = get_logger(__name__)


def create_hyperparameters_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Buat layout responsive dengan flex dan color-coded groups 🎨"""
    
    # Training Parameters Group - Green theme
    training_group = create_group_container(
        "🏋️ Training Parameters",
        "#4CAF50",
        "linear-gradient(135deg, #f8fff8 0%, #e8f5e8 100%)",
        [
            widgets.HBox([
                form_components['epochs'],
                form_components['batch_size']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            )),
            widgets.HBox([
                form_components['learning_rate'],
                form_components['image_size']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            )),
            form_components['workers']
        ]
    )
    
    # Optimizer Settings Group - Blue theme
    optimizer_group = create_group_container(
        "⚙️ Optimizer Settings",
        "#2196F3", 
        "linear-gradient(135deg, #f0f8ff 0%, #e3f2fd 100%)",
        [
            form_components['optimizer_type'],
            widgets.HBox([
                form_components['weight_decay'],
                form_components['momentum']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            ))
        ]
    )
    
    # Scheduler Configuration Group - Orange theme
    scheduler_group = create_group_container(
        "📈 Learning Rate Scheduler",
        "#FF9800",
        "linear-gradient(135deg, #fff8f0 0%, #fff3e0 100%)",
        [
            form_components['scheduler_type'],
            widgets.HBox([
                form_components['warmup_epochs'],
                form_components['min_lr']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            ))
        ]
    )
    
    # Loss Configuration Group - Purple theme
    loss_group = create_group_container(
        "🎯 Loss Configuration",
        "#9C27B0",
        "linear-gradient(135deg, #faf4ff 0%, #f3e5f5 100%)",
        [
            widgets.HBox([
                form_components['box_loss_gain'],
                form_components['cls_loss_gain']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            )),
            form_components['obj_loss_gain']
        ]
    )
    
    # Control Parameters Group - Red theme
    control_group = create_group_container(
        "⏹️ Early Stopping & Checkpoint",
        "#F44336",
        "linear-gradient(135deg, #fff5f5 0%, #ffebee 100%)",
        [
            widgets.HBox([
                form_components['early_stopping_enabled'],
                form_components['patience']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            )),
            widgets.HBox([
                form_components['save_best'],
                form_components['save_interval']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            ))
        ]
    )
    
    # Model Inference Group - Blue-grey theme
    inference_group = create_group_container(
        "🔍 Model Inference",
        "#607D8B",
        "linear-gradient(135deg, #f8f9fa 0%, #eceff1 100%)",
        [
            widgets.HBox([
                form_components['conf_thres'],
                form_components['iou_thres']
            ], layout=widgets.Layout(
                display='flex', 
                flex_flow='row wrap', 
                justify_content='space-between'
            )),
            form_components['max_det']
        ]
    )
    
    # Save & Reset buttons
    save_reset_buttons = create_save_reset_buttons()
    
    # Responsive CSS
    responsive_css = apply_mobile_breakpoints()
    
    # Header HTML
    header_html = widgets.HTML(
        value="<div class='hyperparams-container'><h2 style='text-align: center; color: #333; margin-bottom: 30px; font-size: 24px;'>🎛️ Hyperparameters Configuration</h2></div>"
    )
    
    # Two-column layouts
    medium_groups_hbox = widgets.HBox([
        optimizer_group,
        scheduler_group
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='flex-start',
        gap='15px'
    ))
    
    smaller_groups_hbox = widgets.HBox([
        loss_group,
        control_group
    ], layout=widgets.Layout(
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='flex-start',
        gap='15px'
    ))
    
    # Centered buttons
    buttons_hbox = widgets.HBox([save_reset_buttons], layout=widgets.Layout(
        justify_content='center',
        margin='30px 0 20px 0'
    ))
    
    # Main content VBox
    content_vbox = widgets.VBox([
        training_group,
        medium_groups_hbox,
        smaller_groups_hbox,
        inference_group,
        buttons_hbox
    ], layout=widgets.Layout(
        display='flex',
        flex_direction='column',
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='0 15px'
    ))
    
    # Main responsive layout
    main_container = widgets.VBox([
        responsive_css,
        header_html,
        content_vbox
    ], layout=widgets.Layout(
        display='flex',
        flex_direction='column',
        width='100%',
        min_height='100vh'
    ))
    
    logger.info("✅ Layout hyperparameters responsive dengan color-coded groups berhasil dibuat")
    
    return {
        'main_layout': main_container,
        'training_group': training_group,
        'optimizer_group': optimizer_group,
        'scheduler_group': scheduler_group,
        'loss_group': loss_group,
        'control_group': control_group,
        'inference_group': inference_group,
        'save_button': save_reset_buttons.children[0],
        'reset_button': save_reset_buttons.children[1]
    }