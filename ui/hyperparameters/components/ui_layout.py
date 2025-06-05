"""
File: smartcash/ui/hyperparameters/components/ui_layout.py
Deskripsi: Layout arrangement untuk hyperparameters dengan grid responsif dan tidak ada horizontal scrollbar
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.hyperparameters.utils.form_helpers import create_section_card, create_responsive_grid_layout


def create_hyperparameters_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create layout dengan section cards dan responsive grid untuk mencegah horizontal scrollbar"""
    
    # Training parameters section
    training_section = create_section_card(
        "📊 Parameter Training", [
            form_components['epochs_slider'],
            form_components['batch_size_slider'],
            form_components['learning_rate_slider'],
            form_components['image_size_slider'],
            form_components['mixed_precision_checkbox']
        ], '#2196f3'
    )
    
    # Optimizer & scheduler section
    optimizer_section = create_section_card(
        "⚙️ Optimizer & Scheduler", [
            form_components['optimizer_dropdown'],
            form_components['weight_decay_slider'],
            form_components['momentum_slider'],
            widgets.HTML("<hr style='margin: 8px 0; border: 0; border-top: 1px solid #eee;'>"),
            form_components['scheduler_dropdown'],
            form_components['warmup_epochs_slider']
        ], '#9c27b0'
    )
    
    # Advanced parameters section
    advanced_section = create_section_card(
        "🔧 Parameter Lanjutan", [
            widgets.HTML("<b style='color: #666; font-size: 12px;'>LOSS WEIGHTS</b>"),
            form_components['box_loss_gain_slider'],
            form_components['cls_loss_gain_slider'],
            form_components['obj_loss_gain_slider'],
            widgets.HTML("<hr style='margin: 8px 0; border: 0; border-top: 1px solid #eee;'>"),
            widgets.HTML("<b style='color: #666; font-size: 12px;'>TRAINING CONTROL</b>"),
            form_components['gradient_accumulation_slider'],
            form_components['gradient_clipping_slider']
        ], '#ff9800'
    )
    
    # Control & checkpoint section
    control_section = create_section_card(
        "🛑 Early Stopping & Checkpoint", [
            form_components['early_stopping_checkbox'],
            form_components['patience_slider'],
            form_components['min_delta_slider'],
            widgets.HTML("<hr style='margin: 8px 0; border: 0; border-top: 1px solid #eee;'>"),
            form_components['save_best_checkbox'],
            form_components['checkpoint_metric_dropdown']
        ], '#4caf50'
    )
    
    # Create responsive grid layout untuk cards
    cards_grid = create_responsive_grid_layout([
        training_section,
        optimizer_section,
        advanced_section,
        control_section
    ])
    
    # Header component
    header = create_header(
        title="Konfigurasi Hyperparameter",
        description="Pengaturan parameter pelatihan untuk optimasi model deteksi mata uang",
        icon=ICONS.get('settings', '⚙️')
    )
    
    # Main container dengan layout yang tidak overflow
    main_container = widgets.VBox([
        header,
        form_components['status_panel'],
        form_components['summary_cards'],
        cards_grid,
        form_components['button_container']
    ], layout=widgets.Layout(
        width='100%', max_width='100%', padding='10px',
        overflow='hidden'
    ))
    
    # Return components dengan required keys untuk ConfigCellInitializer
    return {
        'main_container': main_container,
        'save_button': form_components['save_button'],
        'reset_button': form_components['reset_button'],
        'status_panel': form_components['status_panel'],
        'summary_cards': form_components['summary_cards'],
        'header': header,
        'cards_grid': cards_grid,
        **form_components  # Include all form components
    }