"""
File: smartcash/ui/strategy/components/ui_layout.py
Deskripsi: Layout compact untuk strategy config dengan grid dan flexbox yang responsive
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.strategy.components.ui_form import create_config_summary_card


def create_strategy_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create compact layout fokus pada strategy (bukan hyperparameters)"""
    
    # Validation strategy section
    validation_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 5px 0; color: #333;'>✅ Strategi Validasi</h4>"),
        widgets.HBox([
            widgets.VBox([
                form_components['val_frequency_slider'],
                form_components['iou_thres_slider']
            ], layout=widgets.Layout(width='48%')),
            widgets.VBox([
                form_components['conf_thres_slider'],
                form_components['max_detections_slider']
            ], layout=widgets.Layout(width='48%'))
        ], layout=widgets.Layout(justify_content='space-between', width='100%'))
    ], layout=widgets.Layout(padding='8px', border='1px solid #e0e0e0', border_radius='4px', margin='5px 0'))
    
    # Training utilities section
    utils_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 5px 0; color: #333;'>🔧 Utilitas Training</h4>"),
        widgets.HBox([
            widgets.VBox([
                form_components['experiment_name_text'],
                form_components['checkpoint_dir_text'],
                form_components['log_metrics_slider']
            ], layout=widgets.Layout(width='48%')),
            widgets.VBox([
                form_components['visualize_batch_slider'],
                form_components['gradient_clipping_slider'],
                form_components['layer_mode_dropdown']
            ], layout=widgets.Layout(width='48%'))
        ], layout=widgets.Layout(justify_content='space-between', width='100%')),
        form_components['tensorboard_checkbox']
    ], layout=widgets.Layout(padding='8px', border='1px solid #e0e0e0', border_radius='4px', margin='5px 0'))
    
    # Multi-scale section
    multiscale_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 5px 0; color: #333;'>🔄 Multi-scale Training</h4>"),
        form_components['multi_scale_checkbox'],
        widgets.HBox([
            form_components['img_size_min_slider'],
            form_components['img_size_max_slider']
        ], layout=widgets.Layout(justify_content='space-between', width='100%'))
    ], layout=widgets.Layout(padding='8px', border='1px solid #e0e0e0', border_radius='4px', margin='5px 0'))
    
    # Create single column layout (no tabs needed for fewer components)
    content_area = widgets.VBox([
        validation_section,
        utils_section,
        multiscale_section
    ])
    
    # Create header
    header = create_header(
        title="Konfigurasi Strategi Training",
        description="Pengaturan strategi training (hyperparameters tersedia di modul terpisah)",
        icon=ICONS.get('training', '🏋️')
    )
    
    # Summary card placeholder
    summary_card = widgets.HTML(value="<div style='padding: 10px; background: #f8f9fa; border-radius: 4px; margin: 10px 0;'>📊 Loading konfigurasi...</div>")
    
    # Main container
    main_container = widgets.VBox([
        header,
        form_components['status_panel'],
        summary_card,
        content_area,
        form_components['button_container']
    ], layout=widgets.Layout(width='100%', max_width='100%', padding='10px', overflow='hidden'))
    
    return {
        'main_container': main_container,
        'save_button': form_components['save_button'],
        'reset_button': form_components['reset_button'],
        'status_panel': form_components['status_panel'],
        'summary_card': summary_card,
        'content_area': content_area,
        'header': header,
        **form_components
    }


def update_summary_card(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update summary card dengan konfigurasi terbaru"""
    if 'summary_card' in ui_components:
        summary_card_widget = create_config_summary_card(config)
        ui_components['summary_card'].value = summary_card_widget.value