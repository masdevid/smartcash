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
    """Create responsive layout dengan flexbox dan colorful borders"""
    
    # Validation section dengan responsive flexbox
    validation_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>✅ Strategi Validasi</h4>"),
        widgets.HBox([
            widgets.VBox([
                form_components['val_frequency_slider'],
                form_components['iou_thres_slider']
            ], layout=widgets.Layout(flex='1 1 auto', margin='0 4px 0 0')),
            widgets.VBox([
                form_components['conf_thres_slider'],
                form_components['max_detections_slider']
            ], layout=widgets.Layout(flex='1 1 auto', margin='0 0 0 4px'))
        ], layout=widgets.Layout(display='flex', width='100%', max_width='100%'))
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden',
        border='2px solid transparent', 
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #4ecdc4, #44b8b5) border-box'
    ))
    
    # Utils section dengan grid responsive
    utils_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>🔧 Utilitas Training</h4>"),
        widgets.HBox([
            widgets.VBox([
                form_components['experiment_name_text'],
                form_components['checkpoint_dir_text'],
                form_components['log_metrics_slider']
            ], layout=widgets.Layout(flex='1 1 auto', margin='0 4px 0 0')),
            widgets.VBox([
                form_components['visualize_batch_slider'],
                form_components['gradient_clipping_slider'],
                form_components['layer_mode_dropdown']
            ], layout=widgets.Layout(flex='1 1 auto', margin='0 0 0 4px'))
        ], layout=widgets.Layout(display='flex', width='100%', max_width='100%')),
        form_components['tensorboard_checkbox']
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden',
        border='2px solid transparent',
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #ff6b6b, #ee5a5a) border-box'
    ))
    
    # Multi-scale section
    multiscale_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>🔄 Multi-scale Training</h4>"),
        form_components['multi_scale_checkbox'],
        widgets.HBox([
            form_components['img_size_min_slider'],
            form_components['img_size_max_slider']
        ], layout=widgets.Layout(display='flex', width='100%', max_width='100%'))
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden',
        border='2px solid transparent',
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #45b7d1, #3aa3d0) border-box'
    ))
    
    # Content area dengan flexbox
    content_area = widgets.VBox([
        validation_section,
        utils_section,
        multiscale_section
    ], layout=widgets.Layout(
        display='flex', 
        flex_flow='column',
        width='100%', 
        max_width='100%',
        overflow='hidden'
    ))
    
    # Header
    header = create_header(
        title="Konfigurasi Strategi Training",
        description="Pengaturan strategi training (hyperparameters tersedia di modul terpisah)",
        icon=ICONS.get('training', '🏋️')
    )
    
    # Summary card placeholder
    summary_card = widgets.HTML(value="<div style='padding: 10px; background: #f8f9fa; border-radius: 4px; margin: 8px 0;'>📊 Loading konfigurasi...</div>")
    
    # Main container dengan overflow protection
    main_container = widgets.VBox([
        header,
        summary_card,
        content_area,
        form_components['button_container']
    ], layout=widgets.Layout(
        width='100%', 
        max_width='100%', 
        padding='8px', 
        overflow='hidden',
        display='flex',
        flex_flow='column'
    ))
    
    return {
        'main_container': main_container,
        'save_button': form_components['save_button'],
        'reset_button': form_components['reset_button'],
        'summary_card': summary_card,
        'content_area': content_area,
        'header': header,
        **form_components
    }


def update_summary_card(ui_components: Dict[str, Any], config: Dict[str, Any], last_saved: str = None) -> None:
    """Update summary card dengan timestamp"""
    if 'summary_card' in ui_components:
        import datetime
        if not last_saved:
            last_saved = datetime.datetime.now().strftime("%H:%M:%S")
        summary_card_widget = create_config_summary_card(config, last_saved)
        ui_components['summary_card'].value = summary_card_widget.value