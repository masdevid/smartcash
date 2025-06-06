"""
File: smartcash/ui/strategy/components/ui_layout.py
Deskripsi: Layout responsive untuk strategy config dengan overflow fixes
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.strategy.components.ui_form import create_config_summary_card


def create_strategy_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create responsive layout dengan overflow protection"""
    
    # One-liner untuk widget wrapper dengan overflow protection
    vbox_wrapper = lambda children, margin='0 4px 0 0': widgets.VBox(children, layout=widgets.Layout(flex='1 1 auto', margin=margin, min_width='0', overflow='hidden'))
    
    # Validation section dengan overflow fixes
    validation_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>✅ Strategi Validasi</h4>"),
        widgets.HBox([
            vbox_wrapper([
                form_components['val_frequency_slider'],
                form_components['iou_thres_slider']
            ]),
            vbox_wrapper([
                form_components['conf_thres_slider'],
                form_components['max_detections_slider']
            ], '0 0 0 4px')
        ], layout=widgets.Layout(display='flex', width='100%', max_width='100%', overflow='hidden'))
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden', min_width='0',
        border='2px solid transparent', 
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #4ecdc4, #44b8b5) border-box'
    ))
    
    # Utils section dengan overflow fixes
    utils_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>🔧 Utilitas Training</h4>"),
        widgets.HBox([
            vbox_wrapper([
                form_components['experiment_name_text'],
                form_components['checkpoint_dir_text'],
                form_components['log_metrics_slider']
            ]),
            vbox_wrapper([
                form_components['visualize_batch_slider'],
                form_components['gradient_clipping_slider'],
                form_components['layer_mode_dropdown']
            ], '0 0 0 4px')
        ], layout=widgets.Layout(display='flex', width='100%', max_width='100%', overflow='hidden')),
        form_components['tensorboard_checkbox']
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden', min_width='0',
        border='2px solid transparent',
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #ff6b6b, #ee5a5a) border-box'
    ))
    
    # Multi-scale section dengan stacked layout untuk narrow screens
    multiscale_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>🔄 Multi-scale Training</h4>"),
        form_components['multi_scale_checkbox'],
        widgets.VBox([
            form_components['img_size_min_slider'],
            form_components['img_size_max_slider']
        ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden', min_width='0',
        border='2px solid transparent',
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #45b7d1, #3aa3d0) border-box'
    ))
    
    # Content area dengan full overflow protection
    content_area = widgets.VBox([
        validation_section,
        utils_section,
        multiscale_section
    ], layout=widgets.Layout(
        display='flex', 
        flex_flow='column',
        width='100%', 
        max_width='100%',
        overflow='hidden',
        min_width='0'
    ))
    
    # Header
    header = create_header(
        title="Konfigurasi Strategi Training",
        description="Pengaturan strategi training (hyperparameters tersedia di modul terpisah)",
        icon=ICONS.get('training', '🏋️')
    )
    
    # Summary card placeholder dengan overflow protection
    summary_card = widgets.HTML(
        value="<div style='padding: 10px; background: #f8f9fa; border-radius: 4px; margin: 8px 0; overflow: hidden;'>📊 Loading konfigurasi...</div>",
        layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden')
    )
    
    # Main container dengan complete overflow protection
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
        flex_flow='column',
        min_width='0'
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
    """Update summary card dengan timestamp - one-liner conditional timestamp"""
    if 'summary_card' in ui_components:
        timestamp = last_saved or __import__('datetime').datetime.now().strftime("%H:%M:%S")
        summary_widget = create_config_summary_card(config, timestamp)
        ui_components['summary_card'].value = summary_widget.value