"""
File: smartcash/ui/strategy/components/ui_layout.py
Deskripsi: Layout responsive dengan CSS Grid tanpa horizontal overflow
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.strategy.components.ui_form import create_config_summary_card


def create_strategy_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create responsive layout dengan CSS Grid untuk prevent overflow"""
    
    # Grid styling untuk responsive layout - one-liner grid
    grid_style = "display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 8px; width: 100%; max-width: 100%;"
    section_style = "padding: 12px; margin: 6px 0; border-radius: 8px; overflow: hidden; border: 2px solid transparent;"
    
    # Validation section dengan CSS Grid
    validation_grid = widgets.HTML(f"""
    <div style="{section_style} background: linear-gradient(white, white) padding-box, linear-gradient(45deg, #4ecdc4, #44b8b5) border-box;">
        <h4 style="margin: 8px 0; color: #333; font-size: 14px;">✅ Strategi Validasi</h4>
        <div style="{grid_style}">
            <div id="val-col-1"></div>
            <div id="val-col-2"></div>
        </div>
    </div>
    """)
    
    validation_col1 = widgets.VBox([
        form_components['val_frequency_slider'],
        form_components['iou_thres_slider']
    ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    
    validation_col2 = widgets.VBox([
        form_components['conf_thres_slider'],
        form_components['max_detections_slider']
    ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    
    validation_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>✅ Strategi Validasi</h4>"),
        widgets.GridBox([validation_col1, validation_col2], 
                       layout=widgets.Layout(
                           width='100%', 
                           max_width='100%',
                           grid_template_columns='repeat(auto-fit, minmax(200px, 1fr))',
                           grid_gap='8px',
                           overflow='hidden'
                       ))
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden',
        border='2px solid transparent', width='100%', max_width='100%',
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #4ecdc4, #44b8b5) border-box'
    ))
    
    # Utils section dengan GridBox
    utils_col1 = widgets.VBox([
        form_components['experiment_name_text'],
        form_components['checkpoint_dir_text'],
        form_components['log_metrics_slider']
    ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    
    utils_col2 = widgets.VBox([
        form_components['visualize_batch_slider'],
        form_components['gradient_clipping_slider'],
        form_components['layer_mode_dropdown']
    ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    
    utils_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>🔧 Utilitas Training</h4>"),
        widgets.GridBox([utils_col1, utils_col2], 
                       layout=widgets.Layout(
                           width='100%', 
                           max_width='100%',
                           grid_template_columns='repeat(auto-fit, minmax(200px, 1fr))',
                           grid_gap='8px',
                           overflow='hidden'
                       )),
        form_components['tensorboard_checkbox']
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden',
        border='2px solid transparent', width='100%', max_width='100%',
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #ff6b6b, #ee5a5a) border-box'
    ))
    
    # Multi-scale section dengan single column
    multiscale_col = widgets.VBox([
        form_components['multi_scale_checkbox'],
        widgets.GridBox([
            form_components['img_size_min_slider'],
            form_components['img_size_max_slider']
        ], layout=widgets.Layout(
            width='100%', 
            max_width='100%',
            grid_template_columns='repeat(auto-fit, minmax(150px, 1fr))',
            grid_gap='8px',
            overflow='hidden'
        ))
    ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))
    
    multiscale_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 8px 0; color: #333; font-size: 14px;'>🔄 Multi-scale Training</h4>"),
        multiscale_col
    ], layout=widgets.Layout(
        padding='12px', margin='6px 0', border_radius='8px', overflow='hidden',
        border='2px solid transparent', width='100%', max_width='100%',
        background='linear-gradient(white, white) padding-box, linear-gradient(45deg, #45b7d1, #3aa3d0) border-box'
    ))
    
    # Content area dengan vertical stack
    content_area = widgets.VBox([
        validation_section,
        utils_section,
        multiscale_section
    ], layout=widgets.Layout(
        width='100%', 
        max_width='100%',
        overflow='hidden'
    ))
    
    # Header dan summary
    header = create_header(
        title="Konfigurasi Strategi Training",
        description="Pengaturan strategi training (hyperparameters tersedia di modul terpisah)",
        icon=ICONS.get('training', '🏋️')
    )
    
    summary_card = widgets.HTML(
        value="<div style='padding: 10px; background: #f8f9fa; border-radius: 4px; margin: 8px 0;'>📊 Loading konfigurasi...</div>",
        layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden')
    )
    
    # Main container dengan overflow control
    main_container = widgets.VBox([
        header,
        summary_card,
        content_area,
        form_components['button_container']
    ], layout=widgets.Layout(
        width='100%', 
        max_width='100%', 
        padding='8px', 
        overflow='hidden'
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
    """Update summary card dengan overflow protection"""
    if 'summary_card' in ui_components:
        timestamp = last_saved or __import__('datetime').datetime.now().strftime("%H:%M:%S")
        summary_widget = create_config_summary_card(config, timestamp)
        ui_components['summary_card'].value = summary_widget.value