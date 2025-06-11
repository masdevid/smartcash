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
    """Create responsive layout dengan optimized overflow prevention"""
    
    # One-liner layout generators dengan overflow fix
    safe_layout = lambda **kw: widgets.Layout(width='100%', max_width='100%', overflow='hidden', **kw)
    grid_layout = lambda min_w='180px': safe_layout(grid_template_columns=f'repeat(auto-fit, minmax({min_w}, 1fr))', grid_gap='6px')
    section_layout = lambda bg: safe_layout(padding='10px 14px', margin='6px 0', border_radius='8px', border='2px solid #f9f9f9', background=bg)
    header_html = lambda title, icon: widgets.HTML(f"<h4 style='margin: 6px 0; color: #333; font-size: 14px; overflow: hidden; text-overflow: ellipsis;'>{icon} {title}</h4>")
    
    # Apply overflow fix to existing form components
    [setattr(widget.layout, 'overflow', 'hidden') and setattr(widget.layout, 'max_width', '100%') 
     for widget in form_components.values() if hasattr(widget, 'layout')]
    
    # Section builders dengan one-liner approach
    build_section = lambda title, icon, bg, widgets_grid: widgets.VBox([
        header_html(title, icon),
        widgets.GridBox(widgets_grid, layout=grid_layout())
    ], layout=section_layout(bg))
    
    # Widget column builders
    validation_cols = [
        widgets.VBox([form_components['val_frequency_slider'], form_components['iou_thres_slider']], layout=safe_layout()),
        widgets.VBox([form_components['conf_thres_slider'], form_components['max_detections_slider']], layout=safe_layout())
    ]
    
    utils_cols = [
        widgets.VBox([form_components['experiment_name_text'], form_components['checkpoint_dir_text'], form_components['log_metrics_slider']], layout=safe_layout()),
        widgets.VBox([form_components['visualize_batch_slider'], form_components['gradient_clipping_slider'], form_components['layer_mode_dropdown']], layout=safe_layout())
    ]
    
    multiscale_cols = [
        widgets.VBox([form_components['img_size_min_slider']], layout=safe_layout()),
        widgets.VBox([form_components['img_size_max_slider']], layout=safe_layout())
    ]
    
    # Build sections dengan gradient backgrounds
    validation_section = build_section('Strategi Validasi', '✅', 'linear-gradient(white, white) padding-box, linear-gradient(45deg, #4ecdc4, #44b8b5) border-box', validation_cols)
    
    utils_section = widgets.VBox([
        header_html('Utilitas Training', '🔧'),
        widgets.GridBox(utils_cols, layout=grid_layout()),
        form_components['tensorboard_checkbox']
    ], layout=section_layout('linear-gradient(white, white) padding-box, linear-gradient(45deg, #ff6b6b, #ee5a5a) border-box'))
    
    multiscale_section = widgets.VBox([
        header_html('Multi-scale Training', '🔄'),
        form_components['multi_scale_checkbox'],
        widgets.GridBox(multiscale_cols, layout=grid_layout('120px'))
    ], layout=section_layout('linear-gradient(white, white) padding-box, linear-gradient(45deg, #45b7d1, #3aa3d0) border-box'))
    
    
    # Content assembly dengan one-liner components
    content_area = widgets.VBox([validation_section, utils_section, multiscale_section], layout=safe_layout())
    header = create_header("Konfigurasi Strategi Training", "Pengaturan strategi training (hyperparameters tersedia di modul terpisah)", ICONS.get('training', '🏋️'))
    summary_card = widgets.HTML("<div style='padding: 10px; background: #f8f9fa; border-radius: 4px; margin: 8px 0; overflow: hidden;'>📊 Loading konfigurasi...</div>", layout=safe_layout())
    
    # Main container assembly
    main_container = widgets.VBox([header, summary_card, content_area, form_components['button_container']], layout=safe_layout(padding='6px'))
    
    # One-liner return dengan spread form_components
    return {'main_container': main_container, 'save_button': form_components['save_button'], 'reset_button': form_components['reset_button'], 'summary_card': summary_card, 'content_area': content_area, 'header': header, **form_components}


def update_summary_card(ui_components: Dict[str, Any], config: Dict[str, Any], last_saved: str = None) -> None:
    """Update summary card dengan overflow protection"""
    if 'summary_card' in ui_components:
        timestamp = last_saved or __import__('datetime').datetime.now().strftime("%H:%M:%S")
        summary_widget = create_config_summary_card(config, timestamp)
        ui_components['summary_card'].value = summary_widget.value