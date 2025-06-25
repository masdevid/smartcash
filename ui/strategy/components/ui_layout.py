"""
File: smartcash/ui/strategy/components/ui_layout.py
Deskripsi: Layout strategy dengan summary card gabungan
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable
from IPython.display import display
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_strategy_layout(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Buat layout strategy dengan grid system yang aman"""

    from smartcash.ui.strategy.components.ui_form import create_config_summary_card
    
    # Fungsi helper untuk mendapatkan komponen dengan aman
    def get_component(key: str, default=None):
        widget = form_components.get(key)
        if widget is None:
            logger.warning(f"⚠️ Komponen {key} tidak ditemukan, menggunakan default")
            return default or widgets.HTML(f"<div style='color: red;'>⚠️ {key} not found</div>")
        return widget
    
    # ===== GRID LAYOUT =====
    grid = widgets.GridBox(
        layout=widgets.Layout(
            width='100%',
            grid_gap='15px',
            padding='10px',
            grid_template_columns='repeat(2, 1fr)',
            grid_template_rows='auto auto auto auto auto',
            grid_template_areas='''
                "header header"
                "summary summary"
                "validation training"
                "multiscale multiscale"
                "controls controls"
            '''
        )
    )
    
    # ===== KOMPONEN UTAMA =====
    # Header
    header = widgets.VBox([
        widgets.HTML("<h2 style='color: #8E44AD; margin: 10px 0;'>🎯 Konfigurasi Strategy Training</h2>"),
        widgets.HTML("<p style='color: #666; margin: 5px 0 15px 0;'><i>Konfigurasi strategi training yang tidak overlap dengan hyperparameters</i></p>")
    ], layout=widgets.Layout(grid_area='header'))
    
    # Summary card
    config = _extract_config_from_form(form_components)
    summary_card = create_config_summary_card(config)
    summary_card.layout.grid_area = 'summary'
    
    # ===== BAGIAN VALIDASI =====
    validation_section = widgets.VBox([
        widgets.HTML("<h3 style='color: #2C3E50; margin: 10px 0;'>🔍 Validasi</h3>"),
        get_component('val_frequency_slider'),
        get_component('iou_thres_slider'),
        get_component('conf_thres_slider'),
        get_component('max_detections_slider')
    ], layout=widgets.Layout(
        grid_area='validation',
        border='1px solid #E0E0E0',
        padding='15px',
        border_radius='8px'
    ))
    
    # ===== BAGIAN TRAINING =====
    training_section = widgets.VBox([
        widgets.HTML("<h3 style='color: #2C3E50; margin: 10px 0;'>⚙️ Training</h3>"),
        get_component('experiment_name_text'),
        get_component('tensorboard_checkbox'),
        get_component('log_metrics_slider'),
        get_component('visualize_batch_slider'),
        get_component('layer_mode_dropdown')
    ], layout=widgets.Layout(
        grid_area='training',
        border='1px solid #E0E0E0',
        padding='15px',
        border_radius='8px'
    ))
    
    # ===== BAGIAN MULTI-SCALE =====
    multiscale_section = widgets.VBox([
        widgets.HTML("<h3 style='color: #2C3E50; margin: 10px 0;'>🔄 Multi-scale Training</h3>"),
        get_component('multi_scale_checkbox'),
        widgets.HBox([
            get_component('img_size_min_slider'),
            get_component('img_size_max_slider')
        ])
    ], layout=widgets.Layout(
        grid_area='multiscale',
        border='1px solid #E0E0E0',
        padding='15px',
        border_radius='8px'
    ))
    
    # ===== TOMBOL AKSI =====
    controls_section = widgets.HBox([
        get_component('save_button'),
        get_component('reset_button')
    ], layout=widgets.Layout(
        grid_area='controls',
        justify_content='flex-end',
        margin='10px 0'
    ))
    
    # ===== GABUNGKAN SEMUA KOMPONEN =====
    grid.children = [
        header,
        summary_card,
        validation_section,
        training_section,
        multiscale_section,
        controls_section
    ]
    
    # ===== HASIL AKHIR =====
    ui_components = {
        'main_layout': grid,
        'header': header,
        'summary_card': summary_card,
        'validation_section': validation_section,
        'training_section': training_section,
        'multiscale_section': multiscale_section,
        'controls_section': controls_section
    }
    
    # Gabungkan dengan form components
    ui_components.update(form_components)
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(ui_components)
    return ui_components
   

def _extract_config_from_form(form_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dari form components untuk summary"""
    try:
        config = {
            'validation': {
                'frequency': getattr(form_components.get('val_frequency_slider'), 'value', 1),
                'iou_thres': getattr(form_components.get('iou_thres_slider'), 'value', 0.6),
                'conf_thres': getattr(form_components.get('conf_thres_slider'), 'value', 0.001),
                'max_detections': getattr(form_components.get('max_detections_slider'), 'value', 300)
            },
            'training_utils': {
                'experiment_name': getattr(form_components.get('experiment_name_text'), 'value', 'default'),
                'tensorboard': getattr(form_components.get('tensorboard_checkbox'), 'value', True),
                'log_metrics': getattr(form_components.get('log_metrics_slider'), 'value', 10),
                'visualize_batch': getattr(form_components.get('visualize_batch_slider'), 'value', 100),
                'layer_mode': getattr(form_components.get('layer_mode_dropdown'), 'value', 'single')
            },
            'multi_scale': {
                'enabled': getattr(form_components.get('multi_scale_checkbox'), 'value', False),
                'img_size_min': getattr(form_components.get('img_size_min_slider'), 'value', 320),
                'img_size_max': getattr(form_components.get('img_size_max_slider'), 'value', 640)
            }
        }
        return config
    except Exception as e:
        logger.error(f"❌ Gagal mengekstrak konfigurasi: {str(e)}")
        return {}


def update_summary_display(ui_components: Dict[str, Any]) -> None:
    """Update summary card dengan nilai terbaru"""
    try:
        from .ui_form import create_config_summary_card
        
        config = _extract_config_from_form(ui_components)
        updated_summary = create_config_summary_card(config)
        
        summary_card = ui_components.get('summary_card')
        if summary_card:
            summary_card.value = updated_summary.value
            
        logger.debug("📊 Summary card berhasil diupdate")
        
    except Exception as e:
        logger.error(f"❌ Error update summary: {str(e)}")


def setup_dynamic_summary_updates(ui_components: Dict[str, Any]) -> None:
    """Setup auto-update summary card"""
    
    def on_value_change(change):
        update_summary_display(ui_components)
    
    key_widgets = [
        'val_frequency_slider', 'iou_thres_slider', 'conf_thres_slider', 'max_detections_slider',
        'experiment_name_text', 'tensorboard_checkbox', 'layer_mode_dropdown',
        'log_metrics_slider', 'visualize_batch_slider',
        'multi_scale_checkbox', 'img_size_min_slider', 'img_size_max_slider'
    ]
    
    for widget_name in key_widgets:
        widget = ui_components.get(widget_name)
        if widget:
            widget.observe(on_value_change, names='value')
    