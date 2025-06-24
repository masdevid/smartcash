"""
File: smartcash/ui/strategy/components/ui_form.py
Deskripsi: Komponen form untuk konfigurasi strategy training
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from IPython.display import display
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def create_strategy_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat form widgets untuk strategy configuration - UI only"""
    try:
        logger.info("🎛️  Membuat form strategy...")
        
        # Extract config sections dengan safe defaults
        validation = config.get('validation', {})
        training_utils = config.get('training_utils', {}) 
        multi_scale = config.get('multi_scale', {})
        
        form_widgets = {}
        
        # ===== VALIDATION SECTION =====
        form_widgets.update({
            'val_frequency_slider': widgets.IntSlider(
                value=validation.get('frequency', 1),
                min=1, max=10, step=1,
                description='🔄 Val Frequency:',
                style={'description_width': '120px'}
            ),
            
            'iou_thres_slider': widgets.FloatSlider(
                value=validation.get('iou_thres', 0.6),
                min=0.1, max=1.0, step=0.05,
                description='🎯 IoU Threshold:',
                style={'description_width': '120px'}
            ),
            
            'conf_thres_slider': widgets.FloatSlider(
                value=validation.get('conf_thres', 0.001),
                min=0.001, max=0.5, step=0.001,
                description='📊 Conf Threshold:',
                style={'description_width': '120px'}
            ),
            
            'max_detections_slider': widgets.IntSlider(
                value=validation.get('max_detections', 300),
                min=10, max=1000, step=10,
                description='🔍 Max Detections:',
                style={'description_width': '120px'}
            )
        })
        
        # ===== TRAINING UTILITIES SECTION =====
        form_widgets.update({
            'experiment_name_text': widgets.Text(
                value=training_utils.get('experiment_name', 'efficient_optimized_single'),
                description='🏷️  Nama Eksperimen:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='400px')
            ),
            
            'tensorboard_checkbox': widgets.Checkbox(
                value=training_utils.get('tensorboard', True),
                description='📈 Enable Tensorboard',
                style={'description_width': '120px'}
            ),
            
            'log_metrics_slider': widgets.IntSlider(
                value=training_utils.get('log_metrics', 10),
                min=1, max=100, step=5,
                description='📝 Log Metrics Every:',
                style={'description_width': '120px'}
            ),
            
            'visualize_batch_slider': widgets.IntSlider(
                value=training_utils.get('visualize_batch', 100),
                min=10, max=500, step=10,
                description='🖼️ Visualize Every:',
                style={'description_width': '120px'}
            ),
            
            'layer_mode_dropdown': widgets.Dropdown(
                value=training_utils.get('layer_mode', 'single'),
                options=['single', 'multilayer'],
                description='🏗️ Layer Mode:',
                style={'description_width': '120px'}
            )
        })
        
        # ===== MULTI-SCALE SECTION =====
        form_widgets.update({
            'multi_scale_checkbox': widgets.Checkbox(
                value=multi_scale.get('enabled', True),
                description='🔄 Enable Multi-scale',
                style={'description_width': '120px'}
            ),
            
            'img_size_min_slider': widgets.IntSlider(
                value=multi_scale.get('img_size_min', 320),
                min=128, max=1024, step=32,
                description='📏 Min Size:',
                style={'description_width': '120px'}
            ),
            
            'img_size_max_slider': widgets.IntSlider(
                value=multi_scale.get('img_size_max', 640),
                min=512, max=1024, step=32,
                description='📐 Max Size:',
                style={'description_width': '120px'}
            )
        })
        
        # ===== CONTROL BUTTONS =====
        form_widgets.update({
            'save_button': widgets.Button(
                description='💾 Simpan Strategy',
                button_style='success',
                tooltip='Simpan konfigurasi strategy'
            ),
            
            'reset_button': widgets.Button(
                description='🔄 Reset ke Default',
                button_style='warning',
                tooltip='Reset ke konfigurasi default'
            )
        })
        
        logger.info("✅ Strategy form berhasil dibuat dengan 15 widgets")
        return form_widgets
        
    except Exception as e:
        logger.error(f"❌ Error membuat strategy form: {str(e)}")
        raise


def create_config_summary_card(config: Dict[str, Any], last_saved: str = None) -> widgets.HTML:
    """Buat summary card gabungan untuk semua konfigurasi - UI only"""
    try:
        # Extract values dengan safe defaults
        validation = config.get('validation', {})
        training_utils = config.get('training_utils', {})
        multi_scale = config.get('multi_scale', {})
        
        timestamp = f" | 📅 {last_saved}" if last_saved else ""
        
        summary_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 8px; padding: 16px; margin: 10px 0; color: white;">
            <h4 style="margin: 0 0 12px 0; color: white;">📊 Ringkasan Konfigurasi Strategy{timestamp}</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; font-size: 13px;">
                <div>
                    <strong>🔍 Validation:</strong><br>
                    • Frequency: {validation.get('frequency', 1)}<br>
                    • IoU: {validation.get('iou_thres', 0.6)}<br>
                    • Conf: {validation.get('conf_thres', 0.001)}
                </div>
                <div>
                    <strong>🛠️ Training Utils:</strong><br>
                    • Exp: {training_utils.get('experiment_name', 'default')[:15]}...<br>
                    • Tensorboard: {'🟢' if training_utils.get('tensorboard', True) else '🔴'}<br>
                    • Layer: {training_utils.get('layer_mode', 'single')}
                </div>
                <div>
                    <strong>🔄 Multi-scale:</strong><br>
                    • Enabled: {'🟢' if multi_scale.get('enabled', True) else '🔴'}<br>
                    • Size: {multi_scale.get('img_size_min', 320)}-{multi_scale.get('img_size_max', 640)}<br>
                    • Step: 32px
                </div>
            </div>
        </div>
        """
        
        return widgets.HTML(value=summary_html)
        
    except Exception as e:
        logger.error(f"❌ Error membuat summary card: {str(e)}")
        return widgets.HTML(value=f"<div style='color: red;'>❌ Error: {str(e)}</div>")