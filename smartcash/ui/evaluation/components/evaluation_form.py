"""
File: smartcash/ui/evaluation/components/evaluation_form.py
Deskripsi: Form components untuk model evaluation dengan checkpoint selection dan test configuration
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.action_buttons import create_action_buttons

def create_evaluation_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat form components untuk evaluation dengan one-liner style"""
    
    checkpoint_config = config.get('checkpoint', {})
    test_config = config.get('test_data', {})
    eval_config = config.get('evaluation', {})
    
    # One-liner component creation
    components = {
        # Checkpoint Selection
        'auto_select_checkbox': widgets.Checkbox(
            value=checkpoint_config.get('auto_select_best', True),
            description='Auto pilih checkpoint terbaik',
            style={'description_width': '200px'}
        ),
        
        'checkpoint_path_text': widgets.Text(
            value=checkpoint_config.get('custom_checkpoint_path', ''),
            placeholder='Path ke checkpoint custom (opsional)',
            description='Custom checkpoint:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'validation_metrics_select': widgets.SelectMultiple(
            options=['mAP@0.5', 'mAP@0.5:0.95', 'precision', 'recall', 'f1-score'],
            value=checkpoint_config.get('validation_metrics', ['mAP@0.5', 'mAP@0.5:0.95']),
            description='Metrik validasi:',
            rows=3,
            style={'description_width': '120px'},
            layout=widgets.Layout(width='100%')
        ),
        
        # Test Data Configuration
        'test_folder_text': widgets.Text(
            value=test_config.get('test_folder', 'data/test'),
            description='Folder test data:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'apply_augmentation_checkbox': widgets.Checkbox(
            value=test_config.get('apply_augmentation', True),
            description='Terapkan augmentasi untuk testing',
            style={'description_width': '200px'}
        ),
        
        'batch_size_slider': widgets.IntSlider(
            value=test_config.get('batch_size', 16),
            min=1, max=64, step=1,
            description='Batch size:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'image_size_dropdown': widgets.Dropdown(
            options=[320, 416, 512, 640],
            value=test_config.get('image_size', 416),
            description='Image size:',
            style={'description_width': '100px'}
        ),
        
        'confidence_slider': widgets.FloatSlider(
            value=test_config.get('confidence_threshold', 0.25),
            min=0.1, max=1.0, step=0.05,
            description='Confidence:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'iou_slider': widgets.FloatSlider(
            value=test_config.get('iou_threshold', 0.45),
            min=0.1, max=1.0, step=0.05,
            description='IoU threshold:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        # Evaluation Options
        'save_predictions_checkbox': widgets.Checkbox(
            value=eval_config.get('save_predictions', True),
            description='Simpan hasil prediksi',
            style={'description_width': '150px'}
        ),
        
        'save_metrics_checkbox': widgets.Checkbox(
            value=eval_config.get('save_metrics', True),
            description='Simpan metrik evaluasi',
            style={'description_width': '150px'}
        ),
        
        'confusion_matrix_checkbox': widgets.Checkbox(
            value=eval_config.get('generate_confusion_matrix', True),
            description='Generate confusion matrix',
            style={'description_width': '180px'}
        ),
        
        'visualize_results_checkbox': widgets.Checkbox(
            value=eval_config.get('visualize_results', True),
            description='Visualisasi hasil prediksi',
            style={'description_width': '160px'}
        )
    }
    
    # Action buttons dengan one-liner
    action_buttons = create_action_buttons(
        primary_label="🚀 Evaluasi Model",
        primary_icon="evaluate",
        secondary_buttons=[("🔍 Cek Checkpoint", "search", "info"), ("🧹 Reset Config", "reset", "warning")],
        cleanup_enabled=False,
        primary_style='success'
    )
    
    # Save reset buttons
    save_reset_buttons = create_save_reset_buttons(
        save_tooltip="Simpan konfigurasi evaluasi",
        reset_tooltip="Reset ke konfigurasi default",
        with_sync_info=True,
        sync_message="Konfigurasi evaluasi akan disimpan untuk sesi berikutnya."
    )
    
    # Merge dengan action buttons
    components.update({
        **action_buttons,
        **save_reset_buttons,
        'evaluate_button': action_buttons['download_button'],  # Alias untuk consistency
        'check_button': action_buttons['check_button']
    })
    
    return components

def create_metrics_display() -> Dict[str, Any]:
    """Buat components untuk display metrics dengan one-liner style"""
    return {
        'metrics_table': widgets.HTML(
            value="<div style='padding: 10px; text-align: center; color: #666;'>📊 Metrik evaluasi akan ditampilkan setelah proses evaluasi selesai</div>",
            layout=widgets.Layout(width='100%', min_height='300px', border='1px solid #ddd', padding='10px')
        ),
        
        'confusion_matrix_output': widgets.Output(
            layout=widgets.Layout(width='100%', max_height='400px', overflow='auto')
        ),
        
        'predictions_output': widgets.Output(
            layout=widgets.Layout(width='100%', max_height='300px', overflow='auto')
        )
    }