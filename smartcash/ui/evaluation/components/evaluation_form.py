"""
File: smartcash/ui/evaluation/components/evaluation_form.py
Deskripsi: Form components untuk model evaluation dengan checkpoint selection, test configuration, dan skenario pengujian
"""

import ipywidgets as widgets
from typing import Dict, Any, List
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components import create_save_reset_buttons, create_status_panel, create_section_title

def create_evaluation_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Membuat form evaluasi dengan komponen UI"""
    checkpoint_config = config.get('checkpoint', {})
    test_config = config.get('test_data', {})
    eval_config = config.get('evaluation', {})
    scenario_config = config.get('scenario', {})
    
    # Inisialisasi konfigurasi skenario jika belum ada
    if 'scenario' not in config:
        config['scenario'] = {
            'selected_scenario': 'scenario_1',
            'save_to_drive': True,
            'drive_path': '/content/drive/MyDrive/SmartCash/evaluation_results',
            'test_folder': '/content/drive/MyDrive/SmartCash/dataset/test'
        }
    
    # One-liner component creation
    components = {
        # Scenario Selection
        'scenario_dropdown': widgets.Dropdown(
            options=[
                ('Skenario 1: YOLOv5 Default (CSPDarknet) - Variasi Posisi', 'scenario_1'),
                ('Skenario 2: YOLOv5 Default (CSPDarknet) - Variasi Pencahayaan', 'scenario_2'),
                ('Skenario 3: YOLOv5 dengan EfficientNet-B4 - Variasi Posisi', 'scenario_3'),
                ('Skenario 4: YOLOv5 dengan EfficientNet-B4 - Variasi Pencahayaan', 'scenario_4')
            ],
            value=scenario_config.get('selected_scenario', 'scenario_1'),
            description='Skenario Pengujian:',
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'scenario_description': widgets.HTML(
            value="<div style='padding: 10px; background-color: #f8f9fa; border-left: 3px solid #4CAF50; margin: 10px 0;'>"
                  "<p><b>Skenario Pengujian:</b> Pilih skenario pengujian dari dropdown di atas</p>"
                  "<p><b>Backbone:</b> -</p>"
                  "<p><b>Tipe Augmentasi:</b> -</p>"
                  "<p><small>Hasil prediksi akan disimpan ke drive sesuai dengan nama skenario</small></p>"
                  "</div>",
            layout=widgets.Layout(width='100%')
        ),
        
        # Checkpoint Selection
        'auto_select_checkbox': widgets.Checkbox(
            value=checkpoint_config.get('auto_select_best', True),
            description='Auto pilih checkpoint terbaik',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
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
            value=scenario_config.get('test_folder', '/content/drive/MyDrive/SmartCash/dataset/test'),
            placeholder='Path ke folder test data',
            description='Test folder:',
            description_tooltip='Folder yang berisi gambar test untuk evaluasi model',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'apply_augmentation_checkbox': widgets.Checkbox(
            value=test_config.get('apply_augmentation', False),
            description='Terapkan augmentasi pada test data',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        ),
        
        'batch_size_slider': widgets.IntSlider(
            value=test_config.get('batch_size', 16),
            min=1, max=64, step=1,
            description='Batch size:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
        ),
        
        'image_size_dropdown': widgets.Dropdown(
            options=[320, 416, 512, 640, 736, 832, 960, 1024, 1280],
            value=test_config.get('image_size', 640),
            description='Image size:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='100%')
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
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        ),
        
        'save_metrics_checkbox': widgets.Checkbox(
            value=eval_config.get('save_metrics', True),
            description='Simpan metrik evaluasi',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        ),
        
        'save_to_drive_checkbox': widgets.Checkbox(
            value=scenario_config.get('save_to_drive', True),
            description='Simpan hasil ke Google Drive',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        ),
        
        'drive_path_text': widgets.Text(
            value=scenario_config.get('drive_path', '/content/drive/MyDrive/SmartCash/evaluation_results'),
            placeholder='Path ke folder Google Drive untuk menyimpan hasil',
            description='Drive path:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='100%'),
            disabled=scenario_config.get('save_to_drive', True)
        ),
        
        'confusion_matrix_checkbox': widgets.Checkbox(
            value=eval_config.get('generate_confusion_matrix', True),
            description='Generate confusion matrix',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        ),
        
        'visualize_results_checkbox': widgets.Checkbox(
            value=eval_config.get('visualize_results', True),
            description='Visualisasi hasil prediksi',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        ),
        
        'inference_time_checkbox': widgets.Checkbox(
            value=eval_config.get('show_inference_time', True),
            description='Tampilkan metrik waktu inferensi',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='100%')
        )
    }
    
    # Action buttons for evaluation using the new API
    action_components = create_action_buttons(
        primary_label="Jalankan Evaluasi",
        primary_icon="▶️",
        secondary_buttons=[
            ("Cek Checkpoint", "🔍", "info"),
        ],
        cleanup_enabled=False,
        button_width='180px',
        primary_style='success'
    )
    
    # Get buttons using new API
    evaluate_button = action_components.get('primary_button')
    check_button = action_components.get('secondary_buttons', [None])[0] if action_components.get('secondary_buttons') else None
    
    # Fallback button creation if any button is missing
    if evaluate_button is None:
        print("[WARNING] Evaluate button not found, creating fallback")
        evaluate_button = widgets.Button(description='▶️ Jalankan Evaluasi', 
                                      button_style='success')
        evaluate_button.layout = widgets.Layout(width='180px')
    
    if check_button is None:
        print("[WARNING] Check button not found, creating fallback")
        check_button = widgets.Button(description='🔍 Cek Checkpoint')
        check_button.style.button_color = '#f0f0f0'
        check_button.layout = widgets.Layout(width='180px')
    
    # Buat checkpoint_selector container untuk mengelompokkan elemen-elemen terkait checkpoint
    checkpoint_selector = widgets.VBox([
        components['auto_select_checkbox'],
        components['checkpoint_path_text'],
        widgets.HBox([components['validation_metrics_select']])
    ], layout=widgets.Layout(margin='10px 0px'))
    
    # Update components with buttons and container
    components.update({
        'evaluate_button': evaluate_button,
        'check_button': check_button,
        'action_buttons_container': action_components.get('container', 
                                                       widgets.HBox([evaluate_button, check_button])),
        'primary_button': evaluate_button,  # For backward compatibility
        'secondary_buttons': [check_button] if check_button else [],  # For backward compatibility
        'checkpoint_selector': checkpoint_selector  # Tambahkan checkpoint_selector sebagai komponen tunggal
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