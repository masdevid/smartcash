"""
File: smartcash/ui/evaluation/components/evaluation_form.py
Deskripsi: Form components untuk model evaluation dengan checkpoint selection, test configuration, dan skenario pengujian
"""

import ipywidgets as widgets
from typing import Dict, Any, List
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.action_buttons import create_action_buttons

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
            value=scenario_config.get('test_folder', '/content/drive/MyDrive/SmartCash/dataset/test'),
            placeholder='Path ke folder test data',
            description='Test folder:',
            description_tooltip='Folder yang berisi gambar test untuk evaluasi model',
            style={'description_width': '80px'},
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
        
        'save_to_drive_checkbox': widgets.Checkbox(
            value=scenario_config.get('save_to_drive', True),
            description='Simpan hasil ke Google Drive',
            style={'description_width': '180px'}
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
            style={'description_width': '180px'}
        ),
        
        'visualize_results_checkbox': widgets.Checkbox(
            value=eval_config.get('visualize_results', True),
            description='Visualisasi hasil prediksi',
            style={'description_width': '160px'}
        )
    }
    
    # Action buttons untuk menjalankan evaluasi sesuai skenario
    action_buttons = create_action_buttons(
        primary_text="Jalankan Evaluasi",
        secondary_text="Batal",
        primary_tooltip="Jalankan evaluasi model sesuai skenario yang dipilih",
        secondary_tooltip="Batalkan proses evaluasi yang sedang berjalan",
        primary_icon="play",
        secondary_icon="stop",
        cleanup_enabled=False,
        primary_style='success'
    )
    
    # Buat checkpoint_selector container untuk mengelompokkan elemen-elemen terkait checkpoint
    checkpoint_selector = widgets.VBox([
        components['auto_select_checkbox'],
        components['checkpoint_path_text'],
        widgets.HBox([components['validation_metrics_select']])
    ], layout=widgets.Layout(margin='10px 0px'))
    
    # Merge dengan action buttons dan tambahkan checkpoint_selector
    components.update({
        **action_buttons,
        'evaluate_button': action_buttons['primary_button'],  # Tombol utama untuk evaluasi
        'cancel_button': action_buttons['secondary_button'],  # Tombol untuk membatalkan evaluasi
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