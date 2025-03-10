"""
File: smartcash/ui_components/evaluation_components.py
Author: Alfrida Sabar
Deskripsi: Komponen UI untuk evaluasi model, menampilkan metrics dan performa model setelah training.
"""

import ipywidgets as widgets
from IPython.display import display, HTML

def create_model_selector_controls():
    """
    Buat kontrol untuk memilih model yang akan dievaluasi.
    
    Returns:
        Dictionary berisi widget untuk memilih model dan dataset
    """
    # Widget untuk memilih model
    model_dropdown = widgets.Dropdown(
        options=[],
        description='Model:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    # Widget untuk memilih dataset testing
    dataset_dropdown = widgets.Dropdown(
        options=[
            ('Dataset Test Default', None),
            ('Dataset Test (Posisi Bervariasi)', 'data/test_position_varied'),
            ('Dataset Test (Pencahayaan Bervariasi)', 'data/test_lighting_varied')
        ],
        value=None,
        description='Dataset Testing:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    
    return {
        'model_dropdown': model_dropdown,
        'dataset_dropdown': dataset_dropdown
    }

def create_evaluation_settings():
    """
    Buat pengaturan untuk evaluasi model.
    
    Returns:
        Dictionary berisi widget untuk pengaturan evaluasi
    """
    # Confidence threshold slider
    conf_threshold_slider = widgets.FloatSlider(
        value=0.25, 
        min=0.1, 
        max=0.9, 
        step=0.05, 
        description='Confidence Threshold:',
        style={'description_width': 'initial'}
    )
    
    # Number of runs slider
    num_runs_slider = widgets.IntSlider(
        value=3,
        min=1,
        max=5,
        step=1,
        description='Jumlah Run:',
        style={'description_width': 'initial'}
    )
    
    # Checkboxes for display options
    confusion_matrix_checkbox = widgets.Checkbox(
        value=True,
        description='Tampilkan confusion matrix',
        style={'description_width': 'initial'}
    )
    
    class_metrics_checkbox = widgets.Checkbox(
        value=True,
        description='Tampilkan metrik per kelas',
        style={'description_width': 'initial'}
    )
    
    # Group settings into an accordion
    settings = widgets.VBox([
        conf_threshold_slider,
        num_runs_slider,
        confusion_matrix_checkbox,
        class_metrics_checkbox
    ])
    
    accordion = widgets.Accordion(children=[settings])
    accordion.set_title(0, 'Pengaturan Evaluasi')
    
    return {
        'accordion': accordion,
        'conf_threshold_slider': conf_threshold_slider,
        'num_runs_slider': num_runs_slider,
        'confusion_matrix_checkbox': confusion_matrix_checkbox,
        'class_metrics_checkbox': class_metrics_checkbox
    }

def create_evaluation_ui():
    """
    Buat UI lengkap untuk evaluasi model.
    
    Returns:
        Dictionary berisi semua komponen UI untuk evaluasi model
    """
    # Buat header dengan styling
    header = widgets.HTML("<h2>📊 Evaluasi Model</h2>")
    description = widgets.HTML("<p>Jalankan evaluasi untuk menilai performa model setelah training.</p>")
    
    # Buat model selector controls
    selector_controls = create_model_selector_controls()
    
    # Buat evaluation settings
    settings = create_evaluation_settings()
    
    # Tombol untuk evaluasi
    run_evaluation_button = widgets.Button(
        description='Evaluasi Model',
        button_style='primary',
        icon='check'
    )
    
    # Output area
    evaluation_output = widgets.Output()
    
    # Gabungkan semua komponen dalam layout utama
    main_ui = widgets.VBox([
        header,
        description,
        selector_controls['model_dropdown'],
        selector_controls['dataset_dropdown'],
        settings['accordion'],
        run_evaluation_button,
        evaluation_output
    ])
    
    # Return struktur UI dan komponen individual untuk handler
    return {
        'ui': main_ui,
        'model_dropdown': selector_controls['model_dropdown'],
        'dataset_dropdown': selector_controls['dataset_dropdown'],
        'settings_accordion': settings['accordion'],
        'conf_threshold_slider': settings['conf_threshold_slider'],
        'num_runs_slider': settings['num_runs_slider'],
        'confusion_matrix_checkbox': settings['confusion_matrix_checkbox'],
        'class_metrics_checkbox': settings['class_metrics_checkbox'],
        'run_evaluation_button': run_evaluation_button,
        'evaluation_output': evaluation_output
    }