"""smartcash/ui/training/components/training_component.py

Komponen UI untuk training module.
"""

import ipywidgets as widgets
import pandas as pd
import plotly.graph_objs as go
from IPython.display import display
from smartcash.ui.components import (
    progress_tracker,
    save_reset_buttons,
    log_accordion,
    metric_card
)
from smartcash.ui.utils import fallback_utils, ui_utils
from .confusion_matrix import ConfusionMatrixAccordion

def create_training_ui(config):
    """Membuat UI untuk training module"""
    # 1. Progress Tracker 3 level
    progress = progress_tracker.create_progress_tracker(
        levels=['Epoch', 'Batch', 'Overall'],
        descriptions=['Progress Epoch', 'Progress Batch', 'Progress Keseluruhan'],
        layout=widgets.Layout(margin='0 0 20px 0')
    )
    
    # 2. Status dan Confirmation Area
    status_panel = fallback_utils.create_status_panel()
    
    # 3. Action Buttons
    buttons = save_reset_buttons.create_action_buttons(
        primary_label='Mulai Training',
        secondary_labels=['Jeda', 'Hentikan'],
        button_style='primary',
        layout=widgets.Layout(margin='0 0 20px 0')
    )
    
    # 4. Metrics Cards Dashboard
    metrics_cards = _create_metrics_dashboard()
    
    # 5. Accordion Charts
    charts_accordion = _create_charts_accordion()
    
    # 6. Confusion Matrix Accordion
    confusion_matrix = ConfusionMatrixAccordion()
    
    # 7. Log Accordion
    log_acc = log_accordion.create_log_accordion()
    
    # Container Utama dengan styling konsisten
    main_container = widgets.VBox([
        ui_utils.create_section_header('🏋️‍♂️ Training Module'),
        progress['container'],
        status_panel,
        buttons['container'],
        ui_utils.create_section_header('📊 Live Metrics'),
        metrics_cards,
        ui_utils.create_section_header('📈 Training Charts'),
        charts_accordion,
        ui_utils.create_section_header('🧮 Confusion Matrix'),
        confusion_matrix.accordion,
        ui_utils.create_section_header('📝 Training Log'),
        log_acc
    ], layout=widgets.Layout(
        width='100%',
        padding='20px',
        border='1px solid #e0e0e0',
        border_radius='5px',
        margin='0 0 20px 0'
    ))
    
    return {
        'container': main_container,
        'progress': progress,
        'status_panel': status_panel,
        'buttons': buttons,
        'metrics_cards': metrics_cards,
        'charts_accordion': charts_accordion,
        'confusion_matrix': confusion_matrix,
        'log_accordion': log_acc
    }

def _create_metrics_dashboard():
    """Membuat dashboard metrik"""
    # Daftar metrik yang akan ditampilkan
    metrics = [
        ('mAP', '📊', 'Mean Average Precision'),
        ('Loss', '📉', 'Training Loss'),
        ('Akurasi', '🎯', 'Akurasi Deteksi'),
        ('Presisi', '✅', 'Presisi Deteksi'),
        ('F1', '⚖️', 'F1 Score'),
        ('Waktu Inferensi', '⏱️', 'Rata-rata Waktu Inferensi (ms)')
    ]
    
    # Buat kartu metrik
    cards = []
    for metric, icon, tooltip in metrics:
        cards.append(metric_card.create_metric_card(
            title=metric,
            value='0',
            icon=icon,
            description=tooltip,
            layout=widgets.Layout(width='95%')
        ))
    
    # Atur dalam grid responsif
    grid = widgets.GridBox(
        cards,
        layout=widgets.Layout(
            grid_template_columns='repeat(3, 1fr)',
            grid_gap='15px',
            width='100%',
            margin='0 0 20px 0'
        )
    )
    return grid

def _create_charts_accordion():
    """Membuat accordion untuk grafik"""
    # Buat plot placeholder
    loss_plot = go.FigureWidget(data=[go.Scatter(x=[], y=[], name='Loss')])
    map_plot = go.FigureWidget(data=[go.Scatter(x=[], y=[], name='mAP')])
    
    # Atur layout plot
    loss_plot.update_layout(
        title='Loss Curve',
        height=300,
        template='plotly_white'
    )
    map_plot.update_layout(
        title='mAP Curve',
        height=300,
        template='plotly_white'
    )
    
    # Buat accordion
    accordion = widgets.Accordion(children=[
        widgets.VBox([loss_plot], layout=widgets.Layout(padding='10px')),
        widgets.VBox([map_plot], layout=widgets.Layout(padding='10px'))
    ])
    accordion.set_title(0, '📉 Loss Curve')
    accordion.set_title(1, '📊 mAP Curve')
    
    return accordion

def _create_confusion_matrix_accordion():
    """Membuat accordion untuk confusion matrix"""
    # Buat placeholder matrix
    classes = ['Rp1000', 'Rp2000', 'Rp5000', 'Rp10000', 'Rp20000', 'Rp50000', 'Rp100000']
    data = [[0]*len(classes) for _ in classes]
    
    # Buat heatmap
    heatmap = go.FigureWidget(data=go.Heatmap(
        z=data,
        x=classes,
        y=classes,
        colorscale='Blues',
        hoverongaps=False
    ))
    
    heatmap.update_layout(
        title='Confusion Matrix',
        height=500,
        xaxis_title='Predicted',
        yaxis_title='Actual',
        template='plotly_white'
    )
    
    # Buat accordion
    accordion = widgets.Accordion(children=[
        widgets.VBox([heatmap], layout=widgets.Layout(padding='10px'))
    ])
    accordion.set_title(0, '🧮 Confusion Matrix')
    
    return accordion
