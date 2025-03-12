"""
File: smartcash/ui_components/training_strategy.py
Author: Refactor
Deskripsi: Komponen UI untuk konfigurasi strategi training model SmartCash (optimized).
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import (
    create_component_header, 
    create_section_title,
    create_info_alert,
    styled_html
)

def create_training_strategy_ui():
    """Buat komponen UI untuk konfigurasi strategi training model."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Training Strategy",
        "Konfigurasi strategi dan teknik optimasi untuk training model SmartCash",
        "🎯"
    )
    
    # Augmentation strategy section
    augmentation_section = create_section_title("Data Augmentation Strategy", "🔄")
    
    augmentation_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Enable data augmentation',
            style={'description_width': 'initial'}
        ),
        widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.05,
            description='Mosaic prob:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.05,
            description='Flip prob:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.FloatSlider(
            value=0.3,
            min=0,
            max=1.0,
            step=0.05,
            description='Scale jitter:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=False,
            description='Enable mixup',
            style={'description_width': 'initial'}
        )
    ])
    
    augmentation_details = widgets.HTML("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; color: #2c3e50">
        <h4 style="margin-top: 0; color: #2c3e50">📝 Augmentation Parameter Details</h4>
        <table style="width: 100%; border-collapse: collapse">
            <tr style="background-color: #f1f1f1">
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd">Parameter</th>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd">Description</th>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd">Impact</th>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>Mosaic</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Combines 4 training images into one</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Improves small object detection</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>Flip</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Horizontal image flipping</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Adds orientation variation</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>Scale</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Random scaling of images</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Improves scale invariance</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>Mixup</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Blends two images together</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Helps with boundary regularization</td>
            </tr>
        </table>
    </div>
    """)
    
    # Optimization strategy section
    optimization_section = create_section_title("Optimization Strategy", "⚙️")
    
    optimization_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Use mixed precision (FP16)',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Use Exponential Moving Average (EMA)',
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='10px 0')
        ),
        widgets.Checkbox(
            value=False,
            description='Use Stochastic Weight Averaging (SWA)',
            style={'description_width': 'initial'}
        ),
        widgets.Dropdown(
            options=['cosine', 'linear', 'step', 'constant'],
            value='cosine',
            description='LR schedule:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='60%', margin='10px 0')
        ),
        widgets.FloatText(
            value=0.01,
            description='Weight decay:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
    ])
    
    optimization_details = widgets.HTML("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; color: #2c3e50">
        <h4 style="margin-top: 0; color: #2c3e50">📝 Optimization Technique Details</h4>
        <table style="width: 100%; border-collapse: collapse">
            <tr style="background-color: #f1f1f1">
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd">Technique</th>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd">Description</th>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd">Effect</th>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>Mixed Precision</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Uses FP16 for speedup</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">2-3x faster training with lower memory</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>EMA</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Maintains moving average of weights</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Stabilizes training, better generalization</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>SWA</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Averages weights from multiple checkpoints</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Higher test accuracy, better generalization</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>Cosine LR</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Gradual learning rate decay</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Smooth convergence to optimum</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd"><b>Weight Decay</b></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">L2 regularization technique</td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd">Prevents overfitting by limiting weights</td>
            </tr>
        </table>
    </div>
    """)