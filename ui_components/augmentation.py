"""
File: smartcash/ui_components/augmentation.py
Author: Alfrida Sabar (refactored)
Deskripsi: Komponen UI untuk augmentasi dataset SmartCash.
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

def create_augmentation_ui():
    """Buat komponen UI untuk augmentasi dataset."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = widgets.HTML("<h2>🔄 Dataset Augmentation</h2><p>Augmentasi dataset untuk meningkatkan variasi dan jumlah data training</p>")
    
    # Augmentation options
    augmentation_options = widgets.VBox([
        widgets.SelectMultiple(
            options=['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation'],
            value=['Combined (Recommended)'],
            description='Augmentations:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%', height='100px')
        ),
        widgets.BoundedIntText(
            value=2,
            min=1,
            max=10,
            description='Variations:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Text(
            value='aug',
            description='Prefix:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=True,
            description='Validate results',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Resume if interrupted',
            style={'description_width': 'initial'}
        )
    ])
    
    # Advanced options for each augmentation type
    position_options = widgets.VBox([
        widgets.FloatSlider(value=0.5, min=0, max=1, step=0.05, description='Rotation prob:', 
                            style={'description_width': 'initial'}, layout=widgets.Layout(width='60%')),
        widgets.IntSlider(value=30, min=0, max=180, description='Max angle:', 
                          style={'description_width': 'initial'}, layout=widgets.Layout(width='60%')),
        widgets.FloatSlider(value=0.5, min=0, max=1, step=0.05, description='Flip prob:', 
                            style={'description_width': 'initial'}, layout=widgets.Layout(width='60%')),
        widgets.FloatSlider(value=0.3, min=0, max=1, step=0.05, description='Scale ratio:', 
                            style={'description_width': 'initial'}, layout=widgets.Layout(width='60%'))
    ])
    
    lighting_options = widgets.VBox([
        widgets.FloatSlider(value=0.5, min=0, max=1, step=0.05, description='Brightness prob:', 
                            style={'description_width': 'initial'}, layout=widgets.Layout(width='60%')),
        widgets.FloatSlider(value=0.3, min=0, max=1, step=0.05, description='Brightness limit:', 
                            style={'description_width': 'initial'}, layout=widgets.Layout(width='60%')),
        widgets.FloatSlider(value=0.5, min=0, max=1, step=0.05, description='Contrast prob:', 
                            style={'description_width': 'initial'}, layout=widgets.Layout(width='60%')),
        widgets.FloatSlider(value=0.3, min=0, max=1, step=0.05, description='Contrast limit:', 
                            style={'description_width': 'initial'}, layout=widgets.Layout(width='60%'))
    ])
    
    # Advanced settings accordion
    advanced_settings = widgets.Accordion(children=[position_options, lighting_options], 
                                          selected_index=None)
    advanced_settings.set_title(0, "🔄 Position Augmentation Parameters")
    advanced_settings.set_title(1, "💡 Lighting Augmentation Parameters")
    
    # Augmentation button and progress
    augmentation_button = widgets.Button(
        description='Run Augmentation',
        button_style='primary',
        icon='random'
    )
    
    augmentation_status = widgets.Output()
    augmentation_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout={'visibility': 'hidden'}
    )
    
    # Info box with examples
    info_box = widgets.HTML("""
    <div style="padding: 10px; background-color: #d1ecf1; border-left: 4px solid #0c5460; 
             color: #0c5460; margin: 10px 0; border-radius: 4px;">
        <h4 style="margin-top: 0;">ℹ️ Tentang Augmentasi Dataset</h4>
        <p>Augmentasi data adalah teknik untuk meningkatkan jumlah dan variasi data dengan menerapkan transformasi yang mempertahankan informasi label.</p>
        <p><strong>Jenis augmentasi:</strong></p>
        <ul>
            <li><strong>Combined</strong>: Kombinasi beberapa transformasi (direkomendasikan)</li>
            <li><strong>Position</strong>: Rotasi, flip, translasi, dan scaling</li>
            <li><strong>Lighting</strong>: Perubahan brightness, contrast, dan saturation</li>
            <li><strong>Extreme Rotation</strong>: Rotasi dengan sudut besar (untuk robustness)</li>
        </ul>
    </div>
    """)
    
    examples_tab = widgets.Tab()
    examples_tab.children = [
        widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Combined Augmentation</h4>
            <ul>
                <li>Flip horizontal + brightness adjustment</li>
                <li>Small rotation + contrast adjustment</li>
                <li>Small translation + saturation adjustment</li>
            </ul>
            <p><i>Ideal untuk meningkatkan robust model terhadap berbagai kondisi</i></p>
        </div>
        """),
        widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Position Variations</h4>
            <ul>
                <li>Horizontal & vertical flips</li>
                <li>Rotations (-30° to 30°)</li>
                <li>Scale changes (±20%)</li>
                <li>Translations in all directions</li>
            </ul>
            <p><i>Cocok untuk deteksi objek dengan posisi bervariasi</i></p>
        </div>
        """),
        widgets.HTML("""
        <div style="padding: 10px;">
            <h4>Lighting Variations</h4>
            <ul>
                <li>Brightness adjustments (±30%)</li>
                <li>Contrast changes (±20%)</li>
                <li>Saturation modifications (±25%)</li>
                <li>Slight blur or sharpening</li>
            </ul>
            <p><i>Ideal untuk meningkatkan robustness pada berbagai kondisi pencahayaan</i></p>
        </div>
        """)
    ]
    
    examples_tab.set_title(0, "Combined")
    examples_tab.set_title(1, "Position")
    examples_tab.set_title(2, "Lighting")
    
    # Pasang semua komponen
    main_container.children = [
        header,
        info_box,
        augmentation_options,
        advanced_settings,
        examples_tab,
        widgets.HBox([augmentation_button]),
        augmentation_progress,
        augmentation_status
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'augmentation_options': augmentation_options,
        'position_options': position_options,
        'lighting_options': lighting_options,
        'augmentation_button': augmentation_button,
        'augmentation_progress': augmentation_progress,
        'augmentation_status': augmentation_status,
        'examples_tab': examples_tab
    }
    
    return ui_components