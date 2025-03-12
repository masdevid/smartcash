"""
File: smartcash/ui_components/augmentation.py
Author: Refactored
Deskripsi: Komponen UI untuk augmentasi dataset SmartCash dengan pendekatan DRY.
"""

import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.utils.ui_utils import (
    create_component_header, 
    create_info_box, 
    create_section_title,
    styled_html
)

def create_augmentation_ui():
    """Buat komponen UI untuk augmentasi dataset dengan pendekatan DRY."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Dataset Augmentation",
        "Augmentasi dataset untuk meningkatkan variasi dan jumlah data training",
        "🎨"
    )
    
    # Augmentation options
    augmentation_section = create_section_title("Augmentation Options", "🔄")
    
    augmentation_options = widgets.VBox([
        widgets.SelectMultiple(
            options=['Combined (Recommended)', 'Position Variations', 'Lighting Variations', 'Extreme Rotation'],
            value=['Combined (Recommended)'],
            description='Types:',
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
    augmentation_advanced = create_section_title("Advanced Settings", "⚙️")
    
    position_options = widgets.VBox([
        widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.05, 
            description='Rotation prob:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.IntSlider(
            value=30, min=0, max=180, 
            description='Max angle:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.05, 
            description='Flip prob:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, 
            description='Scale ratio:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        )
    ])
    
    lighting_options = widgets.VBox([
        widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.05, 
            description='Brightness prob:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, 
            description='Brightness limit:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.5, min=0, max=1, step=0.05, 
            description='Contrast prob:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        ),
        widgets.FloatSlider(
            value=0.3, min=0, max=1, step=0.05, 
            description='Contrast limit:', 
            style={'description_width': 'initial'}, 
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Advanced settings accordion
    advanced_tabs = widgets.Tab(children=[position_options, lighting_options])
    advanced_tabs.set_title(0, "Position Parameters")
    advanced_tabs.set_title(1, "Lighting Parameters")
    
    # Action buttons
    button_section = create_section_title("Actions", "▶️")
    
    augmentation_button = widgets.Button(
        description='Run Augmentation',
        button_style='primary',
        icon='random',
        layout=widgets.Layout(width='auto')
    )
    
    reset_button = widgets.Button(
        description='Reset Settings',
        button_style='warning',
        icon='refresh',
        layout=widgets.Layout(width='auto')
    )
    
    cleanup_button = widgets.Button(
        description='Clean Augmented Data',
        button_style='danger',
        icon='trash',
        layout=widgets.Layout(width='auto')
    )
    
    restore_button = widgets.Button(
        description='Restore from Backup',
        button_style='info',
        icon='undo',
        layout=widgets.Layout(width='auto', display='none')  # Hidden for future development
    )
    
    save_config_button = widgets.Button(
        description='Save Configuration',
        button_style='success',
        icon='save',
        layout=widgets.Layout(width='auto')
    )
    
    buttons_container = widgets.HBox([
        augmentation_button, 
        reset_button, 
        cleanup_button, 
        save_config_button
    ], layout=widgets.Layout(justify_content='space-between'))
    
    # Progress and status
    augmentation_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='0%',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    augmentation_status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            max_height='200px',
            min_height='100px',
            overflow='auto',
            width='100%'
        )
    )
    
    # Info and examples sections
    info_section = create_section_title("Information", "ℹ️")
    
    info_box = create_info_box(
        "Tentang Augmentasi Dataset",
        """
        <p>Augmentasi data adalah teknik untuk meningkatkan jumlah dan variasi data dengan menerapkan transformasi yang mempertahankan informasi label.</p>
        <p><strong>Jenis augmentasi:</strong></p>
        <ul>
            <li><strong>Combined</strong>: Kombinasi beberapa transformasi (direkomendasikan)</li>
            <li><strong>Position</strong>: Rotasi, flip, translasi, dan scaling</li>
            <li><strong>Lighting</strong>: Perubahan brightness, contrast, dan saturation</li>
            <li><strong>Extreme Rotation</strong>: Rotasi dengan sudut besar (untuk robustness)</li>
        </ul>
        """,
        'info'
    )
    
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
    
    # Results section (initially hidden)
    results_section = create_section_title("Results", "📊")
    results_display = widgets.Output(layout=widgets.Layout(display='none'))
    
    # Pasang semua komponen
    main_container.children = [
        header,
        augmentation_section,
        augmentation_options,
        augmentation_advanced,
        advanced_tabs,
        button_section,
        buttons_container,
        augmentation_progress,
        augmentation_status,
        results_display,
        info_section,
        info_box,
        examples_tab
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'augmentation_options': augmentation_options,
        'position_options': position_options,
        'lighting_options': lighting_options,
        'advanced_tabs': advanced_tabs,
        'augmentation_button': augmentation_button,
        'reset_button': reset_button,
        'cleanup_button': cleanup_button,
        'restore_button': restore_button,
        'save_config_button': save_config_button,
        'augmentation_progress': augmentation_progress,
        'augmentation_status': augmentation_status,
        'examples_tab': examples_tab,
        'results_display': results_display
    }
    
    return ui_components