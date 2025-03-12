"""
File: smartcash/ui_components/split_config.py
Author: Refactor
Deskripsi: Komponen UI untuk konfigurasi split dataset (optimized).
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import create_component_header, create_info_box, create_section_title

def create_split_config_ui():
    """Buat komponen UI untuk konfigurasi split dataset."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Dataset Split Configuration",
        "Konfigurasi pembagian dataset untuk training, validation, dan testing",
        "✂️"
    )
    
    # Panel informasi dataset saat ini
    current_stats = widgets.Output(
        layout=widgets.Layout(
            margin='10px 0',
            border='1px solid #ddd',
            padding='10px',
            max_height='200px',
            overflow='auto'
        )
    )
    
    # Split percentages panel with slider
    split_panel = widgets.VBox([
        widgets.HTML("<h3 style='color:#2c3e50'>🔢 Split Percentages</h3>"),
        widgets.FloatSlider(
            value=70.0,
            min=50.0,
            max=90.0,
            step=1.0,
            description='Train:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='70%'),
            readout_format='.0f'
        ),
        widgets.FloatSlider(
            value=15.0,
            min=5.0,
            max=30.0,
            step=1.0,
            description='Valid:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='70%'),
            readout_format='.0f'
        ),
        widgets.FloatSlider(
            value=15.0,
            min=5.0,
            max=30.0,
            step=1.0,
            description='Test:',
            style={'description_width': '60px'},
            layout=widgets.Layout(width='70%'),
            readout_format='.0f'
        ),
        widgets.Checkbox(
            value=True,
            description='Stratified split (preserve class distribution)',
            style={'description_width': 'initial'}
        )
    ])
    
    # Advanced options panel
    advanced_options = widgets.VBox([
        widgets.IntText(
            value=42,
            description='Random seed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='40%')
        ),
        widgets.Checkbox(
            value=False,
            description='Force resplit dataset',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Preserve folder structure',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Backup dataset before split',
            style={'description_width': 'initial'}
        ),
        widgets.Text(
            value='data/splits_backup',
            description='Backup dir:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='60%')
        )
    ])
    
    # Advanced settings accordion
    advanced_accordion = widgets.Accordion(children=[advanced_options], selected_index=None)
    advanced_accordion.set_title(0, "⚙️ Advanced Split Settings")
    
    # Buttons container
    buttons_container = widgets.HBox([
        widgets.Button(
            description='Apply Split',
            button_style='primary',
            icon='scissors',
            layout=widgets.Layout(margin='0 10px 0 0')
        ),
        widgets.Button(
            description='Reset to Default',
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(margin='0 10px 0 0') 
        ),
        widgets.Button(
            description='Save Configuration',
            button_style='success',
            icon='save',
            layout=widgets.Layout(margin='0')
        )
    ])
    
    # Status output
    split_status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Results output
    stats_output = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto',
            display='none'
        )
    )
    
    # Info box with helpful information
    info_box = create_info_box(
        "Tentang Split Dataset",
        """
        <p style="color: #0c5460">Pembagian dataset menjadi 3 subset:</p>
        <ul style="color: #0c5460">
            <li><strong>Train</strong>: Dataset untuk training model (biasanya 70-80%)</li>
            <li><strong>Validation</strong>: Dataset untuk validasi selama training (biasanya 10-15%)</li>
            <li><strong>Test</strong>: Dataset untuk evaluasi final model (biasanya 10-15%)</li>
        </ul>
        <p style="color: #0c5460">Gunakan <strong>stratified split</strong> untuk memastikan distribusi kelas tetap seimbang di semua subset.</p>
        <p style="color: #0c5460">Total persentase harus 100%. Jika tidak, nilai akan disesuaikan secara otomatis.</p>
        """,
        'info'
    )
    
    # Pasang semua komponen
    main_container.children = [
        header,
        current_stats,
        info_box,
        split_panel,
        advanced_accordion,
        buttons_container,
        split_status,
        stats_output
    ]
    
    # Dictionary untuk akses komponen dari luar
    ui_components = {
        'ui': main_container,
        'current_stats': current_stats,
        'split_sliders': [split_panel.children[1], split_panel.children[2], split_panel.children[3]],
        'stratified': split_panel.children[4],
        'advanced_options': advanced_options,
        'split_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'save_button': buttons_container.children[2],
        'split_status': split_status,
        'stats_output': stats_output
    }
    
    return ui_components