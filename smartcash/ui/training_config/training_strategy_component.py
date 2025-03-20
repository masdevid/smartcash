"""
File: smartcash/ui/training_config/training_strategy_component.py
Deskripsi: Komponen UI untuk konfigurasi strategi training model (tanpa augmentasi)
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_training_strategy_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk konfigurasi strategi training model.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI
    from smartcash.ui.utils.headers import create_header
    from smartcash.ui.utils.alerts import create_info_box
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Header
    header = create_header(
        f"<h3 style='color:{COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['chart']} Training Strategy</h3>",
        "Konfigurasi strategi dan teknik optimasi untuk training model SmartCash"
    )
    
    # Optimization strategy section
    optimization_section = widgets.HTML(
        f"<h3 style='color:{COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Optimization Strategy</h3>"
    )
    
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
    
    # Training policy section
    policy_section = widgets.HTML(
        f"<h3 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['tools']} Training Policy</h3>"
    )
    
    policy_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Save best model only',
            style={'description_width': 'initial'}
        ),
        widgets.IntSlider(
            value=5,
            min=1,
            max=10,
            description='Save every N epochs:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.IntSlider(
            value=15,
            min=5,
            max=30,
            description='Early stopping patience:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%')
        ),
        widgets.Checkbox(
            value=True,
            description='Validate every epoch',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Log to TensorBoard',
            style={'description_width': 'initial'}
        )
    ])
    
    # Strategy summary
    strategy_summary = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            margin='10px 0',
            padding='10px',    
            overflow='auto'
        )
    )
    
    # Tombol aksi
    from smartcash.ui.training_config.config_buttons import create_config_buttons
    buttons_container = create_config_buttons("Strategi Training")
    
    # Status output
    status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Info box with details
    info_box = create_info_box(
        "Tentang Strategi Training",
        """
        <p>Strategi training yang tepat dapat meningkatkan performa model secara signifikan:</p>
        <ul>
            <li><strong>Optimization</strong> - Teknik untuk mempercepat training dan mencapai konvergensi lebih baik</li>
            <li><strong>Training Policy</strong> - Pengaturan untuk proses dan monitoring training</li>
        </ul>
        <p><strong>Rekomendasi:</strong> Gunakan EMA dan cosine scheduler untuk kasus SmartCash. Augmentasi dataset harus dikonfigurasi pada bagian dataset preparation.</p>
        """,
        'info',
        collapsed=True
    )
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        optimization_section,
        optimization_options,
        policy_section,
        policy_options,
        strategy_summary,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        buttons_container,
        status,
        info_box
    ])
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': ui,
        'header': header,
        'optimization_options': optimization_options,
        'policy_options': policy_options,
        'strategy_summary': strategy_summary,
        'save_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'status': status,
        'module_name': 'training_strategy'
    }
    
    return ui_components