"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: Fixed augmentation component dengan proper communicator integration untuk real-time progress
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import components
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.info_accordion import create_info_accordion
from smartcash.ui.components.status_panel import create_status_panel

# Import utilities
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_responsive_container, create_section_header
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.augmentor.communicator import create_communicator

def create_augmentation_ui(env=None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create augmentation UI dengan proper communicator integration untuk real-time progress"""
    
    # Environment setup
    env_manager = env or get_environment_Manager()
    config = config or {}
    
    # === PROGRESS TRACKING SECTION ===
    progress_components = create_progress_tracking_container()
    
    # === STATUS PANEL ===
    status_panel = create_status_panel("Siap untuk augmentasi dataset", "info")
    
    # === CONFIGURATION SECTION ===
    config_section = _create_config_section(config)
    
    # === ACTION BUTTONS SECTION ===
    action_buttons = create_action_buttons(
        primary_label="Mulai Augmentasi",
        primary_icon="play",
        secondary_buttons=[("Check Dataset", "search", "info")],
        cleanup_enabled=True,
        primary_style='success'
    )
    
    # === SAVE/RESET BUTTONS ===
    save_reset_buttons = create_save_reset_buttons(
        with_sync_info=True,
        sync_message="Konfigurasi disimpan ke Google Drive untuk persistensi."
    )
    
    # === CONFIRMATION AREA ===
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%'))
    
    # === INFO ACCORDION ===
    info_content = """
    <div style='padding: 10px;'>
        <h4>🎯 Pipeline Augmentasi SmartCash</h4>
        <p>Pipeline ini melakukan augmentasi data untuk meningkatkan dataset SmartCash:</p>
        <ul>
            <li><strong>📍 Variasi Posisi:</strong> Rotasi, flip, dan transformasi geometri</li>
            <li><strong>💡 Variasi Pencahayaan:</strong> Brightness, kontras, dan shadow</li>
            <li><strong>🔄 Normalisasi:</strong> Standarisasi ke format preprocessed</li>
        </ul>
        <p><strong>Target Split:</strong> Train only (untuk penelitian currency detection)</p>
    </div>
    """
    
    info_accordion = create_info_accordion(
        title="Info Pipeline Augmentasi",
        content=widgets.HTML(info_content),
        icon="info"
    )
    
    # === LOG OUTPUT ===
    log_output = widgets.Output(
        layout=widgets.Layout(
            max_height='250px',
            overflow='auto',
            border='1px solid #ddd',
            padding='8px',
            width='100%'
        )
    )
    
    # === MAIN UI LAYOUT ===
    ui_components = {
        # Core UI elements
        'status_panel': status_panel,
        'log_output': log_output,
        'confirmation_area': confirmation_area,
        
        # Progress tracking
        **progress_components,
        
        # Buttons
        **action_buttons,
        **save_reset_buttons,
        'augment_button': action_buttons['download_button'],  # Map primary button
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Configuration widgets
        **config_section,
        
        # Environment
        'env_manager': env_manager,
        'config': config,
        'logger_namespace': 'smartcash.ui.dataset.augmentation'
    }
    
    # === CREATE COMMUNICATOR ===
    ui_components['comm'] = create_communicator(ui_components)
    
    # === BUILD MAIN LAYOUT ===
    main_ui = create_responsive_container([
        create_section_header("🔄 Dataset Augmentation", icon="🔄"),
        status_panel,
        
        # Configuration section
        widgets.VBox([
            create_section_header("⚙️ Konfigurasi Augmentasi", icon="⚙️"),
            config_section['config_container']
        ]),
        
        # Action buttons section
        widgets.VBox([
            create_section_header("🚀 Aksi", icon="🚀"),
            action_buttons['container'],
            save_reset_buttons['container']
        ]),
        
        # Progress section
        widgets.VBox([
            create_section_header("📊 Progress", icon="📊"),
            progress_components['container']
        ]),
        
        # Log section
        widgets.VBox([
            create_section_header("📋 Log", icon="📋"),
            log_output
        ]),
        
        # Info section
        info_accordion['container'],
        
        # Hidden confirmation area
        confirmation_area
        
    ], container_type="vbox", padding="10px")
    
    ui_components['ui'] = main_ui
    
    return ui_components

def _create_config_section(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration section dengan enhanced widgets"""
    
    # Basic augmentation parameters
    num_variations = widgets.IntSlider(
        value=config.get('num_variations', 2),
        min=1, max=5, step=1,
        description='Variasi per gambar:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    target_count = widgets.IntSlider(
        value=config.get('target_count', 500),
        min=100, max=2000, step=50,
        description='Target count:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%')
    )
    
    output_prefix = widgets.Text(
        value=config.get('output_prefix', 'aug'),
        description='Prefix output:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='50%')
    )
    
    balance_classes = widgets.Checkbox(
        value=config.get('balance_classes', False),
        description='Balance classes',
        layout=widgets.Layout(width='50%')
    )
    
    # Augmentation types (research focused)
    augmentation_types = widgets.SelectMultiple(
        options=[
            ('🎯 Combined (Position + Lighting)', 'combined'),
            ('📍 Position Only', 'position'),
            ('💡 Lighting Only', 'lighting')
        ],
        value=['combined'],
        description='Jenis Augmentasi:',
        style={'description_width': '120px'},
        layout=widgets.Layout(width='100%', height='80px')
    )
    
    # Target split (fixed to train)
    target_split = widgets.Dropdown(
        options=[('Train', 'train')],
        value='train',
        description='Target Split:',
        disabled=True,  # Fixed untuk penelitian
        style={'description_width': '120px'},
        layout=widgets.Layout(width='50%')
    )
    
    # Advanced parameters dengan research-optimized values
    advanced_params = widgets.VBox([
        create_section_header("🔧 Parameter Advanced", icon="🔧", font_size="14px"),
        
        widgets.HBox([
            widgets.FloatSlider(value=config.get('fliplr', 0.5), min=0, max=1, step=0.1, description='Flip LR:', style={'description_width': '80px'}, layout=widgets.Layout(width='48%')),
            widgets.IntSlider(value=config.get('degrees', 10), min=0, max=30, description='Rotate (°):', style={'description_width': '80px'}, layout=widgets.Layout(width='48%'))
        ]),
        
        widgets.HBox([
            widgets.FloatSlider(value=config.get('translate', 0.1), min=0, max=0.3, step=0.05, description='Translate:', style={'description_width': '80px'}, layout=widgets.Layout(width='48%')),
            widgets.FloatSlider(value=config.get('scale', 0.1), min=0, max=0.3, step=0.05, description='Scale:', style={'description_width': '80px'}, layout=widgets.Layout(width='48%'))
        ]),
        
        widgets.HBox([
            widgets.FloatSlider(value=config.get('hsv_h', 0.015), min=0, max=0.1, step=0.005, description='HSV H:', style={'description_width': '80px'}, layout=widgets.Layout(width='48%')),
            widgets.FloatSlider(value=config.get('hsv_s', 0.7), min=0, max=1, step=0.1, description='HSV S:', style={'description_width': '80px'}, layout=widgets.Layout(width='48%'))
        ]),
        
        widgets.HBox([
            widgets.FloatSlider(value=config.get('brightness', 0.2), min=0, max=0.5, step=0.05, description='Brightness:', style={'description_width': '80px'}, layout=widgets.Layout(width='48%')),
            widgets.FloatSlider(value=config.get('contrast', 0.2), min=0, max=0.5, step=0.05, description='Contrast:', style={'description_width': '80px'}, layout=widgets.Layout(width='48%'))
        ])
    ])
    
    # Basic config container
    basic_config = widgets.VBox([
        num_variations,
        target_count,
        widgets.HBox([output_prefix, balance_classes]),
        augmentation_types,
        target_split
    ])
    
    # Main config container dengan accordion untuk advanced
    config_accordion = widgets.Accordion([advanced_params])
    config_accordion.set_title(0, "🔧 Parameter Advanced")
    config_accordion.selected_index = None  # Collapsed by default
    
    config_container = widgets.VBox([
        basic_config,
        config_accordion
    ])
    
    return {
        'config_container': config_container,
        
        # Basic parameters
        'num_variations': num_variations,
        'target_count': target_count,
        'output_prefix': output_prefix,
        'balance_classes': balance_classes,
        'augmentation_types': augmentation_types,
        'target_split': target_split,
        
        # Advanced parameters (extract from accordion children)
        'fliplr': advanced_params.children[1].children[0].children[0],
        'degrees': advanced_params.children[1].children[0].children[1],
        'translate': advanced_params.children[1].children[1].children[0],
        'scale': advanced_params.children[1].children[1].children[1],
        'hsv_h': advanced_params.children[1].children[2].children[0],
        'hsv_s': advanced_params.children[1].children[2].children[1],
        'brightness': advanced_params.children[1].children[3].children[0],
        'contrast': advanced_params.children[1].children[3].children[1]
    }