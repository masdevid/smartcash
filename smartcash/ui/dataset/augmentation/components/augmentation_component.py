"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: Fixed augmentation component dengan existing widgets integration dan one-liner style
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import existing components
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.components.info_accordion import create_info_accordion
from smartcash.ui.components.status_panel import create_status_panel

# Import existing widget components
from .basic_options_widget import create_basic_options_widget
from .advanced_options_widget import create_advanced_options_widget
from .augmentation_types_widget import create_augmentation_types_widget
from .split_selector import create_split_selector
from . import form_fields

# Import utilities
from smartcash.ui.utils.layout_utils import create_responsive_container, create_section_header
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.augmentor.communicator import create_communicator

def create_augmentation_ui(env=None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create augmentation UI dengan existing widgets integration dan one-liner style"""
    
    # One-liner environment dan config setup
    env_manager, config = env or get_environment_manager(), config or {}
    
    # One-liner UI components creation
    progress_components = create_progress_tracking_container()
    status_panel = create_status_panel("Siap untuk augmentasi dataset", "info")
    
    # Create existing widget components
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    aug_types = create_augmentation_types_widget()
    split_selector = create_split_selector()
    
    # Form fields dengan config values
    num_variations = form_fields.num_variations_field(config)
    target_count = form_fields.target_count_field(config)
    output_prefix = form_fields.output_prefix_field(config)
    augmentation_types = form_fields.augmentation_types_field(config)
    target_split = form_fields.split_target_field(config)
    balance_classes = form_fields.balance_classes_field(config)
    
    # One-liner action buttons
    action_buttons = create_action_buttons(
        primary_label="Mulai Augmentasi", primary_icon="play",
        secondary_buttons=[("Check Dataset", "search", "info")],
        cleanup_enabled=True, primary_style='success'
    )
    
    # One-liner save/reset buttons
    save_reset_buttons = create_save_reset_buttons(
        with_sync_info=True,
        sync_message="Konfigurasi disimpan ke Google Drive untuk persistensi."
    )
    
    # One-liner widgets creation
    confirmation_area = widgets.Output(layout=widgets.Layout(width='100%'))
    log_output = widgets.Output(layout=widgets.Layout(max_height='250px', overflow='auto', border='1px solid #ddd', padding='8px', width='100%'))
    
    # One-liner info accordion
    info_content = widgets.HTML("""
    <div style='padding: 10px;'>
        <h4>🎯 Pipeline Augmentasi SmartCash</h4>
        <p>Pipeline untuk augmentasi dataset currency detection:</p>
        <ul>
            <li><strong>📍 Variasi Posisi:</strong> Rotasi, flip, transformasi geometri</li>
            <li><strong>💡 Variasi Pencahayaan:</strong> Brightness, kontras, shadow</li>
            <li><strong>🔄 Normalisasi:</strong> Standarisasi ke format preprocessed</li>
        </ul>
        <p><strong>Target:</strong> Train split untuk penelitian currency detection</p>
    </div>
    """)
    info_accordion = create_info_accordion("Info Pipeline Augmentasi", info_content, "info")
    
    # One-liner configuration section
    config_section = widgets.VBox([
        create_section_header("⚙️ Konfigurasi Dasar", "⚙️"),
        widgets.HBox([
            widgets.VBox([num_variations, target_count], layout=widgets.Layout(width='48%')),
            widgets.VBox([output_prefix, balance_classes], layout=widgets.Layout(width='48%'))
        ]),
        aug_types['container'],
        widgets.Accordion([advanced_options['container']], layout=widgets.Layout(width='100%'))
    ])
    
    # Set accordion title
    config_section.children[-1].set_title(0, "🔧 Parameter Advanced")
    config_section.children[-1].selected_index = None
    
    # One-liner UI components assembly
    ui_components = {
        **{k: v for k, v in locals().items() if k in ['status_panel', 'log_output', 'confirmation_area', 'num_variations', 'target_count', 'output_prefix', 'balance_classes', 'augmentation_types', 'target_split']},
        **progress_components, **action_buttons, **save_reset_buttons, **basic_options['widgets'], **advanced_options['widgets'], **aug_types['widgets'],
        'augment_button': action_buttons['download_button'], 'check_button': action_buttons['check_button'], 'cleanup_button': action_buttons.get('cleanup_button'),
        'env_manager': env_manager, 'config': config, 'logger_namespace': 'smartcash.ui.dataset.augmentation'
    }
    
    # One-liner communicator creation
    ui_components['comm'] = create_communicator(ui_components)
    
    # One-liner main UI layout
    ui_components['ui'] = create_responsive_container([
        create_section_header("🔄 Dataset Augmentation", "🔄"),
        status_panel, config_section,
        widgets.VBox([create_section_header("🚀 Aksi", "🚀"), action_buttons['container'], save_reset_buttons['container']]),
        widgets.VBox([create_section_header("📊 Progress", "📊"), progress_components['container']]),
        widgets.VBox([create_section_header("📋 Log", "📋"), log_output]),
        info_accordion['container'], confirmation_area
    ], container_type="vbox", padding="10px")
    
    return ui_components