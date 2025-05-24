"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Fixed UI components dengan konstanta yang benar dan fallback handling
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.utils.layout_utils import create_divider

# Import shared components yang sudah ada
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.progress_tracking import create_progress_tracking_container
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info

# Import local component
from .input_options import create_preprocessing_input_options

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create comprehensive preprocessing UI dengan shared components terbaru dan fixed constants.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        Dict[str, Any]: Dictionary komponen UI dengan keys yang konsisten
    """
    # Safe icon access dengan fallback
    def get_icon(key: str, fallback: str = "⚙️") -> str:
        try:
            return ICONS.get(key, fallback)
        except (NameError, AttributeError, KeyError):
            return fallback
    
    # Safe color access dengan fallback
    def get_color(key: str, fallback: str = "#333") -> str:
        try:
            return COLORS.get(key, fallback)
        except (NameError, AttributeError, KeyError):
            return fallback
    
    # Header dengan safe icon access
    header = create_header(
        f"{get_icon('processing', '🔧')} Dataset Preprocessing", 
        "Preprocessing dataset untuk training model SmartCash dengan normalisasi dan resize"
    )
    
    # Status panel
    status_panel = create_status_panel(
        "Siap memulai preprocessing dataset", "info"
    )
    
    # Input options (2-column layout)
    input_options = create_preprocessing_input_options(config)
    
    # Action buttons dengan consistent naming
    action_buttons = create_action_buttons(
        primary_label="Mulai Preprocessing",
        primary_icon="play",
        secondary_buttons=[
            ("Check Dataset", "search", "info")
        ],
        cleanup_enabled=True
    )
    
    # Save & reset buttons dengan safe creation
    try:
        save_reset_buttons = create_save_reset_buttons(
            save_label="Simpan",
            reset_label="Reset",
            save_tooltip="Simpan konfigurasi preprocessing",
            reset_tooltip="Reset konfigurasi ke default",
            with_sync_info=True,
            sync_message="Konfigurasi akan otomatis disinkronkan dengan Google Drive."
        )
    except TypeError:
        # Fallback jika parameter tidak didukung
        save_reset_buttons = create_save_reset_buttons(
            save_label="Simpan",
            reset_label="Reset"
        )
    
    # Log accordion
    log_components = create_log_accordion(
        module_name='preprocessing',
        height='250px'
    )
    
    # Help info dengan safe creation
    try:
        from smartcash.ui.info_boxes.preprocessing_info import get_preprocessing_info
        help_panel = get_preprocessing_info()
    except Exception:
        # Fallback help panel
        help_panel = widgets.HTML(
            value="""
            <div style="padding: 15px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
                <h5>💡 Info Preprocessing</h5>
                <p>Preprocessing akan mengubah ukuran gambar dan menormalisasi pixel untuk training yang optimal.</p>
                <ul>
                    <li><strong>Resolusi:</strong> Ukuran output gambar (direkomendasikan 640x640)</li>
                    <li><strong>Normalisasi:</strong> Metode normalisasi pixel (minmax direkomendasikan)</li>
                    <li><strong>Workers:</strong> Jumlah thread paralel (4-8 untuk performa optimal)</li>
                    <li><strong>Split:</strong> Bagian dataset yang diproses (all untuk semua split)</li>
                </ul>
                <div style="margin-top: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px;">
                    <strong>💡 Tips:</strong> Gunakan "Check Dataset" untuk memvalidasi struktur dataset sebelum preprocessing.
                </div>
            </div>
            """
        )
    
    # Progress tracking dengan 3-level system
    progress_components = create_progress_tracking_container()
    
    # Main UI assembly dengan proper spacing dan safe constants
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {get_color('dark', '#333')}; margin: 20px 0 10px 0;'>{get_icon('settings', '⚙️')} Konfigurasi Preprocessing</h4>"),
        input_options,
        save_reset_buttons['container'],
        create_divider(),
        widgets.HTML(f"<h4 style='color: {get_color('dark', '#333')}; margin: 15px 0 10px 0;'>{get_icon('play', '▶️')} Aksi Preprocessing</h4>"),
        action_buttons['container'],
        progress_components['container'],
        log_components['log_accordion'],
        create_divider(),
        help_panel
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Compile semua komponen dengan consistent key naming
    ui_components = {
        # Main UI
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        
        # Input components (direct references dari input_options)
        'input_options': input_options,
        'resolution_dropdown': getattr(input_options, 'resolution_dropdown', None),
        'normalization_dropdown': getattr(input_options, 'normalization_dropdown', None), 
        'worker_slider': getattr(input_options, 'worker_slider', None),
        'split_dropdown': getattr(input_options, 'split_dropdown', None),
        
        # Action buttons (consistent dengan handler expectations)
        'action_buttons': action_buttons,
        'preprocess_button': action_buttons['download_button'],  # Primary button mapped
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Save/reset buttons (consistent dengan handler expectations)
        'save_reset_buttons': save_reset_buttons,
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Progress components (3-level progress system)
        'progress_components': progress_components,
        'progress_container': progress_components['container'],
        'show_for_operation': progress_components.get('show_for_operation'),
        'update_progress': progress_components.get('update_progress'),
        'complete_operation': progress_components.get('complete_operation'),
        'error_operation': progress_components.get('error_operation'),
        'reset_all': progress_components.get('reset_all'),
        
        # Log components (consistent dengan logger bridge expectations)
        'log_components': log_components,
        'log_accordion': log_components['log_accordion'],
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],  # Alias for logger bridge
        
        # UI info areas
        'help_panel': help_panel,
        
        # Module metadata
        'module_name': 'preprocessing'
    }
    
    # Validate components untuk prevent None references
    critical_components = ['preprocess_button', 'check_button', 'save_button', 'reset_button']
    for comp_name in critical_components:
        if ui_components.get(comp_name) is None:
            # Create fallback button
            ui_components[comp_name] = widgets.Button(
                description=comp_name.replace('_', ' ').title(),
                button_style='primary' if 'preprocess' in comp_name else '',
                disabled=True,
                tooltip=f"Component {comp_name} tidak tersedia"
            )
    
    return ui_components