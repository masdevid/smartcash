"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Fixed UI components dengan struktur yang benar dan akses komponen yang proper
"""

import traceback
import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import with fallbacks
try:
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"⚠️ Warning: {str(e)}")
    IMPORT_SUCCESS = False

def _create_error_component(name: str, error: str) -> widgets.HTML:
    """Create error component dengan styling konsisten"""
    return widgets.HTML(f'<div style="color: #d32f2f; padding: 8px; font-size: 13px;">⚠️ {name}: {error}</div>')

def _create_fallback_ui(error: str) -> Dict[str, Any]:
    """Create minimal fallback UI dengan error message"""
    return {
        'ui': widgets.VBox([_create_error_component("UI Error", error)]),
        'status_panel': widgets.Output(),
        'log_output': widgets.Output(),
        'log_accordion': widgets.Accordion(children=[widgets.Output()]),
        'confirmation_area': widgets.Output(),
        'progress_tracker': None,
        'preprocess_button': widgets.Button(description="Error", disabled=True),
        'check_button': widgets.Button(description="Error", disabled=True),
        'cleanup_button': widgets.Button(description="Error", disabled=True),
        'save_button': widgets.Button(description="Error", disabled=True),
        'reset_button': widgets.Button(description="Error", disabled=True)
    }

def _create_confirmation_area() -> widgets.Output:
    """Create styled confirmation area"""
    return widgets.Output(layout=widgets.Layout(
        width='100%', min_height='50px', max_height='200px', 
        margin='10px 0', padding='5px', border='1px solid #e0e0e0',
        border_radius='4px', overflow='auto', background_color='#fafafa'
    ))

def _create_section_header(title: str, color: str) -> widgets.HTML:
    """Create styled section header"""
    return widgets.HTML(f"""
        <div style='
            font-weight: 600; color: {color}; font-size: 15px; 
            margin: 5px 0 10px 0; padding-bottom: 5px; 
            border-bottom: 2px solid {color}20;'
        >{title}</div>
    """)

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """🎯 Create preprocessing UI dengan struktur yang benar dan critical components yang accessible"""
    if not IMPORT_SUCCESS:
        return _create_fallback_ui("Gagal memuat komponen UI yang diperlukan")
    
    try:
        config = config or {}
        
        # === CORE COMPONENTS WITH PROPER ERROR HANDLING ===
        header = create_header("🔧 Dataset Preprocessing", "Preprocessing dataset dengan validasi dan real-time progress")
        status_panel = create_status_panel("🚀 Siap memulai preprocessing dataset", "info")
        
        # Input options dengan error handling
        try:
            input_options = create_preprocessing_input_options(config)
        except Exception as e:
            print(f"⚠️ Error creating input options: {str(e)}")
            input_options = widgets.VBox([_create_error_component('Input Options', str(e))])
        
        # Save/Reset buttons dengan error handling
        try:
            save_reset_buttons = create_save_reset_buttons(
                save_label="Simpan", reset_label="Reset",
                with_sync_info=True, sync_message="Konfigurasi disinkronkan dengan backend"
            )
            save_button = save_reset_buttons.get('save_button')
            reset_button = save_reset_buttons.get('reset_button')
            save_reset_container = save_reset_buttons.get('container')
        except Exception as e:
            print(f"⚠️ Error creating save/reset buttons: {str(e)}")
            save_button = widgets.Button(description="Simpan", disabled=True)
            reset_button = widgets.Button(description="Reset", disabled=True)
            save_reset_container = widgets.HBox([save_button, reset_button])
        
        # Action buttons dengan error handling
        try:
            action_buttons = create_action_buttons(
                primary_label="🚀 Mulai Preprocessing",
                primary_icon="play",
                secondary_buttons=[("🔍 Check Dataset", "search", "info")],
                cleanup_enabled=True,
                button_width='180px'
            )
            preprocess_button = action_buttons.get('download_button')  # primary button mapped
            check_button = action_buttons.get('check_button')
            cleanup_button = action_buttons.get('cleanup_button')
            action_container = action_buttons.get('container')
        except Exception as e:
            print(f"⚠️ Error creating action buttons: {str(e)}")
            preprocess_button = widgets.Button(description="🚀 Mulai Preprocessing", button_style='primary')
            check_button = widgets.Button(description="🔍 Check Dataset", button_style='info')
            cleanup_button = widgets.Button(description="🧹 Cleanup Dataset", button_style='warning')
            action_container = widgets.HBox([preprocess_button, check_button, cleanup_button])
        
        # Create confirmation area
        confirmation_area = _create_confirmation_area()
        
        # Progress tracker dengan error handling
        try:
            progress_tracker_components = create_dual_progress_tracker(
                operation="Dataset Preprocessing",
                auto_hide=False
            )
            progress_tracker = progress_tracker_components.get('tracker')
            progress_container = progress_tracker_components.get('container', widgets.VBox([]))
        except Exception as e:
            print(f"⚠️ Error creating progress tracker: {str(e)}")
            progress_tracker = None
            progress_container = widgets.VBox([])
        
        # Log accordion dengan error handling
        try:
            log_components = create_log_accordion(module_name='preprocessing', height='250px')
            log_output = log_components.get('log_output')
            log_accordion = log_components.get('log_accordion')
        except Exception as e:
            print(f"⚠️ Error creating log accordion: {str(e)}")
            log_output = widgets.Output()
            log_accordion = widgets.Accordion(children=[log_output])
            log_accordion.set_title(0, "📋 Log Preprocessing")
        
        # Create UI sections dengan structure yang jelas
        action_section = widgets.VBox([
            _create_section_header("🚀 Pipeline Operations", "#28a745"),
            action_container,
            widgets.HTML("<div style='margin: 5px 0 2px 0; font-size: 13px; color: #666;'>"
                        "<strong>📋 Status & Konfirmasi:</strong></div>"),
            confirmation_area
        ], layout=widgets.Layout(
            width='100%', margin='10px 0', padding='12px',
            border='1px solid #e0e0e0', border_radius='8px',
            background_color='#f9f9f9'
        ))
        
        # Config section
        config_section = widgets.VBox([
            widgets.Box([save_reset_container], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ], layout=widgets.Layout(margin='8px 0'))
        
        # Main UI assembly
        ui = widgets.VBox([
            header,
            status_panel,
            input_options,
            config_section,
            action_section,
            progress_container,
            log_accordion
        ], layout=widgets.Layout(
            width='100%',
            max_width='1200px',
            margin='0 auto',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            box_shadow='0 2px 4px rgba(0,0,0,0.05)'
        ))
        
        # Extract input components dengan safe access
        def safe_get_attr(obj, attr_name, default=None):
            """Safely get attribute dari object"""
            try:
                return getattr(obj, attr_name, default)
            except (AttributeError, TypeError):
                return default
        
        # Return components dengan PROPER STRUCTURE untuk critical components
        ui_components = {
            # CRITICAL COMPONENTS (harus ada untuk _get_critical_components)
            'ui': ui,
            'preprocess_button': preprocess_button,
            'check_button': check_button, 
            'cleanup_button': cleanup_button,
            'save_button': save_button,
            'reset_button': reset_button,
            'log_output': log_output,
            'status_panel': status_panel,
            
            # UI SECTIONS
            'header': header,
            'input_options': input_options,
            'action_section': action_section,
            'config_section': config_section,
            'confirmation_area': confirmation_area,
            'progress_tracker': progress_tracker,
            'progress_container': progress_container,
            'log_accordion': log_accordion,
            
            # INPUT COMPONENTS (extracted dari input_options)
            'resolution_dropdown': safe_get_attr(input_options, 'resolution_dropdown'),
            'normalization_dropdown': safe_get_attr(input_options, 'normalization_dropdown'),
            'target_splits_select': safe_get_attr(input_options, 'target_splits_select'),
            'batch_size_input': safe_get_attr(input_options, 'batch_size_input'),
            'validation_checkbox': safe_get_attr(input_options, 'validation_checkbox'),
            'preserve_aspect_checkbox': safe_get_attr(input_options, 'preserve_aspect_checkbox'),
            'move_invalid_checkbox': safe_get_attr(input_options, 'move_invalid_checkbox'),
            'invalid_dir_input': safe_get_attr(input_options, 'invalid_dir_input'),
            
            # METADATA
            'module_name': 'preprocessing',
            'ui_initialized': True,
            'data_dir': config.get('data', {}).get('dir', 'data')
        }
        
        return ui_components
        
    except Exception as e:
        error_msg = f"Gagal membuat UI: {str(e)}"
        print(f"⚠️ {error_msg}")
        traceback.print_exc()
        return _create_fallback_ui(error_msg)