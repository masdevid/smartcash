"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: Fixed UI components dengan confirmation area yang benar, dual progress warna seragam, dan layout yang optimal
"""

import traceback
import ipywidgets as widgets
from typing import Dict, Any, Optional, Tuple

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
    """Create error component with consistent styling"""
    return widgets.HTML(f'<div style="color: #d32f2f; padding: 8px; font-size: 13px;">⚠️ {name}: {error}</div>')

def _create_fallback_ui(error: str) -> Dict[str, Any]:
    """Create minimal fallback UI with error message"""
    return {
        'container': widgets.VBox([_create_error_component("UI Error", error)]),
        'status_panel': widgets.Output(),
        'log_output': widgets.Output(),
        'log_accordion': widgets.Accordion(children=[widgets.Output()]),
        'confirmation_area': widgets.Output(),
        'progress_tracker': None,
        'action_buttons': {'container': widgets.VBox()},
        'save_reset_buttons': {'container': widgets.VBox()}
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
    """Create preprocessing UI with proper error handling and fallbacks"""
    if not IMPORT_SUCCESS:
        return _create_fallback_ui("Gagal memuat komponen UI yang diperlukan")
    
    try:
        config = config or {}
        
        # === CORE COMPONENTS WITH FALLBACKS ===
        header = create_header("🔧 Dataset Preprocessing", "Preprocessing dataset dengan validasi dan real-time progress")
        status_panel = create_status_panel("🚀 Siap memulai preprocessing dataset", "info")
        
        # Input options with fallback
        try:
            input_options = create_preprocessing_input_options(config)
            if not hasattr(input_options, 'get') or 'container' not in input_options:
                raise ValueError("Format input_options tidak valid")
        except Exception as e:
            input_options = {'container': _create_error_component('Input Options', str(e))}
        
        # Save/Reset buttons with fallback
        try:
            save_reset_buttons = create_save_reset_buttons(
                save_label="Simpan", reset_label="Reset",
                with_sync_info=True, sync_message="Konfigurasi disinkronkan dengan backend"
            )
            if not hasattr(save_reset_buttons, 'get') or 'container' not in save_reset_buttons:
                raise ValueError("Format save_reset_buttons tidak valid")
        except Exception as e:
            save_reset_buttons = {'container': _create_error_component('Save/Reset Buttons', str(e))}
        
        # Action buttons with fallback
        try:
            action_buttons = create_action_buttons(
                primary_label="🚀 Mulai Preprocessing",
                primary_icon="play",
                secondary_buttons=[("🔍 Check Dataset", "search", "info")],
                cleanup_enabled=True,
                button_width='180px'
            )
            if not hasattr(action_buttons, 'get') or 'container' not in action_buttons:
                raise ValueError("Format action_buttons tidak valid")
        except Exception as e:
            action_buttons = {'container': _create_error_component('Action Buttons', str(e))}
        
        # Create confirmation area
        confirmation_area = _create_confirmation_area()
        
        # Progress tracker with fallback
        try:
            progress_tracker_components = create_dual_progress_tracker(
                operation="Dataset Preprocessing",
                auto_hide=False
            )
            if isinstance(progress_tracker_components, dict):
                progress_tracker = progress_tracker_components.get('tracker')
                progress_container = progress_tracker_components.get('container', widgets.VBox([]))
            else:
                progress_tracker = progress_tracker_components
                progress_container = widgets.VBox([progress_tracker])
        except Exception as e:
            progress_tracker = None
            progress_container = _create_error_component('Progress Tracker', str(e))
        
        # Log accordion with fallback
        try:
            log_components = create_log_accordion(module_name='preprocessing', height='250px')
            log_output = log_components.get('log_output', widgets.Output())
            log_accordion = log_components.get('log_accordion', widgets.Accordion(children=[widgets.Output()]))
        except Exception as e:
            log_output = widgets.Output()
            log_accordion = widgets.Accordion(children=[widgets.Output()])
            with log_output:
                print(f"⚠️ Gagal memuat log accordion: {str(e)}")
        
        # Create UI sections
        action_section = widgets.VBox([
            _create_section_header("🚀 Pipeline Operations", "#28a745"),
            action_buttons.get('container', widgets.VBox()),
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
            widgets.Box([save_reset_buttons.get('container', widgets.VBox())], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ], layout=widgets.Layout(margin='8px 0'))
        
        # Main UI assembly
        ui = widgets.VBox([
            header,
            status_panel,
            input_options.get('container', widgets.VBox()),
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
        
        # Return components with error handling
        # Prepare components dictionary
        components = {
            'ui': ui,
            'components': {
                'header': header,
                'status_panel': status_panel,
                'input_options': input_options,
                'save_reset_buttons': save_reset_buttons,
                'action_buttons': action_buttons,
                'confirmation_area': confirmation_area,
                'progress_tracker': progress_tracker,
                'progress_container': progress_container,
                'log_output': log_output,
                'log_accordion': log_accordion,
                'action_section': action_section,
                'config_section': config_section,
                
                # Input components
                'resolution_dropdown': getattr(input_options.get('components', {}), 'resolution_dropdown', None),
                'normalization_dropdown': getattr(input_options.get('components', {}), 'normalization_dropdown', None),
                'target_splits_select': getattr(input_options.get('components', {}), 'target_splits_select', None),
                'batch_size_input': getattr(input_options.get('components', {}), 'batch_size_input', None),
                'validation_checkbox': getattr(input_options.get('components', {}), 'validation_checkbox', None),
                'preserve_aspect_checkbox': getattr(input_options.get('components', {}), 'preserve_aspect_checkbox', None),
                'move_invalid_checkbox': getattr(input_options.get('components', {}), 'move_invalid_checkbox', None),
                'invalid_dir_input': getattr(input_options.get('components', {}), 'invalid_dir_input', None),
                
                # Module metadata
                'module_name': 'preprocessing',
                'ui_initialized': True
            }
        }
        
        return components
        
    except Exception as e:
        error_msg = f"Gagal membuat UI: {str(e)}"
        print(f"⚠️ {error_msg}")
        traceback.print_exc()
        return _create_fallback_ui(error_msg)
    
    return ui_components

def _create_section_header(title: str, color: str) -> widgets.HTML:
    """Create styled section header"""
    return widgets.HTML(f"""
    <h4 style="color: #333; margin: 8px 0 6px 0; border-bottom: 2px solid {color}; 
               font-size: 14px; padding-bottom: 4px; font-weight: 600;">
        {title}
    </h4>
    """)