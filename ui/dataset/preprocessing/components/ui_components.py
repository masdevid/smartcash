"""
File: smartcash/ui/dataset/preprocessing/components/ui_components.py
Deskripsi: UI components untuk preprocessing dataset dengan fallback sederhana
"""

import ipywidgets as widgets
import traceback
from typing import Dict, Any, Optional, Tuple

# Import dengan fallback
try:
    from smartcash.ui.utils.header_utils import create_header
except ImportError:
    def create_header(title: str, description: str = "") -> widgets.HTML:
        return widgets.HTML(f'<h3 style="color: #2c3e50;">{title}</h3><p>{description}</p>')

try:
    from smartcash.ui.components.action_buttons import create_action_buttons
except ImportError:
    def create_action_buttons(**kwargs):
        return {'container': widgets.VBox(), 'download_button': widgets.Button(description='Mulai Preprocessing')}

try:
    from smartcash.ui.components.status_panel import create_status_panel
except ImportError:
    def create_status_panel(message: str, status: str = 'info') -> widgets.Output:
        output = widgets.Output()
        with output:
            print(f"[{status.upper()}] {message}")
        return output

try:
    from smartcash.ui.components.log_accordion import create_log_accordion
except ImportError:
    def create_log_accordion(module_name: str = 'preprocessing', height: str = '250px') -> Dict[str, Any]:
        output = widgets.Output(layout={'height': height, 'overflow': 'auto'})
        with output:
            print(f"Log {module_name} akan ditampilkan di sini")
        return {'log_output': output, 'log_accordion': output}

try:
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
except ImportError:
    def create_save_reset_buttons(**kwargs) -> Dict[str, Any]:
        return {
            'container': widgets.HBox(),
            'save_button': widgets.Button(description='Simpan'),
            'reset_button': widgets.Button(description='Reset')
        }

try:
    from smartcash.ui.components.progress_tracker import create_dual_progress_tracker
except ImportError:
    def create_dual_progress_tracker(operation: str, auto_hide: bool = True) -> Dict[str, Any]:
        return {
            'tracker': widgets.FloatProgress(
                value=0,
                min=0,
                max=100,
                description=f'{operation}:'
            ),
            'container': widgets.VBox()
        }

try:
    from smartcash.ui.dataset.preprocessing.components.input_options import create_preprocessing_input_options
except ImportError:
    def create_preprocessing_input_options(config: Dict[str, Any] = None) -> widgets.VBox:
        return widgets.VBox([
            widgets.Label('Opsi Preprocessing (Error loading full UI)'),
            widgets.HTML('<div style="color: red;">⚠️ Komponen input tidak dapat dimuat. Silakan periksa instalasi.</div>')
        ])

def _create_fallback_ui(error_message: str = None) -> Dict[str, Any]:
    """Membuat UI fallback sederhana dengan pesan error"""
    error_message = error_message or "Terjadi kesalahan dalam memuat UI preprocessing"
    
    # Buat progress tracker sederhana
    progress = widgets.FloatProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        style={'bar_color': '#2196F3'},
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    status = widgets.Output()
    with status:
        print("⚠️", error_message)
    
    ui = widgets.VBox([
        widgets.HTML('<h3>🔧 Dataset Preprocessing</h3>'),
        widgets.HTML(f'<div style="color: #d32f2f; padding: 10px; margin: 10px 0; border: 1px solid #ffcdd2; border-radius: 4px; background-color: #ffebee;">⚠️ {error_message}</div>'),
        progress,
        status
    ])
    
    return {
        'ui': ui,
        'progress_tracker': progress,
        'status': status,
        'is_fallback': True,
        'ui_initialized': True,
        'module_name': 'preprocessing',
        'preprocess_button': widgets.Button(description='Mulai Preprocessing'),
        'check_button': widgets.Button(description='Check Dataset'),
        'save_button': widgets.Button(description='Simpan'),
        'reset_button': widgets.Button(description='Reset'),
        'confirmation_area': status,
        'status_panel': status,
        'log_output': status,
        'log_accordion': status,
        'input_options': widgets.VBox()
    }

def create_preprocessing_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Membuat UI untuk preprocessing dataset dengan fallback sederhana"""
    try:
        config = config or {}
        
        # === CORE COMPONENTS ===
        try:
            header = create_header("🔧 Dataset Preprocessing", "Preprocessing dataset dengan validasi dan real-time progress")
        except Exception as e:
            header = widgets.HTML('<h3>🔧 Dataset Preprocessing</h3>')
        
        try:
            status_panel = create_status_panel("🚀 Siap memulai preprocessing dataset", "info")
        except Exception as e:
            status_panel = widgets.Output()
            with status_panel:
                print("Status: Siap memulai preprocessing dataset")
        
        try:
            input_options = create_preprocessing_input_options(config)
        except Exception as e:
            input_options = widgets.VBox([
                widgets.Label('Opsi Preprocessing'),
                widgets.HTML('<div style="color: #f39c12;">⚠️ Beberapa opsi tidak tersedia</div>')
            ])
        
        # === SAVE/RESET BUTTONS ===
        try:
            save_reset_buttons = create_save_reset_buttons(
                save_label="Simpan", 
                reset_label="Reset",
                with_sync_info=True, 
                sync_message="Konfigurasi disinkronkan dengan backend"
            )
        except Exception as e:
            save_reset_buttons = {
                'container': widgets.HBox([
                    widgets.Button(description='Simpan'),
                    widgets.Button(description='Reset')
                ]),
                'save_button': widgets.Button(description='Simpan'),
                'reset_button': widgets.Button(description='Reset')
            }
        
        # === ACTION BUTTONS ===
        try:
            action_buttons = create_action_buttons(
                primary_label="🚀 Mulai Preprocessing",
                primary_icon="play",
                secondary_buttons=[("🔍 Check Dataset", "search", "info")],
                cleanup_enabled=True,
                button_width='180px'
            )
        except Exception as e:
            action_buttons = {
                'container': widgets.HBox([
                    widgets.Button(description='🚀 Mulai Preprocessing'),
                    widgets.Button(description='🔍 Check Dataset')
                ]),
                'download_button': widgets.Button(description='🚀 Mulai Preprocessing'),
                'check_button': widgets.Button(description='🔍 Check Dataset'),
                'cleanup_button': None
            }
        
        # === CONFIRMATION AREA ===
        confirmation_area = widgets.Output(layout=widgets.Layout(
            width='100%', 
            min_height='50px',
            max_height='200px', 
            margin='10px 0',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='4px',
            overflow='auto',
            background_color='#f8f9fa'
        ))
        
        # === PROGRESS TRACKER ===
        try:
            progress_tracker_components = create_dual_progress_tracker(
                operation="Dataset Preprocessing",
                auto_hide=True
            )
            
            if isinstance(progress_tracker_components, dict):
                progress_tracker = progress_tracker_components.get('tracker')
                progress_container = progress_tracker_components.get('container', widgets.VBox([]))
            else:
                progress_tracker = progress_tracker_components
                progress_container = widgets.VBox([progress_tracker])
        except Exception as e:
            progress_tracker = widgets.FloatProgress(
                value=0,
                min=0,
                max=100,
                description='Progress:',
                style={'bar_color': '#007bff'}
            )
            progress_container = widgets.VBox([progress_tracker])
        
        # === LOG ACCORDION ===
        try:
            log_components = create_log_accordion(module_name='preprocessing', height='250px')
            log_output = log_components.get('log_output', widgets.Output())
            log_accordion = log_components.get('log_accordion', widgets.Accordion([widgets.Output()]))
        except Exception as e:
            log_output = widgets.Output()
            with log_output:
                print("Log preprocessing akan ditampilkan di sini")
            log_accordion = widgets.Accordion(children=[log_output])
            log_accordion.set_title(0, '📋 Log Preprocessing')
        
        # === ACTION SECTION ===
        action_section = widgets.VBox([
            _create_section_header("🚀 Pipeline Operations", "#28a745"),
            action_buttons.get('container', widgets.HTML("Tombol tidak tersedia")),
            widgets.HTML("<div style='margin: 5px 0 2px 0; font-size: 13px; color: #666;'><strong>📋 Status & Konfirmasi:</strong></div>"),
            confirmation_area
        ], layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            padding='12px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#f9f9f9'
        ))
        
        # === CONFIG SECTION ===
        config_section = widgets.VBox([
            widgets.Box([save_reset_buttons.get('container', widgets.HTML(""))], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ], layout=widgets.Layout(margin='8px 0'))
        
        # === MAIN UI ASSEMBLY ===
        ui_children = [
            header,
            status_panel,
            input_options,
            config_section,
            action_section,
            progress_container,
            log_accordion
        ]
        
        # Hapus komponen yang None
        ui_children = [c for c in ui_children if c is not None]
        
        ui = widgets.VBox(
            ui_children,
            layout=widgets.Layout(
                width='100%', 
                max_width='100%',
                display='flex',
                flex_flow='column',
                align_items='stretch',
                overflow='hidden',
                padding='10px'
            )
        )
        
        # === COMPONENTS MAPPING ===
        ui_components = {
            'ui': ui,
            'preprocess_button': action_buttons.get('download_button'),
            'check_button': action_buttons.get('check_button'),
            'cleanup_button': action_buttons.get('cleanup_button'),
            'save_button': save_reset_buttons.get('save_button'),
            'reset_button': save_reset_buttons.get('reset_button'),
            'confirmation_area': confirmation_area,
            'status_panel': status_panel,
            'progress_tracker': progress_tracker,
            'progress_container': progress_container,
            'log_output': log_output,
            'log_accordion': log_accordion,
            'status': log_output,
            'input_options': input_options,
            'module_name': 'preprocessing',
            'ui_initialized': True
        }
        
        # Tambahkan komponen input opsional
        for attr in ['resolution_dropdown', 'normalization_dropdown', 'target_splits_select', 
                    'batch_size_input', 'validation_checkbox', 'preserve_aspect_checkbox', 
                    'move_invalid_checkbox', 'invalid_dir_input']:
            ui_components[attr] = getattr(input_options, attr, None)
        
        return ui_components
        
    except Exception as e:
        error_msg = f"Gagal memuat UI: {str(e)}"
        traceback.print_exc()
        return _create_fallback_ui(error_msg)

def _create_section_header(title: str, color: str) -> widgets.HTML:
    """Create styled section header with fallback"""
    try:
        return widgets.HTML(f"""
        <h4 style="color: #333; margin: 8px 0 6px 0; border-bottom: 2px solid {color}; 
                   font-size: 14px; padding-bottom: 4px; font-weight: 600;">
            {title}
        </h4>
        """)
    except Exception as e:
        return widgets.HTML(f'<div style="color: #333; font-weight: bold;">{title}</div>')