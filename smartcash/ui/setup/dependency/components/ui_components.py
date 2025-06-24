"""
File: smartcash/ui/setup/dependency/components/ui_components.py
Deskripsi: Fixed UI components tanpa nested fallbacks, langsung display error UI jika gagal
"""

from typing import Dict, Any, Optional
import traceback
import ipywidgets as widgets
from IPython.display import display, HTML

def create_dependency_main_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create dependency installer UI dengan simple error handling
    
    Args:
        config: Konfigurasi UI
        
    Returns:
        Dictionary berisi komponen UI atau error UI yang bisa di-display
    """
    try:
        config = config or {}
        
        # Import components yang diperlukan
        from smartcash.ui.utils.constants import COLORS, ICONS
        from smartcash.ui.components import (
            create_header,
            create_action_buttons,
            create_dual_progress_tracker as create_progress_tracker,
            create_status_panel,
            create_log_accordion,
            create_save_reset_buttons,
            create_divider
        )
        from smartcash.ui.setup.dependency.utils import create_package_selector_grid
        
        # Helper untuk icon
        def get_icon(key: str, fallback: str = "📦") -> str:
            return ICONS.get(key, fallback)
        
        # Create komponen utama
        header = create_header(
            f"{get_icon('download')} Dependency Installer",
            "Setup packages yang diperlukan untuk SmartCash"
        )
        
        status_panel = create_status_panel(
            "Pilih packages yang akan diinstall dan klik tombol install",
            "info"
        )
        
        # Package selector
        package_selector = create_package_selector_grid(config)
        
        # Custom packages textarea
        custom_packages = widgets.Textarea(
            placeholder='Package tambahan (satu per baris)\ncontoh: numpy>=1.21.0\nopencv-python>=4.5.0',
            layout=widgets.Layout(
                width='100%',
                height='90px',
                margin='8px 0',
                border='1px solid #ddd',
                border_radius='4px'
            )
        )
        
        # Action buttons
        action_buttons = create_action_buttons(
            primary_label="Install",
            secondary_label="Analyze", 
            warning_label="System Report",
            primary_icon=get_icon('download'),
            secondary_icon=get_icon('search'),
            warning_icon=get_icon('info'),
            button_width='120px'
        )
        
        # Save/reset buttons
        save_reset_buttons = create_save_reset_buttons(
            save_label="Simpan",
            reset_label="Reset",
            button_width='100px'
        )
        
        # Progress tracker
        from smartcash.ui.info_boxes import get_dependencies_info
        progress_info = get_dependencies_info()
        progress_tracker = create_progress_tracker(progress_info)
        
        # Log accordion
        log_accordion = create_log_accordion()
        
        # Main UI assembly
        main_ui = widgets.VBox([
            header,
            status_panel,
            widgets.HTML(value="<h4>📦 Package Selection</h4>"),
            package_selector['widget'],
            widgets.HTML(value="<h4>➕ Custom Packages</h4>"),
            custom_packages,
            create_divider(),
            widgets.HBox([
                action_buttons['download_button'],
                action_buttons['check_button'], 
                action_buttons['cleanup_button']
            ]),
            widgets.HBox([
                save_reset_buttons['save_button'],
                save_reset_buttons['reset_button']
            ]),
            create_divider(),
            progress_tracker['widget'],
            log_accordion['widget']
        ])
        
        # Return komponen lengkap
        components = {
            'ui': main_ui,
            'header': header,
            'status_panel': status_panel,
            'package_selector': package_selector['widget'],
            'custom_packages': custom_packages,
            'install_button': action_buttons['download_button'],
            'analyze_button': action_buttons['check_button'],
            'check_button': action_buttons['check_button'],
            'save_button': save_reset_buttons['save_button'], 
            'reset_button': save_reset_buttons['reset_button'],
            'progress_tracker': progress_tracker['widget'],
            'progress_container': progress_tracker,
            'log_output': log_accordion['log_output'],
            'show_for_operation': progress_tracker.get('show_for_operation', lambda x: None),
            'update_progress': progress_tracker.get('update_progress', lambda x, y, z: None),
            'complete_operation': progress_tracker.get('complete_operation', lambda x: None),
            'error_operation': progress_tracker.get('error_operation', lambda x: None),
            'reset_all': progress_tracker.get('reset_all', lambda: None),
            'show_success': getattr(status_panel, 'show_success', lambda x: None),
            'show_error': getattr(status_panel, 'show_error', lambda x: None),
            'show_warning': getattr(status_panel, 'show_warning', lambda x: None),
            'show_info': getattr(status_panel, 'show_info', lambda x: None)
        }
        
        # Add package checkboxes jika ada
        if 'checkboxes' in package_selector:
            components.update(package_selector['checkboxes'])
        
        return components
        
    except Exception as e:
        # Buat error UI yang bisa di-display langsung
        error_type = type(e).__name__
        error_msg = str(e) or "Terjadi kesalahan yang tidak diketahui"
        error_traceback = traceback.format_exc()
        
        print(f"❌ Error dalam create_dependency_main_ui: {error_msg}")
        print(f"📋 Type: {error_type}")
        print("🔍 Traceback:")
        print(error_traceback)
        
        # Create error UI yang langsung bisa ditampilkan
        error_header = widgets.HTML(
            value=f"""
            <div style='background: #ffebee; padding: 16px; border-radius: 8px; border-left: 4px solid #f44336; margin: 8px 0;'>
                <h3 style='color: #c62828; margin: 0 0 8px 0;'>❌ Error membuat UI Components</h3>
                <p style='margin: 0 0 8px 0;'><strong>Module:</strong> dependency</p>
                <p style='margin: 0 0 8px 0;'><strong>Error:</strong> {error_msg}</p>
                <p style='margin: 0;'><strong>Type:</strong> {error_type}</p>
            </div>
            """,
            layout=widgets.Layout(width='100%')
        )
        
        error_details = widgets.Output()
        with error_details:
            print("🔍 Full Traceback:")
            print(error_traceback)
        
        error_accordion = widgets.Accordion(
            children=[error_details],
            titles=['📋 Error Details']
        )
        
        retry_button = widgets.Button(
            description="🔄 Retry",
            button_style='info',
            layout=widgets.Layout(width='100px', margin='8px 0')
        )
        
        error_ui = widgets.VBox([
            error_header,
            retry_button,
            error_accordion
        ])
        
        # Return error UI yang bisa langsung di-display
        return {
            'ui': error_ui,
            'error': True,
            'error_message': error_msg,
            'error_type': error_type,
            'traceback': error_traceback,
            'header': error_header,
            'status_panel': error_header,  # Use header as status panel
            'log_output': error_details,
            'retry_button': retry_button,
            # Minimal required components untuk prevent further errors
            'install_button': widgets.Button(description="Error", disabled=True),
            'analyze_button': widgets.Button(description="Error", disabled=True),
            'check_button': widgets.Button(description="Error", disabled=True),
            'save_button': widgets.Button(description="Error", disabled=True),
            'reset_button': widgets.Button(description="Error", disabled=True),
            'show_success': lambda x: print(f"✅ {x}"),
            'show_error': lambda x: print(f"❌ {x}"),
            'show_warning': lambda x: print(f"⚠️ {x}"),
            'show_info': lambda x: print(f"ℹ️ {x}"),
            'show_for_operation': lambda x: None,
            'update_progress': lambda x, y, z: None,
            'complete_operation': lambda x: None,
            'error_operation': lambda x: None,
            'reset_all': lambda: None
        }