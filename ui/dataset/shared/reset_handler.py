"""
File: smartcash/ui/dataset/shared/reset_handler.py
Deskripsi: Handler untuk mereset UI dan konfigurasi ke default untuk preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.ui.dataset.shared.status_panel import update_status_panel

def setup_reset_handler(ui_components: Dict[str, Any], config_handler: Dict[str, Callable], module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Setup handler untuk reset UI dan konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config_handler: Dictionary berisi fungsi-fungsi handler konfigurasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk tombol reset
    def on_reset_click(b):
        """Reset UI dan konfigurasi ke default."""
        # Reset UI
        reset_ui()
        
        # Tentukan fungsi-fungsi yang digunakan berdasarkan module_type
        if 'load_default_config' in config_handler and 'update_ui_from_config' in config_handler:
            try:
                # Load konfigurasi default
                default_config = config_handler['load_default_config']()
                
                # Update UI dari konfigurasi default
                config_handler['update_ui_from_config'](ui_components, default_config)
                
                # Update status panel
                update_status_panel(ui_components, "success", f"{ICONS['success']} Konfigurasi direset ke nilai default")
                
                # Reset logs
                with ui_components['status']:
                    clear_output()
                    display(create_status_indicator("success", f"{ICONS['success']} UI dan konfigurasi berhasil direset ke nilai default"))
                
                # Log success jika logger tersedia
                if logger: logger.success(f"{ICONS['success']} Konfigurasi berhasil direset ke nilai default")
            except Exception as e:
                update_status_panel(ui_components, "warning", f"{ICONS['warning']} Reset sebagian: {str(e)}")
                with ui_components['status']: 
                    clear_output()
                    display(create_status_indicator("warning", f"{ICONS['warning']} Reset sebagian: {str(e)}"))
                if logger: logger.warning(f"{ICONS['warning']} Error saat reset konfigurasi: {str(e)}")
        else:
            # Tampilkan pesan jika fungsi-fungsi tidak tersedia
            update_status_panel(ui_components, "info", f"{ICONS['info']} UI direset ke kondisi awal")
            with ui_components['status']: 
                clear_output()
                display(create_status_indicator("info", f"{ICONS['info']} UI direset ke kondisi awal"))
    
    # Function untuk reset UI ke kondisi default
    def reset_ui():
        """Reset komponen UI ke kondisi default."""
        # Reset tombol dan progress
        cleanup_ui()
        
        # Reset containers
        for component in ['summary_container', 'visualization_container']:
            if component in ui_components:
                ui_components[component].layout.display = 'none'
                with ui_components[component]: clear_output()
        
        # Hide buttons
        for btn in ['visualization_buttons', 'cleanup_button']:
            if btn in ui_components: ui_components[btn].layout.display = 'none'
        
        # Reset logs dan accordion
        if 'status' in ui_components:
            with ui_components['status']: clear_output()
        if 'log_accordion' in ui_components:
            ui_components['log_accordion'].selected_index = None
    
    # Function untuk cleanup UI setelah processing
    def cleanup_ui():
        """Kembalikan UI ke kondisi operasional setelah processing."""
        # Kembalikan tombol proses dan sembunyikan tombol stop
        process_button_key = 'preprocess_button' if module_type == 'preprocessing' else 'augment_button'
        stop_button_key = 'stop_button'
        
        if process_button_key in ui_components:
            ui_components[process_button_key].layout.display = 'block'
        if stop_button_key in ui_components:
            ui_components[stop_button_key].layout.display = 'none'
        
        # Enable kembali semua tombol
        for btn in ['save_button', 'reset_button', 'cleanup_button', process_button_key]:
            if btn in ui_components:
                ui_components[btn].disabled = False
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        else:
            # Fallback jika fungsi reset tidak tersedia
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 0
                ui_components['progress_bar'].layout.visibility = 'hidden'
            if 'current_progress' in ui_components:
                ui_components['current_progress'].value = 0
                ui_components['current_progress'].layout.visibility = 'hidden'
        
        # Reset flag running
        process_running_key = 'preprocessing_running' if module_type == 'preprocessing' else 'augmentation_running'
        ui_components[process_running_key] = False
    
    # Register handler untuk tombol reset
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(on_reset_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_reset_click': on_reset_click,
        'reset_ui': reset_ui,
        'cleanup_ui': cleanup_ui
    })
    
    return ui_components