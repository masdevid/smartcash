"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk modul konfigurasi environment dengan implementasi DRY
"""

from typing import Dict, Any
from smartcash.ui.utils.base_initializer import initialize_module_ui
from smartcash.ui.setup.env_config_component import create_env_config_ui
from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers
from smartcash.ui.utils.ui_logger import auto_intercept_when_ready, log_to_ui

# Helper function untuk disable UI selama processing
def _disable_ui_during_processing(ui_components: Dict[str, Any], disable: bool = True) -> None:
    """Menonaktifkan atau mengaktifkan komponen UI selama proses berjalan."""
    # Daftar komponen yang perlu dinonaktifkan
    disable_components = [
        'drive_button', 'directory_button', 'check_button', 'save_button'
    ]
    
    # Disable/enable komponen
    for component in disable_components:
        if component in ui_components and hasattr(ui_components[component], 'disabled'):
            ui_components[component].disabled = disable

# Helper function untuk cleanup UI
def _cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """Membersihkan UI setelah proses selesai."""
    # Aktifkan kembali UI
    _disable_ui_during_processing(ui_components, False)
    
    # Reset progress bar jika ada
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'layout'):
        ui_components['progress_bar'].layout.visibility = 'hidden'
    if 'progress_message' in ui_components and hasattr(ui_components['progress_message'], 'layout'):
        ui_components['progress_message'].layout.visibility = 'hidden'

def setup_env_config_specific(ui_components: Dict[str, Any], env: Any, config: Any) -> Dict[str, Any]:
    """Setup handler spesifik untuk env config dengan tambahan fungsi helper"""
    # Setup handler
    ui_components = setup_env_config_handlers(ui_components, env, config)
    
    # Tambahkan fungsi helper
    ui_components.update({
        'disable_ui_during_processing': _disable_ui_during_processing,
        'cleanup_ui': _cleanup_ui
    })
    
    return ui_components

def initialize_env_config_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk konfigurasi environment.
    
    Returns:
        Dictionary komponen UI yang terinisialisasi
    """
    # Tombol yang perlu diattach dengan ui_components
    button_keys = ['save_button', 'reset_button', 'check_button']
    
    # Gunakan base initializer
    ui_components = initialize_module_ui(
        module_name='env_config',
        create_ui_func=create_env_config_ui,
        setup_specific_handlers_func=setup_env_config_specific,
        button_keys=button_keys
    )
    
    # Aktifkan UI logger saat UI sudah siap
    auto_intercept_when_ready(ui_components)
    
    # Log informasi inisialisasi
    log_to_ui(ui_components, "Environment config UI berhasil diinisialisasi", "success", "✅")
    
    return ui_components