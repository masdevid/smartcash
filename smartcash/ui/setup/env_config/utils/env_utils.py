"""
File: smartcash/ui/utils/env_ui_utils.py
Deskripsi: Utilitas UI khusus untuk environment config - mengkonsolidasi helper functions
"""

from typing import Dict, Any, Optional
from IPython.display import display

# Module logger name untuk konsistensi
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[ENV_CONFIG_LOGGER_NAMESPACE]

def update_status(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """
    Update status panel dengan pesan dan tipe yang tepat
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    if 'status_panel' in ui_components:
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(ui_components['status_panel'], message, status_type)

def update_progress(ui_components: Dict[str, Any], value: float, message: str = "") -> None:
    """
    Update progress bar dan message
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0.0 - 1.0)
        message: Pesan progress opsional
    """
    if 'progress_bar' in ui_components:
        from smartcash.ui.components.progress_tracking import update_progress as update_progress_component
        update_progress_component(
            ui_components,
            int(value * 100),  # Convert to percentage
            100,
            message
        )

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """
    Log message ke output panel dengan formatting yang konsisten
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan dilog
        level: Level log (info, success, warning, error)
        icon: Ikon opsional
    """
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_log'):
        # Gunakan append_log method jika tersedia
        ui_components['log_output'].append_log(
            message, 
            level, 
            namespace=ui_components.get('logger_namespace')
        )
    elif 'log_output' in ui_components:
        # Fallback ke display biasa
        from smartcash.ui.utils.alert_utils import create_status_indicator
        with ui_components['log_output']:
            display(create_status_indicator(level, message, icon))

def reset_ui_state(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI state ke kondisi awal
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset progress
    if 'progress_bar' in ui_components:
        from smartcash.ui.components.progress_tracking import reset_progress
        reset_progress(ui_components)
    
    # Reset status
    update_status(ui_components, "Siap untuk mengkonfigurasi environment", "info")
    
    # Clear log output
    if 'log_output' in ui_components:
        ui_components['log_output'].clear_output(wait=True)

def show_environment_summary(ui_components: Dict[str, Any], env_status: Dict[str, Any]) -> None:
    """
    Tampilkan ringkasan status environment
    
    Args:
        ui_components: Dictionary komponen UI
        env_status: Status environment dari check
    """
    # Summary message
    env_type = "Google Colab" if env_status.get('is_colab', False) else "Lokal"
    drive_status = "Terhubung" if env_status.get('drive_connected', False) else "Tidak terhubung"
    ready_status = "Siap" if env_status.get('ready', False) else "Perlu setup"
    
    summary = f"""
    📊 **Ringkasan Environment:**
    - **Tipe**: {env_type}
    - **Google Drive**: {drive_status}
    - **Status**: {ready_status}
    """
    
    log_message(ui_components, summary, "info", "📋")

def handle_ui_error(ui_components: Dict[str, Any], error: Exception, context: str = "") -> None:
    """
    Handle error dengan logging dan status update yang proper
    
    Args:
        ui_components: Dictionary komponen UI
        error: Exception yang terjadi
        context: Konteks dimana error terjadi
    """
    error_msg = f"Error {context}: {str(error)}" if context else f"Error: {str(error)}"
    
    # Update status dan log
    update_status(ui_components, error_msg, "error")
    log_message(ui_components, error_msg, "error", "❌")
    
    # Enable button jika ada
    if 'setup_button' in ui_components:
        ui_components['setup_button'].disabled = False

def validate_ui_components(ui_components: Dict[str, Any]) -> bool:
    """
    Validate bahwa UI components memiliki elemen yang diperlukan
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika valid, False jika tidak
    """
    required_components = ['ui_layout']
    optional_components = ['status_panel', 'log_output', 'progress_bar', 'setup_button']
    
    # Check required components
    for component in required_components:
        if component not in ui_components:
            return False
    
    # Log missing optional components
    missing_optional = [comp for comp in optional_components if comp not in ui_components]
    if missing_optional:
        print(f"⚠️ Optional UI components tidak ditemukan: {', '.join(missing_optional)}")
    
    return True