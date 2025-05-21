"""
File: smartcash/ui/dataset/preprocessing/utils/ui_state_manager.py
Deskripsi: Utilitas untuk mengelola state UI preprocessing
"""

from typing import Dict, Any, Optional, List, Callable
import time
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.utils.constants import ICONS, COLORS
from IPython.display import display

def update_status_panel(ui_components: Dict[str, Any], status: str, message: str = None) -> None:
    """
    Update status panel UI dengan status dan pesan.
    
    Args:
        ui_components: Dictionary komponen UI
        status: Status preprocessing (idle, running, success, error, warning)
        message: Pesan status opsional
    """
    if 'status_panel' not in ui_components:
        return
    
    status_panel = ui_components['status_panel']
    
    # Map status ke warna dan icon
    status_map = {
        'idle': {'color': 'info', 'icon': ICONS['idle']},
        'info': {'color': 'info', 'icon': ICONS['info']},
        'running': {'color': 'primary', 'icon': ICONS['running']},
        'success': {'color': 'success', 'icon': ICONS['success']},
        'error': {'color': 'danger', 'icon': ICONS['error']},
        'warning': {'color': 'warning', 'icon': ICONS['warning']},
    }
    
    # Default ke info jika status tidak valid
    status_info = status_map.get(status, status_map['info'])
    
    # Format pesan
    if message:
        formatted_message = f"{status_info['icon']} {message}"
    else:
        default_messages = {
            'idle': "Preprocessing siap dijalankan",
            'info': "Informasi preprocessing",
            'running': "Preprocessing sedang berjalan...",
            'success': "Preprocessing berhasil diselesaikan",
            'error': "Terjadi kesalahan saat preprocessing",
            'warning': "Perhatian diperlukan",
        }
        formatted_message = f"{status_info['icon']} {default_messages.get(status, 'Status tidak diketahui')}"
    
    # Update panel
    if hasattr(status_panel, 'type') and hasattr(status_panel, 'message'):
        status_panel.type = status_info['color']
        status_panel.message = formatted_message
    else:
        # Alternatif jika status_panel tidak memiliki atribut yang diharapkan
        from ipywidgets import HTML
        if isinstance(status_panel, HTML):
            color = COLORS.get(status_info['color'], COLORS['dark'])
            status_panel.value = f"<div style='color: {color};'>{formatted_message}</div>"

def reset_ui_after_preprocessing(ui_components: Dict[str, Any]) -> None:
    """
    Reset UI ke state awal setelah preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = 0
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Reset progress labels
    for label_key in ['overall_label', 'step_label', 'current_progress']:
        if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
            ui_components[label_key].value = ""
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'hidden'
    
    # Enable preprocessing button
    if 'preprocess_button' in ui_components:
        ui_components['preprocess_button'].disabled = False
    
    # Disable stop button
    if 'stop_button' in ui_components:
        ui_components['stop_button'].disabled = True
    
    # Update status panel
    update_status_panel(ui_components, 'idle')
    
    # Bersihkan area konfirmasi
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Log reset
    log_message(ui_components, "UI telah direset ke state awal", "info", "🔄")

def update_ui_state(ui_components: Dict[str, Any], state: str, message: Optional[str] = None) -> None:
    """
    Update seluruh UI berdasarkan state.
    
    Args:
        ui_components: Dictionary komponen UI
        state: State baru (idle, running, success, error, warning, info)
        message: Pesan opsional untuk ditampilkan
    """
    # Update status panel
    update_status_panel(ui_components, state, message)
    
    # Handle button states
    if state == 'running':
        # Disable preprocessing button
        if 'preprocess_button' in ui_components:
            ui_components['preprocess_button'].disabled = True
        
        # Enable stop button
        if 'stop_button' in ui_components:
            ui_components['stop_button'].disabled = False
            
        # Disable save button
        if 'save_button' in ui_components:
            ui_components['save_button'].disabled = True
            
        # Disable reset button
        if 'reset_button' in ui_components:
            ui_components['reset_button'].disabled = True
            
        # Disable cleanup button
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].disabled = True
            
    elif state in ['idle', 'success', 'error', 'warning', 'info']:
        # Enable preprocessing button
        if 'preprocess_button' in ui_components:
            ui_components['preprocess_button'].disabled = False
        
        # Disable stop button
        if 'stop_button' in ui_components:
            ui_components['stop_button'].disabled = True
            
        # Enable save button
        if 'save_button' in ui_components:
            ui_components['save_button'].disabled = False
            
        # Enable reset button
        if 'reset_button' in ui_components:
            ui_components['reset_button'].disabled = False
            
        # Enable cleanup button
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].disabled = False
    
    # Log state change
    log_message(ui_components, f"UI state diubah ke {state}{': ' + message if message else ''}", "debug", "🔄")

def update_ui_before_preprocessing(ui_components: Dict[str, Any]) -> None:
    """
    Update UI components sebelum preprocessing dimulai.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Update button states
    if 'preprocess_button' in ui_components and hasattr(ui_components['preprocess_button'], 'disabled'):
        ui_components['preprocess_button'].disabled = True
    
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'disabled'):
        ui_components['stop_button'].disabled = False
    
    if 'reset_button' in ui_components and hasattr(ui_components['reset_button'], 'disabled'):
        ui_components['reset_button'].disabled = True
    
    if 'cleanup_button' in ui_components and hasattr(ui_components['cleanup_button'], 'disabled'):
        ui_components['cleanup_button'].disabled = True
    
    if 'save_button' in ui_components and hasattr(ui_components['save_button'], 'disabled'):
        ui_components['save_button'].disabled = True
    
    # Update flags
    ui_components['preprocessing_running'] = True
    
    # Update status
    update_status_panel(ui_components, "running", "Preprocessing sedang berjalan...")
    
    # Log update berhasil
    log_message(ui_components, "UI dipersiapkan untuk preprocessing", "debug", "🔄")

def is_preprocessing_running(ui_components: Dict[str, Any]) -> bool:
    """
    Cek apakah preprocessing sedang berjalan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        bool: True jika preprocessing sedang berjalan, False jika tidak
    """
    return ui_components.get('preprocessing_running', False)

def set_preprocessing_state(ui_components: Dict[str, Any], running: bool) -> None:
    """
    Set state preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        running: State running baru
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Update flag
    ui_components['preprocessing_running'] = running
    
    # Update UI berdasarkan state
    if running:
        update_ui_before_preprocessing(ui_components)
    else:
        reset_ui_after_preprocessing(ui_components)

def toggle_input_controls(ui_components: Dict[str, Any], disabled: bool) -> None:
    """
    Toggle control input berdasarkan state.
    
    Args:
        ui_components: Dictionary komponen UI
        disabled: True untuk disable controls, False untuk enable
    """
    # Skip jika ui_components tidak valid
    if not isinstance(ui_components, dict):
        return
    
    # Identifikasi semua input controls
    input_controls = []
    
    # Tambahkan semua widget yang memiliki atribut disabled
    for key, widget in ui_components.items():
        if hasattr(widget, 'disabled') and key not in [
            'preprocess_button', 'stop_button', 'reset_button', 'cleanup_button', 'save_button'
        ]:
            input_controls.append(widget)
    
    # Toggle disabled state untuk semua controls
    for control in input_controls:
        control.disabled = disabled

def ensure_confirmation_area(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pastikan UI memiliki area konfirmasi yang valid.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah ditambahkan area konfirmasi
    """
    # Pastikan kita memiliki UI area untuk konfirmasi
    if 'confirmation_area' not in ui_components:
        from ipywidgets import Output
        ui_components['confirmation_area'] = Output()
        log_message(ui_components, "Area konfirmasi dibuat otomatis", "info", "ℹ️")
        
        # Tambahkan ke UI jika ada area untuk itu
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                # Coba tambahkan ke UI container (bukan UI ideal, tapi berfungsi sebagai fallback)
                children = list(ui_components['ui'].children)
                children.append(ui_components['confirmation_area'])
                ui_components['ui'].children = tuple(children)
            except Exception as e:
                log_message(ui_components, f"Tidak bisa menambahkan area konfirmasi ke UI: {str(e)}", "warning", "⚠️")
    
    return ui_components

def show_confirmation(ui_components: Dict[str, Any], title: str, message: str, 
                     on_confirm: Callable, on_cancel: Callable) -> None:
    """
    Tampilkan dialog konfirmasi.
    
    Args:
        ui_components: Dictionary komponen UI
        title: Judul konfirmasi
        message: Pesan konfirmasi
        on_confirm: Fungsi yang dipanggil saat konfirmasi
        on_cancel: Fungsi yang dipanggil saat pembatalan
    """
    # Pastikan ada area konfirmasi
    ui_components = ensure_confirmation_area(ui_components)
    
    # Import dialog komponen
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    
    # Bersihkan area konfirmasi
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Tampilkan dialog konfirmasi
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title=title, 
            message=message,
            on_confirm=on_confirm,
            on_cancel=on_cancel
        )
        display(dialog)

def reset_after_operation(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Reset UI setelah operasi selesai dan re-enable tombol.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Tombol yang akan diaktifkan kembali
    """
    # Re-enable tombol
    if button and hasattr(button, 'disabled'):
        button.disabled = False
    
    # Reset UI untuk operasi flags
    ui_components['preprocessing_running'] = False
    ui_components['cleanup_running'] = False
    ui_components['stop_requested'] = False
    
    # Update status panel
    update_status_panel(ui_components, 'idle', "Siap untuk memulai preprocessing baru")
    
    # Bersihkan area konfirmasi
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output() 