"""
File: smartcash/ui/dataset/augmentation/utils/ui_state_manager.py
Deskripsi: Manager state UI untuk augmentasi dataset (lengkap)
"""

import time
from typing import Dict, Any, Optional
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

def disable_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """Nonaktifkan/aktifkan tombol-tombol UI."""
    button_keys = ['augment_button', 'reset_button', 'cleanup_button', 'save_button']
    
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disabled
            
            if hasattr(ui_components[key], 'layout'):
                if disabled and key in ['reset_button', 'cleanup_button']:
                    ui_components[key].layout.display = 'none'
                elif not disabled:
                    ui_components[key].layout.display = 'inline-block'

def update_ui_before_augmentation(ui_components: Dict[str, Any]) -> None:
    """Update UI sebelum proses augmentasi dimulai."""
    # Nonaktifkan semua tombol kecuali stop
    disable_buttons(ui_components, True)
    
    # Tampilkan tombol stop, sembunyikan tombol augment
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
        ui_components['stop_button'].layout.display = 'block'
        ui_components['stop_button'].disabled = False
    
    if 'augment_button' in ui_components and hasattr(ui_components['augment_button'], 'layout'):
        ui_components['augment_button'].layout.display = 'none'
    
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
    
    # Update status
    update_status_panel(ui_components, 'Mempersiapkan augmentasi...', 'info')
    log_message(ui_components, "UI dipersiapkan untuk augmentasi", "info", "🔧")

def reset_ui_after_augmentation(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses augmentasi selesai."""
    # Aktifkan kembali semua tombol
    disable_buttons(ui_components, False)
    
    # Sembunyikan tombol stop, tampilkan tombol augment
    if 'stop_button' in ui_components and hasattr(ui_components['stop_button'], 'layout'):
        ui_components['stop_button'].layout.display = 'none'
    
    if 'augment_button' in ui_components and hasattr(ui_components['augment_button'], 'layout'):
        ui_components['augment_button'].layout.display = 'block'
        ui_components['augment_button'].disabled = False
    
    # Reset progress bar
    from smartcash.ui.dataset.augmentation.utils.progress_manager import reset_progress_bar
    reset_progress_bar(ui_components)
    
    # Sembunyikan progress container setelah delay singkat
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'none'
    
    # Bersihkan area konfirmasi
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
    
    # Set flag augmentation_running ke False
    ui_components['augmentation_running'] = False
    ui_components['stop_requested'] = False
    
    log_message(ui_components, "UI berhasil direset setelah augmentasi", "info", "✅")

def is_augmentation_running(ui_components: Dict[str, Any]) -> bool:
    """Cek apakah augmentasi sedang berjalan."""
    return ui_components.get('augmentation_running', False)

def set_augmentation_state(ui_components: Dict[str, Any], running: bool) -> None:
    """Set state augmentasi running."""
    ui_components['augmentation_running'] = running
    if not running:
        ui_components['stop_requested'] = False
    
    log_message(ui_components, f"State augmentasi: {'🟢 Running' if running else '🔴 Stopped'}", "debug")

def update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel dengan pesan."""
    if 'status_panel' in ui_components:
        from smartcash.ui.components.status_panel import update_status_panel as update_panel
        update_panel(ui_components['status_panel'], message, status)
    elif 'update_status_panel' in ui_components:
        ui_components['update_status_panel'](ui_components, status, f'{"✅" if status == "success" else "ℹ️"} {message}')

def show_confirmation(ui_components: Dict[str, Any], title: str, message: str, on_confirm_callback, on_cancel_callback) -> None:
    """Tampilkan dialog konfirmasi."""
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    from IPython.display import display
    
    # Pastikan ada confirmation_area
    ui_components = ensure_confirmation_area(ui_components)
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        on_confirm_callback()
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        on_cancel_callback()
    
    dialog = create_confirmation_dialog(
        title=title,
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def ensure_confirmation_area(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Pastikan UI memiliki area konfirmasi yang valid."""
    if 'confirmation_area' not in ui_components:
        from ipywidgets import Output
        ui_components['confirmation_area'] = Output()
        log_message(ui_components, "Area konfirmasi dibuat otomatis", "info", "ℹ️")
        
        # Tambahkan ke UI jika memungkinkan
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                children = list(ui_components['ui'].children)
                children.append(ui_components['confirmation_area'])
                ui_components['ui'].children = tuple(children)
            except Exception as e:
                log_message(ui_components, f"Tidak bisa menambahkan area konfirmasi: {str(e)}", "warning", "⚠️")
    
    return ui_components

def reset_after_operation(ui_components: Dict[str, Any], operation_name: str = "operasi") -> None:
    """Reset UI setelah operasi selesai."""
    reset_ui_after_augmentation(ui_components)
    update_status_panel(ui_components, f"{operation_name.capitalize()} selesai", "success")
    log_message(ui_components, f"Reset UI setelah {operation_name}", "info", "🔄")

def update_ui_state(ui_components: Dict[str, Any], state: str, message: str = "") -> None:
    """Update state UI secara umum."""
    if state == "running":
        update_ui_before_augmentation(ui_components)
    elif state == "complete":
        reset_ui_after_augmentation(ui_components)
        if message:
            update_status_panel(ui_components, message, "success")
    elif state == "error":
        reset_ui_after_augmentation(ui_components)
        if message:
            update_status_panel(ui_components, message, "error")
    elif state == "cancelled":
        reset_ui_after_augmentation(ui_components)
        update_status_panel(ui_components, "Operasi dibatalkan", "warning")