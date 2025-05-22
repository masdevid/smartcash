
"""
File: smartcash/ui/dataset/augmentation/utils/ui_state_manager.py
Deskripsi: Manager state UI untuk augmentasi dataset
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

def reset_ui_after_augmentation(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses augmentasi selesai."""
    disable_buttons(ui_components, False)
    
    from smartcash.ui.dataset.augmentation.utils.progress_manager import reset_progress_bar
    reset_progress_bar(ui_components)
    
    # Sembunyikan progress container setelah beberapa detik
    if 'progress_container' in ui_components:
        try:
            time.sleep(0.5)
            ui_components['progress_container'].layout.display = 'none'
        except:
            pass
    
    # Bersihkan area konfirmasi
    if 'confirmation_area' in ui_components:
        ui_components['confirmation_area'].clear_output()
    
    update_status_panel(ui_components, 'Augmentasi selesai', 'success')
    log_message(ui_components, "Proses augmentasi telah selesai", "info", "✅")
    
    # Set flag augmentation_running ke False
    ui_components['augmentation_running'] = False

def update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """Update status panel dengan pesan."""
    if 'status_panel' in ui_components:
        from smartcash.ui.components.status_panel import update_status_panel as update_panel
        update_panel(ui_components['status_panel'], message, status)
    elif 'update_status_panel' in ui_components:
        ui_components['update_status_panel'](ui_components, status, f'{"✅" if status == "success" else "ℹ️"} {message}')

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