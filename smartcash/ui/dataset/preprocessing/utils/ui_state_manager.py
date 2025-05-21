"""
File: smartcash/ui/dataset/preprocessing/utils/ui_state_manager.py
Deskripsi: Utilitas untuk mengelola state UI preprocessing
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
from smartcash.ui.utils.constants import ICONS, COLORS

def update_status_panel(ui_components: Dict[str, Any], status: str, message: str = None) -> None:
    """
    Update status panel UI dengan status dan pesan.
    
    Args:
        ui_components: Dictionary komponen UI
        status: Status preprocessing (idle, running, success, error)
        message: Pesan status opsional
    """
    if 'status_panel' not in ui_components:
        return
    
    status_panel = ui_components['status_panel']
    
    # Map status ke warna dan icon
    status_map = {
        'idle': {'color': 'info', 'icon': ICONS['idle']},
        'running': {'color': 'primary', 'icon': ICONS['running']},
        'success': {'color': 'success', 'icon': ICONS['success']},
        'error': {'color': 'danger', 'icon': ICONS['error']},
        'warning': {'color': 'warning', 'icon': ICONS['warning']},
    }
    
    # Default ke idle jika status tidak valid
    status_info = status_map.get(status, status_map['idle'])
    
    # Format pesan
    if message:
        formatted_message = f"{status_info['icon']} {message}"
    else:
        default_messages = {
            'idle': "Preprocessing siap dijalankan",
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
    
    # Log reset
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message']("UI telah direset ke state awal", "info", "🔄")

def update_ui_state(ui_components: Dict[str, Any], state: str, message: Optional[str] = None) -> None:
    """
    Update seluruh UI berdasarkan state.
    
    Args:
        ui_components: Dictionary komponen UI
        state: State baru (idle, running, success, error)
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
            
    elif state in ['idle', 'success', 'error']:
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
    
    # Log state change
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        log_message = f"UI state diubah ke {state}"
        if message:
            log_message += f": {message}"
        ui_components['log_message'](log_message, "debug", "🔄")

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
    update_status_panel(ui_components, "started", "Preprocessing sedang berjalan...")
    
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