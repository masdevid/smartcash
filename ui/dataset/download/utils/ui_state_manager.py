"""
File: smartcash/ui/dataset/download/utils/ui_state_manager.py
Deskripsi: Utilitas untuk mengelola state UI pada modul download dataset
"""

from typing import Dict, Any, Optional
import time
from smartcash.ui.dataset.download.utils.logger_helper import log_message

def enable_download_button(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Aktifkan tombol download.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    # Aktifkan tombol download jika button adalah widget
    if button and hasattr(button, 'disabled'):
        button.disabled = False
    
    # Aktifkan tombol download dari ui_components
    if 'download_button' in ui_components and hasattr(ui_components['download_button'], 'disabled'):
        ui_components['download_button'].disabled = False
        
def disable_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """
    Nonaktifkan/aktifkan tombol-tombol UI.
    
    Args:
        ui_components: Dictionary komponen UI
        disabled: True untuk nonaktifkan, False untuk aktifkan
    """
    # Daftar tombol yang perlu dinonaktifkan
    button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
    
    # Set status disabled untuk semua tombol
    for key in button_keys:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disabled
            
            # Atur visibilitas tombol jika disabled
            if hasattr(ui_components[key], 'layout'):
                if disabled:
                    # Sembunyikan tombol reset dan cleanup saat proses berjalan
                    if key in ['reset_button', 'cleanup_button']:
                        ui_components[key].layout.display = 'none'
                else:
                    # Tampilkan kembali semua tombol dengan konsisten
                    ui_components[key].layout.display = 'inline-block'

def reset_ui_after_download(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses download selesai."""
    # Aktifkan kembali tombol (fungsi ini juga akan mengatur display='inline-block')
    disable_buttons(ui_components, False)
    
    # Reset progress bar
    from smartcash.ui.dataset.download.utils.progress_manager import reset_progress_bar
    reset_progress_bar(ui_components)
    
    # Sembunyikan progress container setelah beberapa detik
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        # Setelah proses selesai, biarkan progress bar terlihat sebentar
        # kemudian sembunyikan setelah beberapa detik
        try:
            time.sleep(0.5)  # Tunggu sebentar agar pengguna dapat melihat progress selesai
            ui_components['progress_container'].layout.display = 'none'
        except:
            # Abaikan error jika tidak bisa tidur
            pass
    
    # Bersihkan area konfirmasi jika ada
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Update status panel
    update_status_panel(ui_components, 'Download selesai', 'success')
    
    # Log message dengan logger helper
    log_message(ui_components, "Proses download telah selesai", "info", "✅")
    
    # Pastikan log accordion tetap terbuka
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0  # Buka accordion pertama
    
    # Cleanup UI jika tersedia
    if 'cleanup_ui' in ui_components and callable(ui_components['cleanup_ui']):
        ui_components['cleanup_ui'](ui_components)
    elif 'cleanup' in ui_components and callable(ui_components['cleanup']):
        ui_components['cleanup']()
        
    # Set flag download_running ke False jika ada
    ui_components['download_running'] = False

def update_status_panel(ui_components: Dict[str, Any], message: str, status: str) -> None:
    """
    Update status panel dengan cara yang konsisten.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan status
        status: Jenis status (success, info, warning, error)
    """
    if 'status_panel' in ui_components:
        from smartcash.ui.components.status_panel import update_status_panel as update_panel
        update_panel(ui_components['status_panel'], message, status)
    elif 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
        ui_components['update_status_panel'](ui_components, status, f'{"✅" if status == "success" else "ℹ️"} {message}')
    
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

def toggle_input_disabled(ui_components: Dict[str, Any], disabled: bool) -> None:
    """
    Toggle status disabled untuk input widgets.
    
    Args:
        ui_components: Dictionary komponen UI
        disabled: True untuk menonaktifkan, False untuk mengaktifkan
    """
    # Daftar input field yang akan di-toggle
    input_fields = [
        'workspace', 'project', 'version', 'api_key', 
        'output_dir', 'backup_dir', 'backup_checkbox', 
        'validate_dataset', 'save_logs'
    ]
    
    # Set status disabled untuk semua input fields
    for field in input_fields:
        if field in ui_components and hasattr(ui_components[field], 'disabled'):
            ui_components[field].disabled = disabled

def highlight_required_fields(ui_components: Dict[str, Any], highlight: bool = True) -> None:
    """
    Highlight required fields yang belum diisi.
    
    Args:
        ui_components: Dictionary komponen UI
        highlight: True untuk highlight, False untuk menghapus highlight
    """
    # Daftar field yang wajib diisi
    required_fields = ['workspace', 'project', 'version', 'api_key', 'output_dir']
    
    for field in required_fields:
        if field in ui_components and hasattr(ui_components[field], 'layout'):
            if highlight:
                # Cek apakah field kosong
                if not ui_components[field].value:
                    # Highlight dengan border merah
                    ui_components[field].layout.border = '1px solid red'
                else:
                    # Hapus highlight
                    ui_components[field].layout.border = ''
            else:
                # Hapus semua highlight
                ui_components[field].layout.border = ''

def get_ui_values(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dapatkan nilai dari semua komponen UI yang penting.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary nilai komponen UI
    """
    values = {}
    
    # Daftar field yang nilainya akan diambil
    fields = [
        'workspace', 'project', 'version', 'api_key', 
        'output_dir', 'backup_dir', 'backup_checkbox', 
        'validate_dataset', 'save_logs'
    ]
    
    for field in fields:
        if field in ui_components and hasattr(ui_components[field], 'value'):
            values[field] = ui_components[field].value
    
    return values 