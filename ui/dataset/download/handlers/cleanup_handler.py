"""
File: smartcash/ui/dataset/download/handlers/cleanup_handler.py
Deskripsi: Handler untuk tombol cleanup dataset
"""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
import threading

from smartcash.dataset.manager import DatasetManager
from smartcash.ui.dataset.download.utils.notification_manager import notify_log, notify_progress
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
from IPython.display import display

def handle_cleanup_button_click(ui_components: Dict[str, Any], button=None) -> None:
    """
    Handler untuk tombol cleanup dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Tombol yang diklik
    """
    # Disable tombol untuk mencegah multiple click
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        # Dapatkan path dataset dari UI
        output_dir = ui_components.get('output_dir', {}).value
        
        if not output_dir:
            notify_log(
                sender=ui_components,
                message="Direktori output tidak ditentukan",
                level="error"
            )
            update_status_panel(
                ui_components['status_panel'],
                "Direktori output tidak ditentukan",
                "error"
            )
            return
        
        # Tampilkan dialog konfirmasi
        confirm_cleanup(ui_components, output_dir, button)
        
    except Exception as e:
        notify_log(
            sender=ui_components,
            message=f"Error saat cleanup dataset: {str(e)}",
            level="error"
        )
        update_status_panel(
            ui_components['status_panel'],
            f"Error saat cleanup dataset: {str(e)}",
            "error"
        )
    finally:
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def confirm_cleanup(ui_components: Dict[str, Any], output_dir: str, button=None) -> bool:
    """
    Tampilkan dialog konfirmasi untuk cleanup dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Direktori output yang akan dihapus
        button: Tombol yang diklik
        
    Returns:
        bool: True jika konfirmasi berhasil ditampilkan
    """
    # Pastikan output_dir valid
    output_path = Path(output_dir)
    
    # Cek apakah direktori ada
    if not output_path.exists():
        notify_log(
            sender=ui_components,
            message=f"Direktori tidak ditemukan: {output_dir}",
            level="error"
        )
        update_status_panel(
            ui_components['status_panel'],
            f"Direktori tidak ditemukan: {output_dir}",
            "error"
        )
        return False
    
    # Buat pesan konfirmasi
    message = f"Anda akan menghapus dataset di direktori {output_dir}. "
    message += "Tindakan ini tidak dapat dibatalkan. "
    message += "Apakah Anda yakin ingin melanjutkan?"
    
    # Fungsi untuk menjalankan cleanup dan membersihkan dialog
    def confirm_and_execute():
        # Bersihkan area konfirmasi
        ui_components['confirmation_area'].clear_output()
        
        # Jalankan cleanup di thread terpisah
        thread = threading.Thread(
            target=execute_cleanup,
            args=(ui_components, output_dir)
        )
        thread.daemon = True
        thread.start()
    
    # Fungsi untuk membatalkan cleanup
    def cancel_cleanup():
        ui_components['confirmation_area'].clear_output()
        notify_log(
            sender=ui_components,
            message="Cleanup dataset dibatalkan",
            level="info"
        )
        update_status_panel(
            ui_components['status_panel'],
            "Cleanup dataset dibatalkan",
            "info"
        )
        
        # Re-enable tombol
        if button and hasattr(button, 'disabled'):
            button.disabled = False
    
    # Tampilkan dialog konfirmasi
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        dialog = create_confirmation_dialog(
            title="Konfirmasi Hapus Dataset", 
            message=message,
            on_confirm=confirm_and_execute,
            on_cancel=cancel_cleanup
        )
        display(dialog)
    
    # Update status panel
    update_status_panel(
        ui_components['status_panel'],
        "Silakan konfirmasi untuk menghapus dataset",
        "warning"
    )
    
    return True

def execute_cleanup(ui_components: Dict[str, Any], output_dir: str) -> None:
    """
    Jalankan proses cleanup dataset.
    
    Args:
        ui_components: Dictionary komponen UI
        output_dir: Direktori output yang akan dihapus
    """
    # Set flag bahwa cleanup sedang berjalan
    ui_components['cleanup_running'] = True
    
    # Disable semua tombol
    _disable_buttons(ui_components, True)
    
    # Tampilkan progress
    _show_progress(ui_components, "Mempersiapkan penghapusan dataset...")
    
    # Reset progress bar
    _reset_progress_bar(ui_components)
    
    try:
        # Inisialisasi DatasetManager
        dataset_manager = DatasetManager()
        
        # Log bahwa kita akan menghapus dataset
        notify_log(
            sender=ui_components,
            message=f"Menghapus dataset di direktori: {output_dir}",
            level="info"
        )
        
        # Update progress
        _update_progress(ui_components, 10, "Memulai penghapusan dataset...")
        
        # Jalankan cleanup
        result = dataset_manager.cleanup_dataset(
            output_dir,
            backup_before_delete=True,
            show_progress=True
        )
        
        # Update progress
        _update_progress(ui_components, 100, "Dataset berhasil dihapus")
        
        # Log hasil
        if result["status"] == "success":
            notify_log(
                sender=ui_components,
                message=f"Dataset berhasil dihapus: {output_dir}",
                level="success"
            )
            update_status_panel(
                ui_components['status_panel'],
                "Dataset berhasil dihapus",
                "success"
            )
        else:
            notify_log(
                sender=ui_components,
                message=f"Gagal menghapus dataset: {result['message']}",
                level="error"
            )
            update_status_panel(
                ui_components['status_panel'],
                "Gagal menghapus dataset",
                "error"
            )
    except Exception as e:
        notify_log(
            sender=ui_components,
            message=f"Error saat cleanup dataset: {str(e)}",
            level="error"
        )
        update_status_panel(
            ui_components['status_panel'],
            f"Error saat cleanup dataset",
            "error"
        )
    finally:
        # Reset UI
        _reset_ui_after_cleanup(ui_components)

def _reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """Reset progress bar ke kondisi awal."""
    # Reset progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].description = "Progress: 0%"
        ui_components['progress_bar'].layout.visibility = 'hidden'
    
    # Reset label
    for key in ['overall_label', 'step_label', 'current_progress']:
        if key in ui_components and hasattr(ui_components[key], 'value'):
            ui_components[key].value = ""
            ui_components[key].layout.visibility = 'hidden'
    
    # Reset current progress
    if 'current_progress' in ui_components and hasattr(ui_components['current_progress'], 'value'):
        ui_components['current_progress'].value = 0
        ui_components['current_progress'].description = "Step 0/0"
        ui_components['current_progress'].layout.visibility = 'hidden'

def _show_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan progress container dan set pesan."""
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
    # Set flag bahwa cleanup sedang berjalan
    ui_components['cleanup_running'] = True
    
    # Pastikan log accordion terbuka
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0

def _update_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update progress bar dan pesan."""
    # Tampilkan progress container
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
    # Update progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        try:
            # Pastikan progress adalah integer
            progress_value = int(float(progress))
            ui_components['progress_bar'].value = progress_value
            ui_components['progress_bar'].description = f"Progress: {progress_value}%"
            ui_components['progress_bar'].layout.visibility = 'visible'
        except (ValueError, TypeError):
            pass
    
    # Update pesan
    if 'overall_label' in ui_components and hasattr(ui_components['overall_label'], 'value'):
        ui_components['overall_label'].value = message
        ui_components['overall_label'].layout.visibility = 'visible'
    
    # Notifikasi progress
    notify_progress(
        sender=ui_components,
        event_type="update",
        progress=progress,
        total=100,
        message=message
    )

def _disable_buttons(ui_components: Dict[str, Any], disable: bool) -> None:
    """Disable atau enable tombol."""
    # Disable atau enable tombol
    for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
        if key in ui_components and hasattr(ui_components[key], 'disabled'):
            ui_components[key].disabled = disable
            
            # Sembunyikan tombol reset dan cleanup jika disable
            if key in ['reset_button', 'cleanup_button'] and hasattr(ui_components[key], 'layout'):
                ui_components[key].layout.display = 'none' if disable else 'inline-block'

def _reset_ui_after_cleanup(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah cleanup selesai."""
    # Enable tombol
    _disable_buttons(ui_components, False)
    
    # Reset progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = 0
    
    # Bersihkan area konfirmasi
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Reset flag
    ui_components['cleanup_running'] = False
