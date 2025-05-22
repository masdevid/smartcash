"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler untuk pembersihan hasil augmentasi
"""

from typing import Dict, Any
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message, setup_ui_logger
from smartcash.ui.dataset.augmentation.utils.ui_state_manager import update_status_panel, ensure_confirmation_area

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler untuk tombol cleanup augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget (opsional)
    """
    # Setup logger jika belum
    ui_components = setup_ui_logger(ui_components)
    
    # Nonaktifkan tombol selama proses
    if button and hasattr(button, 'disabled'):
        button.disabled = True
    
    try:
        update_status_panel(ui_components, '🧹 Mempersiapkan pembersihan hasil augmentasi...', 'info')
        
        # Dapatkan direktori output
        output_dir = _get_output_directory(ui_components)
        
        if not output_dir:
            log_message(ui_components, "Direktori output tidak ditemukan", "error", "❌")
            return
        
        # Tampilkan konfirmasi
        _show_cleanup_confirmation(ui_components, output_dir)
            
    except Exception as e:
        log_message(ui_components, f"Error saat persiapan cleanup: {str(e)}", "error", "❌")
    finally:
        if button and hasattr(button, 'disabled'):
            button.disabled = False

def _get_output_directory(ui_components: Dict[str, Any]) -> str:
    """Dapatkan direktori output augmentasi."""
    if 'output_dir' in ui_components and hasattr(ui_components['output_dir'], 'value'):
        return ui_components['output_dir'].value
    elif 'config' in ui_components:
        return ui_components['config'].get('augmentation', {}).get('output_dir', 'data/augmented')
    return 'data/augmented'

def _show_cleanup_confirmation(ui_components: Dict[str, Any], output_dir: str) -> None:
    """Tampilkan dialog konfirmasi untuk cleanup."""
    from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
    from IPython.display import display
    
    message = f"Anda akan menghapus hasil augmentasi di {output_dir}. "
    message += "Tindakan ini tidak dapat dibatalkan. Lanjutkan?"
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        log_message(ui_components, "Konfirmasi cleanup diterima", "info", "✅")
        _execute_cleanup(ui_components, output_dir)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        log_message(ui_components, "Cleanup dibatalkan", "info", "❌")
        update_status_panel(ui_components, '❌ Cleanup dibatalkan', 'info')
    
    ui_components = ensure_confirmation_area(ui_components)
    
    dialog = create_confirmation_dialog(
        title="Konfirmasi Cleanup Augmentasi",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)
    
    update_status_panel(ui_components, '⚠️ Silakan konfirmasi untuk melanjutkan cleanup', 'warning')

def _execute_cleanup(ui_components: Dict[str, Any], output_dir: str) -> None:
    """Eksekusi proses cleanup."""
    import os
    import shutil
    from tqdm.notebook import tqdm
    
    try:
        update_status_panel(ui_components, f'🧹 Membersihkan {output_dir}...', 'info')
        
        if not os.path.exists(output_dir):
            log_message(ui_components, f"Direktori {output_dir} tidak ditemukan", "warning", "⚠️")
            return
        
        # Hitung file yang akan dihapus
        total_files = sum([len(files) for _, _, files in os.walk(output_dir)])
        
        if total_files == 0:
            log_message(ui_components, "Tidak ada file yang perlu dibersihkan", "info", "ℹ️")
            return
        
        # Hapus dengan progress
        with tqdm(total=total_files, desc="🗑️ Menghapus file", colour='red', unit='file') as pbar:
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        pbar.update(1)
                    except Exception as e:
                        log_message(ui_components, f"Gagal menghapus {file_path}: {str(e)}", "warning", "⚠️")
                
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except Exception:
                        pass
        
        log_message(ui_components, f"Berhasil membersihkan {total_files} file", "success", "✅")
        update_status_panel(ui_components, "✅ Cleanup berhasil", "success")
        
    except Exception as e:
        log_message(ui_components, f"Error saat cleanup: {str(e)}", "error", "❌")
        update_status_panel(ui_components, f"❌ Error cleanup: {str(e)}", "error")