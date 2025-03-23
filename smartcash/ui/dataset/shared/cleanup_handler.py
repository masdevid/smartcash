"""
File: smartcash/ui/dataset/shared/cleanup_handler.py
Deskripsi: Handler standar pembersihan data untuk komponen preprocessing dan augmentasi
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_cleanup_handler(
    ui_components: Dict[str, Any], 
    folder_to_clean: Optional[str] = None,
    prefixes_to_clean: Optional[List[str]] = None,
    extensions_to_clean: Optional[List[str]] = None,
    confirmation_needed: bool = True,
    update_status_panel_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Setup handler pembersihan data standard.
    
    Args:
        ui_components: Dictionary komponen UI
        folder_to_clean: Direktori utama yang akan dibersihkan (jika None, gunakan dari ui_components)
        prefixes_to_clean: List prefiks file yang akan dibersihkan
        extensions_to_clean: List ekstensi file yang akan dibersihkan
        confirmation_needed: Apakah perlu konfirmasi sebelum menghapus
        update_status_panel_func: Fungsi untuk update status panel
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi untuk menonaktifkan semua tombol saat proses cleanup
    def disable_buttons(disable=True):
        """Nonaktifkan semua tombol saat sedang proses."""
        buttons = [
            'primary_button', 'cleanup_button', 'save_button', 
            'visualize_button', 'compare_button', 'distribution_button'
        ]
        
        for btn_name in buttons:
            if btn_name in ui_components:
                ui_components[btn_name].disabled = disable
    
    # Handler untuk tombol cleanup dengan konfirmasi jika diperlukan
    def on_cleanup_click(b):
        try:
            # Nonaktifkan semua tombol saat proses dimulai
            disable_buttons(True)
            
            # Buat dialog konfirmasi jika diperlukan
            if confirmation_needed:
                try:
                    from smartcash.ui.helpers.ui_helpers import create_confirmation_dialog
                    
                    def on_confirm_cleanup():
                        with ui_components['status']: clear_output(wait=True)
                        perform_cleanup()
                    
                    def on_cancel_cleanup():
                        with ui_components['status']: 
                            clear_output(wait=True)
                            display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
                        # Aktifkan kembali tombol setelah batal
                        disable_buttons(False)
                    
                    # Buat dialog konfirmasi
                    dialog = create_confirmation_dialog(
                        "Konfirmasi Pembersihan Data",
                        "Apakah Anda yakin ingin menghapus data hasil pemrosesan? Tindakan ini tidak dapat dibatalkan.",
                        on_confirm_cleanup, on_cancel_cleanup, "Ya, Hapus Data", "Batal"
                    )
                    
                    with ui_components['status']:
                        clear_output(wait=True)
                        display(dialog)
                    return
                    
                except ImportError:
                    # Lanjutkan tanpa konfirmasi jika fungsi tidak tersedia
                    with ui_components['status']: 
                        display(create_status_indicator(
                            "warning", f"{ICONS['warning']} Konfirmasi tidak tersedia, melanjutkan pembersihan..."
                        ))
                        perform_cleanup()
            else:
                # Langsung lakukan cleanup jika tidak perlu konfirmasi
                perform_cleanup()
                
        except Exception as e:
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
            # Aktifkan kembali tombol jika terjadi error
            disable_buttons(False)
    
    # Fungsi untuk melakukan cleanup dengan progress tracking
    def perform_cleanup():
        # Dapatkan direktori yang akan dibersihkan
        dir_to_clean = folder_to_clean or ui_components.get('preprocessed_dir') or ui_components.get('augmented_dir', 'data/preprocessed')
        
        # Untuk menyimpan status panel
        update_status = update_status_panel_func
        if not update_status and 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            update_status = ui_components['update_status_panel']
        
        try:
            # Tampilkan status proses dimulai
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data..."))
            
            # Update status panel
            if update_status:
                update_status(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data...")
            
            # Notifikasi observer sebelum cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.DATA_CLEANUP_START,
                    sender="cleanup_handler",
                    message=f"Memulai pembersihan data"
                )
            except ImportError:
                pass
            
            # Setup progress tracking
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = 100
                ui_components['progress_bar'].value = 0
                ui_components['progress_bar'].description = "Cleanup: 0%"
                ui_components['progress_bar'].layout.visibility = 'visible'
            
            if 'current_progress' in ui_components:
                ui_components['current_progress'].max = 4  # Tahapan: scan dirs, hapus images, hapus labels, hapus metadata
                ui_components['current_progress'].value = 0
                ui_components['current_progress'].description = "Scanning..."
                ui_components['current_progress'].layout.visibility = 'visible'
            
            # Cari semua file yang perlu dihapus
            paths = Path(dir_to_clean)
            if not paths.exists():
                with ui_components['status']:
                    display(create_status_indicator("warning", 
                        f"{ICONS.get('warning', '⚠️')} Direktori tidak ditemukan: {dir_to_clean}"))
                
                if update_status:
                    update_status(ui_components, "warning", 
                        f"{ICONS.get('warning', '⚠️')} Direktori tidak ditemukan: {dir_to_clean}")
                
                cleanup_ui()
                return
            
            # Step 1: Scan direktori (25%)
            update_cleanup_progress(1, 4, "Scanning direktori...")
            
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 25
                ui_components['progress_bar'].description = "Cleanup: 25%"
            
            # Catat file-file yang akan dihapus
            files_to_delete = []
            
            # Jika diberikan folder utama (preprocessed/augmented), cari di subdirectory
            for split in ['train', 'valid', 'test']:
                for subdir in ['images', 'labels']:
                    dir_path = paths / split / subdir
                    if not dir_path.exists():
                        continue
                    
                    # Tambahkan file berdasarkan prefiks
                    if prefixes_to_clean:
                        for prefix in prefixes_to_clean:
                            # Untuk setiap ekstensi, tambahkan file dengan prefiks yang sesuai
                            exts = extensions_to_clean or ['*.jpg', '*.png', '*.txt', '*.npy']
                            for ext in exts:
                                prefix_pattern = f"{prefix}_*{ext}"
                                files_to_delete.extend(list(dir_path.glob(prefix_pattern)))
                    else:
                        # Jika tidak ada prefiks, hapus semua file (hati-hati!)
                        files_to_delete.extend(list(dir_path.glob('*.*')))
            
            # Jika tidak ada file yang ditemukan di struktur split, coba cari di root folder (untuk augmentation temp)
            if not files_to_delete:
                for subdir in ['images', 'labels']:
                    dir_path = paths / subdir
                    if not dir_path.exists():
                        continue
                    
                    # Tambahkan file berdasarkan prefiks
                    if prefixes_to_clean:
                        for prefix in prefixes_to_clean:
                            # Untuk setiap ekstensi, tambahkan file dengan prefiks yang sesuai
                            exts = extensions_to_clean or ['*.jpg', '*.png', '*.txt', '*.npy']
                            for ext in exts:
                                prefix_pattern = f"{prefix}_*{ext}"
                                files_to_delete.extend(list(dir_path.glob(prefix_pattern)))
                    else:
                        # Jika tidak ada prefiks, hapus semua file (hati-hati!)
                        files_to_delete.extend(list(dir_path.glob('*.*')))
            
            total_files = len(files_to_delete)
            
            # Step 2: Hapus file (50-90%)
            update_cleanup_progress(2, 4, f"Menghapus {total_files} file...")
            
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 50
                ui_components['progress_bar'].description = "Cleanup: 50%"
            
            # Hapus file-file yang ditemukan
            for i, file_path in enumerate(files_to_delete):
                try:
                    os.remove(file_path)
                    
                    # Update progress
                    if 'progress_bar' in ui_components and total_files > 0:
                        progress = 50 + int((i / total_files) * 40)  # 50% - 90%
                        ui_components['progress_bar'].value = progress
                        ui_components['progress_bar'].description = f"Cleanup: {progress}%"
                    
                    # Update current progress
                    if 'current_progress' in ui_components:
                        ui_components['current_progress'].description = f"Menghapus: {i+1}/{total_files}"
                        ui_components['current_progress'].value = 2
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ Gagal menghapus {file_path}: {str(e)}")
            
            # Step 3: Hapus metadata (90-95%)
            update_cleanup_progress(3, 4, "Membersihkan metadata...")
            
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 90
                ui_components['progress_bar'].description = "Cleanup: 90%"
            
            # Hapus direktori metadata jika ada
            metadata_dir = paths / 'metadata'
            if metadata_dir.exists():
                try:
                    shutil.rmtree(metadata_dir)
                except Exception as e:
                    if logger:
                        logger.warning(f"⚠️ Gagal menghapus metadata: {str(e)}")
            
            # Step 4: Selesai (100%)
            update_cleanup_progress(4, 4, f"Pembersihan {total_files} file selesai")
            
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 100
                ui_components['progress_bar'].description = "Cleanup: 100%"
            
            # Notifikasi sukses
            with ui_components['status']:
                display(create_status_indicator("success", 
                    f"{ICONS.get('success', '✅')} Data berhasil dibersihkan, {total_files} file dihapus"))
            
            if update_status:
                update_status(ui_components, "success", 
                    f"{ICONS.get('success', '✅')} Data berhasil dibersihkan, {total_files} file dihapus")
            
            # Notifikasi observer tentang selesai cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.DATA_CLEANUP_END,
                    sender="cleanup_handler",
                    message=f"Pembersihan data selesai, {total_files} file dihapus",
                    files_deleted=total_files
                )
            except ImportError:
                pass
            
            # Sembunyikan elemen UI yang tidak relevan
            hide_ui_elements()
            
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
            
            if update_status:
                update_status(ui_components, "error", 
                    f"{ICONS.get('error', '❌')} Gagal membersihkan data: {str(e)}")
            
            if logger: logger.error(f"{ICONS.get('error', '❌')} Error saat membersihkan data: {str(e)}")
            
            # Notifikasi observer tentang error
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.DATA_CLEANUP_ERROR,
                    sender="cleanup_handler",
                    message=f"Error saat pembersihan data: {str(e)}"
                )
            except ImportError:
                pass
        
        finally:
            # Aktifkan kembali tombol setelah proses selesai
            disable_buttons(False)
            
            # Reset progress bar
            if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
                ui_components['reset_progress_bar']()
    
    # Helper function untuk update progress cleanup
    def update_cleanup_progress(step, total_steps, message):
        """Update progress untuk proses cleanup."""
        # Update current progress
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = step
            ui_components['current_progress'].max = total_steps
            ui_components['current_progress'].description = f"Step {step}/{total_steps}"
        
        # Log message ke status area
        with ui_components['status']:
            display(create_status_indicator("info", f"{ICONS.get('processing', '🔄')} {message}"))
            
        # Log juga ke logger jika tersedia
        if logger:
            logger.info(f"{ICONS.get('processing', '🔄')} {message}")
    
    # Function untuk hide UI elements setelah cleanup
    def hide_ui_elements():
        # Sembunyikan elemen UI yang tidak relevan
        elements_to_hide = [
            'cleanup_button', 'summary_container', 'visualization_container',
            'visualization_buttons', 'visualize_button', 'compare_button', 'distribution_button'
        ]
        
        for element in elements_to_hide:
            if element in ui_components:
                if hasattr(ui_components[element], 'layout'):
                    ui_components[element].layout.display = 'none'
        
        # Clear output containers
        containers_to_clear = ['summary_container', 'visualization_container']
        for container in containers_to_clear:
            if container in ui_components:
                with ui_components[container]:
                    clear_output()
    
    # Function untuk cleanup UI
    def cleanup_ui():
        """Fungsi untuk restore UI setelah cleanup."""
        # Enable tombol yang perlu diaktifkan kembali
        disable_buttons(False)
        
        # Reset progress bar
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        else:
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 0
                ui_components['progress_bar'].layout.visibility = 'hidden'
                
            if 'current_progress' in ui_components:
                ui_components['current_progress'].value = 0
                ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Register handler untuk tombol cleanup
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup,
        'update_cleanup_progress': update_cleanup_progress,
        'cleanup_ui': cleanup_ui,
        'hide_ui_elements': hide_ui_elements
    })
    
    return ui_components