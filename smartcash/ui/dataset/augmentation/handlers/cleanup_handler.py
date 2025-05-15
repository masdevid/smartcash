"""
File: smartcash/ui/dataset/augmentation/handlers/cleanup_handler.py
Deskripsi: Handler pembersihan untuk augmentasi dataset
"""

import time
import os
import shutil
from pathlib import Path
from typing import Dict, Any
from IPython.display import display, clear_output
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def setup_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk membersihkan data hasil augmentasi.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi untuk menonaktifkan semua tombol saat proses cleanup
    def disable_buttons(disable=True):
        """Nonaktifkan semua tombol saat sedang proses."""
        # Daftar tombol yang perlu di-disable
        button_keys = ['cleanup_button', 'save_button', 'reset_button', 
                      'visualize_button', 'compare_button', 'distribution_button',
                      'augment_button']
        
        # Disable semua tombol dalam daftar
        for btn_name in button_keys:
            if btn_name in ui_components:
                ui_components[btn_name].disabled = disable
    
    # Handler untuk tombol cleanup
    def on_cleanup_click(b):
        try:
            # Nonaktifkan semua tombol saat proses dimulai
            disable_buttons(True)
            
            # Buat dialog konfirmasi jika tersedia
            try:
                from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
                
                def on_confirm_cleanup():
                    with ui_components['status']: clear_output(wait=True)
                    perform_cleanup()
                
                def on_cancel_cleanup():
                    with ui_components['status']: 
                        clear_output(wait=True)
                        display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
                    # Aktifkan kembali tombol setelah batal
                    disable_buttons(False)
                
                # Buat pesan konfirmasi
                message = "Apakah Anda yakin ingin menghapus semua data hasil augmentasi?"
                
                # Buat dialog konfirmasi dengan parameter yang benar
                dialog = create_confirmation_dialog(
                    message=message + " Tindakan ini tidak dapat dibatalkan.",
                    on_confirm=on_confirm_cleanup, 
                    on_cancel=on_cancel_cleanup,
                    title="Konfirmasi Penghapusan Data",
                    confirm_label="Ya, Hapus Data", 
                    cancel_label="Batal"
                )
                
                with ui_components['status']:
                    clear_output(wait=True)
                    display(dialog)
                return
                
            except ImportError:
                # Lanjutkan tanpa konfirmasi jika fungsi tidak tersedia
                with ui_components['status']: 
                    display(create_info_alert(
                        "Konfirmasi: Anda akan menghapus semua data hasil augmentasi. Lanjutkan?",
                        "warning", ICONS['warning']
                    ))
                    # Tambahkan tombol konfirmasi manual
                    confirm_btn = widgets.Button(description="Ya, Hapus Data", button_style="danger", icon="trash")
                    cancel_btn = widgets.Button(description="Batal", button_style="info", icon="times")
                    
                    confirm_btn.on_click(lambda b: perform_cleanup())
                    cancel_btn.on_click(lambda b: cancel_cleanup())
                    
                    display(widgets.HBox([confirm_btn, cancel_btn], layout=widgets.Layout(justify_content="center", margin="10px 0")))
                return
                
        except Exception as e:
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
            # Aktifkan kembali tombol jika terjadi error
            disable_buttons(False)
    
    # Fungsi untuk membatalkan cleanup
    def cancel_cleanup():
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
        # Aktifkan kembali tombol setelah batal
        disable_buttons(False)
    
    # Fungsi untuk melakukan cleanup dengan progress tracking
    def perform_cleanup():
        # Dapatkan prefix augmentasi dan direktori target
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        
        try:
            # Tampilkan status proses dimulai
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data augmentasi..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data augmentasi dengan prefix {aug_prefix}...")
            
            # Notifikasi observer sebelum cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_CLEANUP_START,
                    sender="augmentation_handler",
                    message=f"Memulai pembersihan data augmentasi dengan prefix {aug_prefix}"
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
                ui_components['current_progress'].max = 4  # Tahapan: scan dirs, hapus files dari preprocessed, hapus files dari augmented, finalisasi
                ui_components['current_progress'].value = 0
                ui_components['current_progress'].description = "Scanning..."
                ui_components['current_progress'].layout.visibility = 'visible'
            
            # Pembersihan data augmentasi
            start_time = time.time()
            
            # Step 1: Scan direktori (25%)
            update_cleanup_progress(ui_components, 1, 4, "Scanning direktori augmentasi...")
            
            # Pattern file yang akan dihapus
            pattern = f"{aug_prefix}_*.jpg"
            
            # Tracking file yang dihapus
            deleted_files = []
            
            # Step 2: Hapus file dari preprocessed dir (50%)
            update_cleanup_progress(ui_components, 2, 4, "Menghapus file augmentasi dari data preprocessed...")
            
            # Hapus dari preprocessed dir (untuk semua split)
            for split in DEFAULT_SPLITS:
                split_dir = Path(preprocessed_dir) / split / 'images'
                if split_dir.exists():
                    aug_files = list(split_dir.glob(pattern))
                    
                    # Hapus file satu per satu
                    for file in aug_files:
                        try:
                            # Hapus juga file label terkait jika ada
                            label_dir = Path(preprocessed_dir) / split / 'labels'
                            label_file = label_dir / (file.stem + '.txt')
                            if label_file.exists():
                                os.remove(label_file)
                            
                            # Hapus file gambar
                            os.remove(file)
                            deleted_files.append(file)
                        except Exception as e:
                            if logger: logger.warning(f"{ICONS['warning']} Gagal menghapus {file.name}: {str(e)}")
            
            # Update progress
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 50
                ui_components['progress_bar'].description = "Cleanup: 50%"
            
            # Step 3: Hapus file dari augmented dir (75%)
            update_cleanup_progress(ui_components, 3, 4, "Menghapus file augmentasi dari direktori augmented...")
            
            # Hapus dari augmented dir
            augmented_images = Path(augmented_dir) / 'images'
            if augmented_images.exists():
                aug_files = list(augmented_images.glob(pattern))
                
                # Hapus file satu per satu
                for file in aug_files:
                    try:
                        # Hapus juga file label terkait jika ada
                        label_dir = Path(augmented_dir) / 'labels'
                        label_file = label_dir / (file.stem + '.txt')
                        if label_file.exists():
                            os.remove(label_file)
                        
                        # Hapus file gambar
                        os.remove(file)
                        deleted_files.append(file)
                    except Exception as e:
                        if logger: logger.warning(f"{ICONS['warning']} Gagal menghapus {file.name}: {str(e)}")
            
            # Update progress
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 75
                ui_components['progress_bar'].description = "Cleanup: 75%"
            
            # Step 4: Finalisasi (100%)
            update_cleanup_progress(ui_components, 4, 4, "Finalisasi pembersihan...")
            
            # Hapus direktori kosong
            for split in DEFAULT_SPLITS:
                labels_dir = Path(preprocessed_dir) / split / 'labels'
                images_dir = Path(preprocessed_dir) / split / 'images'
                
                # Buat ulang direktori jika kosong (untuk memastikan struktur tetap ada)
                for dir_path in [labels_dir, images_dir]:
                    if dir_path.exists():
                        # Cek apakah kosong
                        is_empty = True
                        for _ in dir_path.iterdir():
                            is_empty = False
                            break
                        
                        # Jika kosong, hapus dan buat ulang
                        if is_empty:
                            try:
                                os.rmdir(dir_path)
                                dir_path.mkdir(parents=True, exist_ok=True)
                            except Exception:
                                pass
            
            # Update progress
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = 100
                ui_components['progress_bar'].description = "Cleanup: 100%"
            
            # Hitung durasi dan tampilkan hasil
            duration = time.time() - start_time
            success_message = f"{ICONS['success']} Pembersihan selesai. {len(deleted_files)} file dihapus dalam {duration:.1f} detik"
            
            with ui_components['status']:
                display(create_status_indicator("success", success_message))
                
            update_status_panel(ui_components, "success", success_message)
            
            # Sembunyikan elemen UI yang tidak relevan
            ui_components['cleanup_button'].layout.display = 'none'
            if 'summary_container' in ui_components:
                ui_components['summary_container'].layout.display = 'none'
            if 'visualization_container' in ui_components:
                ui_components['visualization_container'].layout.display = 'none'
            if 'visualization_buttons' in ui_components:
                ui_components['visualization_buttons'].layout.display = 'none'
            
            # Notifikasi observer
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_CLEANUP_END,
                    sender="augmentation_handler",
                    message=f"Pembersihan data augmentasi selesai: {len(deleted_files)} file dihapus",
                    files_deleted=len(deleted_files),
                    duration=duration
                )
            except ImportError:
                pass
                
            # Aktifkan kembali tombol process setelah cleanup selesai
            ui_components['augment_button'].disabled = False
            
            # Reset progress bar
            if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
                ui_components['reset_progress_bar']()
                
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
            
            update_status_panel(ui_components, "error", 
                f"{ICONS.get('error', '❌')} Gagal membersihkan data: {str(e)}")
            
            if logger: logger.error(f"{ICONS.get('error', '❌')} Error saat membersihkan data: {str(e)}")
            
            # Notifikasi observer tentang error
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_CLEANUP_ERROR,
                    sender="augmentation_handler",
                    message=f"Error saat pembersihan data: {str(e)}"
                )
            except ImportError:
                pass
        
        finally:
            # Aktifkan kembali tombol setelah proses selesai
            disable_buttons(False)
    
    def update_cleanup_progress(ui_components: Dict[str, Any], step: int, total_steps: int, message: str):
        """
        Update progress untuk proses cleanup.
        
        Args:
            ui_components: Dictionary komponen UI
            step: Step saat ini
            total_steps: Total jumlah step
            message: Pesan progress
        """
        # Update current progress
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = step
            ui_components['current_progress'].max = total_steps
            ui_components['current_progress'].description = f"Step {step}/{total_steps}"
        
        # Update step label
        if 'step_label' in ui_components:
            ui_components['step_label'].value = message
        
        # Log message ke status area
        with ui_components['status']:
            display(create_status_indicator("info", f"{ICONS.get('processing', '🔄')} {message}"))
            
        # Log juga ke logger jika tersedia
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"{ICONS.get('processing', '🔄')} {message}")
    
    # Register handler untuk tombol cleanup
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup,
        'cancel_cleanup': cancel_cleanup,
        'disable_buttons': disable_buttons,
        'update_cleanup_progress': update_cleanup_progress
    })
    
    return ui_components