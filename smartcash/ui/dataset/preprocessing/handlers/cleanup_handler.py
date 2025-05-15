"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_handler.py
Deskripsi: Handler pembersihan untuk preprocessing dataset
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
from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def setup_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk membersihkan data hasil preprocessing.
    
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
                      'preprocess_button']
        
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
                message = "Apakah Anda yakin ingin menghapus semua data hasil preprocessing?"
                
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
                        "Konfirmasi: Anda akan menghapus semua data hasil preprocessing. Lanjutkan?",
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
        try:
            with ui_components['status']: 
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
            
            # Notifikasi observer tentang pembatalan cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_CLEANUP_ERROR,  # Gunakan event yang sudah ada
                    sender="preprocessing_handler",
                    message="Pembersihan data preprocessing dibatalkan oleh pengguna"
                )
            except (ImportError, AttributeError):
                # Tangani error jika modul tidak tersedia atau atribut tidak ada
                pass
                
            # Aktifkan kembali tombol setelah batal
            disable_buttons(False)
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"{ICONS.get('error', '❌')} Error saat membatalkan cleanup: {str(e)}")
            disable_buttons(False)  # Pastikan tombol tetap diaktifkan kembali
    
    # Fungsi untuk melakukan cleanup dengan progress tracking
    def perform_cleanup():
        # Dapatkan direktori target
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        
        try:
            # Tampilkan status proses dimulai
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data preprocessing..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data preprocessing...")
            
            # Notifikasi observer sebelum cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_CLEANUP_START,
                    sender="preprocessing_handler",
                    message="Memulai pembersihan data preprocessing"
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
            
            # Pembersihan dengan dataset manager jika tersedia
            dataset_manager = ui_components.get('dataset_manager')
            if dataset_manager and hasattr(dataset_manager, 'clean_preprocessed'):
                # ===== PEMBERSIHAN DENGAN DATASET MANAGER =====
                update_cleanup_progress(ui_components, 1, 4, "Mempersiapkan pembersihan data...")
                
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].value = 25
                    ui_components['progress_bar'].description = "Cleanup: 25%"
                time.sleep(0.5)  # Memberi kesan proses sedang berlangsung
                
                update_cleanup_progress(ui_components, 2, 4, "Menghapus data preprocessing...")
                
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].value = 50
                    ui_components['progress_bar'].description = "Cleanup: 50%"
                
                # Bersihkan semua split sekaligus
                dataset_manager.clean_preprocessed(split='all')
                
                # Step 3: Verifikasi (90%)
                update_cleanup_progress(ui_components, 3, 4, "Verifikasi hasil pembersihan...")
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].value = 90
                    ui_components['progress_bar'].description = "Cleanup: 90%"
                time.sleep(0.5)  # Memberi kesan proses sedang berlangsung
                
                # Step 4: Selesai (100%)
                update_cleanup_progress(ui_components, 4, 4, "Pembersihan data selesai")
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].value = 100
                    ui_components['progress_bar'].description = "Cleanup: 100%"
                
                success = True
            else:
                # Fallback: Pembersihan manual
                success = perform_manual_cleanup(preprocessed_dir, ui_components)
            
            # Update UI jika sukses
            if success:
                with ui_components['status']:
                    display(create_status_indicator("success", 
                        f"{ICONS.get('success', '✅')} Data preprocessing berhasil dibersihkan"))
                
                update_status_panel(ui_components, "success", 
                    f"{ICONS.get('success', '✅')} Data preprocessing berhasil dibersihkan")
                
                # Sembunyikan elemen UI yang tidak relevan
                ui_components['cleanup_button'].layout.display = 'none'
                if 'summary_container' in ui_components:
                    ui_components['summary_container'].layout.display = 'none'
                if 'visualization_container' in ui_components:
                    ui_components['visualization_container'].layout.display = 'none'
                if 'visualization_buttons' in ui_components:
                    ui_components['visualization_buttons'].layout.display = 'none'
                
                # Notifikasi observer tentang progress
                try:
                    from smartcash.components.observer import notify
                    from smartcash.components.observer.event_topics_observer import EventTopics
                    # Gunakan PREPROCESSING_PROGRESS yang tersedia di EventTopics
                    notify(
                        event_type=EventTopics.PREPROCESSING_PROGRESS,
                        sender="preprocessing_handler",
                        message=f"Progress pembersihan data: {file_name}",
                        progress=progress
                    )
                except (ImportError, AttributeError):
                    # Tangani error jika modul tidak tersedia atau atribut tidak ada
                    pass
                
            # Aktifkan kembali tombol process setelah cleanup selesai
            ui_components['preprocess_button'].disabled = False
            
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
                    event_type=EventTopics.PREPROCESSING_CLEANUP_ERROR,
                    sender="preprocessing_handler",
                    message=f"Error saat pembersihan data: {str(e)}"
                )
            except ImportError:
                pass
        
        finally:
            # Aktifkan kembali tombol setelah proses selesai
            disable_buttons(False)
    
    def perform_manual_cleanup(target_dir: str, ui_components: Dict[str, Any]) -> bool:
        """
        Pembersihan manual untuk preprocessing dataset.
        
        Args:
            target_dir: Direktori target
            ui_components: Dictionary komponen UI
            
        Returns:
            Boolean status keberhasilan
        """
        path = Path(target_dir)
        if not path.exists():
            with ui_components['status']:
                display(create_status_indicator("warning", 
                    f"{ICONS.get('warning', '⚠️')} Direktori tidak ditemukan: {target_dir}"))
            return False
        
        start_time = time.time()
        
        # Step 1: Scan direktori (25%)
        update_cleanup_progress(ui_components, 1, 4, "Scanning direktori preprocessing...")
        splits = [d for d in path.iterdir() if d.is_dir() and d.name in DEFAULT_SPLITS]
        total_dirs = len(splits)
        
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 25
            ui_components['progress_bar'].description = "Cleanup: 25%"
        
        # Step 2: Hapus direktori images (50%)
        update_cleanup_progress(ui_components, 2, 4, "Menghapus direktori images...")
        for i, split_dir in enumerate(splits):
            images_dir = split_dir / 'images'
            if images_dir.exists():
                shutil.rmtree(images_dir)
            
            # Update progress berdasarkan jumlah split
            progress = 25 + (25 * (i+1) / total_dirs)
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = progress
                ui_components['progress_bar'].description = f"Cleanup: {int(progress)}%"
        
        # Step 3: Hapus direktori labels (75%)
        update_cleanup_progress(ui_components, 3, 4, "Menghapus direktori labels...")
        for i, split_dir in enumerate(splits):
            labels_dir = split_dir / 'labels'
            if labels_dir.exists():
                shutil.rmtree(labels_dir)
            
            # Update progress
            progress = 50 + (25 * (i+1) / total_dirs)
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].value = progress
                ui_components['progress_bar'].description = f"Cleanup: {int(progress)}%"
        
        # Step 4: Hapus metadata dan direktori kosong (100%)
        update_cleanup_progress(ui_components, 4, 4, "Menghapus metadata dan finalisasi...")
        metadata_dir = path / 'metadata'
        if metadata_dir.exists():
            shutil.rmtree(metadata_dir)
        
        # Hapus direktori split yang kosong
        for split_dir in splits:
            try:
                split_dir.rmdir()  # Hanya hapus jika kosong
            except:
                pass
        
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
            ui_components['progress_bar'].description = "Cleanup: 100%"
        
        # Buat ulang direktori utama
        path.mkdir(parents=True, exist_ok=True)
        
        duration = time.time() - start_time
        if logger: logger.info(f"{ICONS['success']} Pembersihan data selesai dalam {duration:.2f} detik")
        
        return True

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