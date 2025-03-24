"""
File: smartcash/ui/dataset/shared/cleanup_handler.py
Deskripsi: Utilitas bersama untuk pembersihan data yang digunakan oleh preprocessing dan augmentasi
"""

import time
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Callable
from IPython.display import display, clear_output
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.ui.dataset.shared.status_panel import update_status_panel
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

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
    
    # Log message ke status area
    with ui_components['status']:
        display(create_status_indicator("info", f"{ICONS.get('processing', '🔄')} {message}"))
        
    # Log juga ke logger jika tersedia
    logger = ui_components.get('logger')
    if logger:
        logger.info(f"{ICONS.get('processing', '🔄')} {message}")

def setup_shared_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None, 
                              module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Setup handler untuk membersihkan data hasil processing dengan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary komponen UI yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi untuk menonaktifkan semua tombol saat proses cleanup
    def disable_buttons(disable=True):
        """Nonaktifkan semua tombol saat sedang proses."""
        # Daftar tombol yang perlu di-disable
        button_keys = ['cleanup_button', 'save_button', 'reset_button', 
                      'visualize_button', 'compare_button', 'distribution_button']
        
        # Tambahkan tombol spesifik berdasarkan module_type
        if module_type == 'preprocessing':
            button_keys.append('preprocess_button')
        else:
            button_keys.append('augment_button')
        
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
                
                # Buat pesan konfirmasi sesuai module_type
                message = (
                    "Apakah Anda yakin ingin menghapus semua data hasil preprocessing?" if module_type == 'preprocessing'
                    else "Apakah Anda yakin ingin menghapus semua data augmentasi?"
                )
                
                # Buat dialog konfirmasi
                dialog = create_confirmation_dialog(
                    "Konfirmasi Pembersihan Data",
                    message + " Tindakan ini tidak dapat dibatalkan.",
                    on_confirm_cleanup, on_cancel_cleanup, "Ya, Hapus Data", "Batal"
                )
                
                with ui_components['status']:
                    clear_output(wait=True)
                    display(dialog)
                return
                
            except ImportError:
                # Lanjutkan tanpa konfirmasi jika fungsi tidak tersedia
                with ui_components['status']: 
                    display(create_info_alert(
                        "Konfirmasi: Anda akan menghapus semua data hasil processing. Lanjutkan?",
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
        # Dapatkan direktori target berdasarkan module_type
        target_dir = (
            ui_components.get('preprocessed_dir', 'data/preprocessed') if module_type == 'preprocessing' 
            else ui_components.get('augmented_dir', 'data/augmented')
        )
        
        # Tambahan untuk augmentation: direktori preprocessed juga perlu dibersihkan
        additional_dir = ui_components.get('preprocessed_dir', 'data/preprocessed') if module_type == 'augmentation' else None
        
        # Tentukan event_type berdasarkan module_type
        event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
        
        try:
            # Tampilkan status proses dimulai
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data {module_type}..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data {module_type}...")
            
            # Notifikasi observer sebelum cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=getattr(EventTopics, f"{event_type_prefix}_CLEANUP_START"),
                    sender=f"{module_type}_handler",
                    message=f"Memulai pembersihan data {module_type}"
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
            
            # Pembersihan dengan dataset manager jika tersedia untuk preprocessing
            if module_type == 'preprocessing':
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
                    # Fallback: Pembersihan manual untuk preprocessing
                    success = perform_manual_cleanup(target_dir)
            else:
                # Augmentation: Bersihkan file dengan prefix tertentu di kedua lokasi
                success = perform_augmentation_cleanup(target_dir, additional_dir, ui_components)
            
            # Update UI jika sukses
            if success:
                with ui_components['status']:
                    display(create_status_indicator("success", 
                        f"{ICONS.get('success', '✅')} Data {module_type} berhasil dibersihkan"))
                
                update_status_panel(ui_components, "success", 
                    f"{ICONS.get('success', '✅')} Data {module_type} berhasil dibersihkan")
                
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
                        event_type=getattr(EventTopics, f"{event_type_prefix}_CLEANUP_END"),
                        sender=f"{module_type}_handler",
                        message=f"Pembersihan data {module_type} selesai"
                    )
                except ImportError:
                    pass
                
            # Aktifkan kembali tombol process setelah cleanup selesai
            process_button = ui_components['preprocess_button'] if module_type == 'preprocessing' else ui_components['augment_button']
            process_button.disabled = False
            
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
                    event_type=getattr(EventTopics, f"{event_type_prefix}_CLEANUP_ERROR"),
                    sender=f"{module_type}_handler",
                    message=f"Error saat pembersihan data: {str(e)}"
                )
            except ImportError:
                pass
        
        finally:
            # Aktifkan kembali tombol setelah proses selesai
            disable_buttons(False)
    
    def perform_manual_cleanup(target_dir: str) -> bool:
        """
        Pembersihan manual untuk preprocessing dataset.
        
        Args:
            target_dir: Direktori target
            
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
    
    def perform_augmentation_cleanup(main_dir: str, additional_dir: str, ui_components: Dict[str, Any]) -> bool:
        """
        Pembersihan khusus untuk augmentation dataset (dengan dukungan multiple locations).
        
        Args:
            main_dir: Direktori utama augmentasi
            additional_dir: Direktori tambahan (preprocessed) yang juga perlu dibersihkan
            ui_components: Dictionary komponen UI
            
        Returns:
            Boolean status keberhasilan
        """
        # Ambil prefix augmentasi dari UI
        aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
        
        total_files_deleted = 0
        subdirs = ['images', 'labels']
        
        # Step 1: Persiapan (25%)
        update_cleanup_progress(ui_components, 1, 4, "Scanning direktori augmentasi...")
        
        # 1. Cari file di folder temp augmentasi
        temp_files_to_delete = []
        path = Path(main_dir)
        if path.exists():
            for subdir in subdirs:
                dir_path = path / subdir
                if not dir_path.exists():
                    continue
                
                # Cari file dengan pola augmentasi di folder temp
                temp_files_to_delete.extend(list(dir_path.glob(f"{aug_prefix}_*.*")))
                
        # 2. Cari file augmentasi di folder preprocessed (untuk setiap split)
        preprocessed_files_to_delete = []
        if additional_dir:
            preproc_path = Path(additional_dir)
            for split in DEFAULT_SPLITS:
                for subdir in subdirs:
                    dir_path = preproc_path / split / subdir
                    if not dir_path.exists():
                        continue
                    
                    # Cari file dengan pola augmentasi di folder preprocessed
                    preprocessed_files_to_delete.extend(list(dir_path.glob(f"{aug_prefix}_*.*")))
        
        # Gabungkan semua file yang perlu dihapus
        all_files_to_delete = temp_files_to_delete + preprocessed_files_to_delete
        
        # Update progress bar dengan total file
        total_files = len(all_files_to_delete)
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].max = total_files if total_files > 0 else 1
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = f'Total: {total_files} file'
        
        # Step 2-3: Hapus file (75%)
        update_cleanup_progress(ui_components, 2, 4, "Menghapus file augmentasi...")
        
        # Hapus file dengan progress tracking
        for i, file_path in enumerate(all_files_to_delete):
            try:
                # Hapus file
                os.remove(file_path)
                total_files_deleted += 1
                
                # Update progress
                if 'progress_bar' in ui_components and 'current_progress' in ui_components:
                    ui_components['progress_bar'].value = i + 1
                    progress_percent = int((i + 1) / total_files * 100) if total_files > 0 else 100
                    ui_components['current_progress'].value = progress_percent
                    ui_components['current_progress'].description = f'Menghapus: {file_path.name}'
                
                # Report progress via observer (lebih jarang untuk mengurangi overhead)
                if i % max(10, total_files//10) == 0 or i == len(all_files_to_delete) - 1:
                    try:
                        from smartcash.components.observer import notify
                        from smartcash.components.observer.event_topics_observer import EventTopics
                        notify(
                            event_type=EventTopics.AUGMENTATION_CLEANUP_PROGRESS,
                            sender="augmentation_handler",
                            message=f"Menghapus file augmentasi ({i+1}/{total_files})",
                            progress=i+1,
                            total=total_files
                        )
                    except (ImportError, AttributeError):
                        pass
            
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal menghapus {file_path}: {e}")
        
        # Step 4: Finalisasi (100%)
        update_cleanup_progress(ui_components, 4, 4, "Finalisasi cleanup...")
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 100
            ui_components['progress_bar'].description = "Cleanup: 100%"
        
        # Tampilkan status sukses
        with ui_components['status']:
            display(create_status_indicator("success", 
                f"{ICONS.get('success', '✅')} Data augmentasi berhasil dibersihkan. {total_files_deleted} file dihapus"))
        
        # Update status panel
        update_status_panel(
            ui_components,
            "success",
            f"{ICONS.get('success', '✅')} Data augmentasi berhasil dibersihkan. {total_files_deleted} file dihapus"
        )
        
        return True
    
    # Register handler
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