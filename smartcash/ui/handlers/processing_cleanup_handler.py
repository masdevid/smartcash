"""
File: smartcash/ui/handlers/processing_cleanup_handler.py
Deskripsi: Handler pembersihan data bersama untuk modul preprocessing dan augmentasi
"""

import time
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from IPython.display import display, clear_output
import ipywidgets as widgets
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def setup_processing_cleanup_handler(
    ui_components: Dict[str, Any],
    module_type: str = 'preprocessing',
    config: Dict[str, Any] = None,
    env = None
) -> Dict[str, Any]:
    """
    Setup handler untuk membersihkan data hasil processing (preprocessing atau augmentasi).
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Fungsi untuk menonaktifkan semua tombol saat proses cleanup
    def disable_buttons(disable=True):
        """Nonaktifkan semua tombol saat sedang proses."""
        # Tombol primer berdasarkan module_type
        primary_button_key = 'preprocess_button' if module_type == 'preprocessing' else 'augment_button'
        
        # Daftar tombol yang perlu di-disable
        button_keys = ['cleanup_button', 'save_button', 'reset_button', 
                     'visualize_button', 'compare_button', 'distribution_button',
                     primary_button_key]
        
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
                
                # Pesan konfirmasi berdasarkan jenis modul
                message = "Apakah Anda yakin ingin menghapus semua data hasil preprocessing?" if module_type == 'preprocessing' else "Apakah Anda yakin ingin menghapus semua data augmentasi?"
                
                def on_confirm_cleanup():
                    with ui_components['status']: clear_output(wait=True)
                    perform_cleanup()
                
                def on_cancel_cleanup():
                    with ui_components['status']: 
                        clear_output(wait=True)
                        display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
                    # Aktifkan kembali tombol setelah batal
                    disable_buttons(False)
                
                # Buat dialog konfirmasi dengan parameter yang benar
                dialog = create_confirmation_dialog(
                    message=message + " Tindakan ini tidak dapat dibatalkan.",
                    on_confirm=on_confirm_cleanup, 
                    on_cancel=on_cancel_cleanup,
                    title="Konfirmasi Pembersihan Data",
                    confirm_label="Ya, Hapus Data", 
                    cancel_label="Batal"
                )
                
                with ui_components['status']:
                    clear_output(wait=True)
                    display(dialog)
                return
                
            except ImportError:
                # Fallback jika komponen dialog tidak tersedia
                with ui_components['status']: 
                    display(create_info_alert(
                        f"Konfirmasi: Anda akan menghapus semua data hasil {module_type}. Lanjutkan?",
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
        # Dapatkan parameter cleanup berdasarkan module_type
        if module_type == 'preprocessing':
            target_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            file_pattern = None  # hapus semua di preprocessing
            process_name = "preprocessing"
        else:  # augmentation
            target_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            temp_dir = ui_components.get('augmented_dir', 'data/augmented')
            # Dapatkan prefix dari UI
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            file_pattern = f"{aug_prefix}_*"
            process_name = "augmentasi"
        
        # Start time untuk tracking
        start_time = time.time()
        
        try:
            # Tampilkan status proses dimulai
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data {process_name}..."))
            
            # Update status panel
            _update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Memulai pembersihan data {process_name}...")
            
            # Notifikasi observer sebelum cleanup
            _notify_cleanup_start(ui_components, module_type)
            
            # Setup progress tracking
            _setup_progress_tracking(ui_components)
            
            # Fase 1: Analisis file yang akan dihapus
            files_to_delete = _analyze_files_to_delete(
                ui_components=ui_components,
                target_dir=target_dir,
                temp_dir=temp_dir if module_type == 'augmentation' else None,
                file_pattern=file_pattern
            )
            
            # Tampilkan jumlah file yang akan dihapus
            total_files = len(files_to_delete)
            if logger: logger.info(f"{ICONS['info']} Menemukan {total_files} file untuk dihapus")
            
            # Update progress bar untuk jumlah total file
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = total_files if total_files > 0 else 1
                ui_components['progress_bar'].description = f"Total: {total_files} file"
            
            # Tidak ada file yang perlu dihapus
            if total_files == 0:
                with ui_components['status']:
                    display(create_status_indicator("warning", 
                        f"{ICONS['warning']} Tidak ada file yang perlu dihapus"))
                    
                _update_status_panel(ui_components, "warning", 
                    f"{ICONS['warning']} Tidak ada file yang perlu dihapus")
                
                # Selesai
                _cleanup_ui(ui_components)
                return
                
            # Fase 3: Hapus file dengan progress tracking
            deleted_files = _delete_files(ui_components, files_to_delete)
            
            # Fase 4: Post-cleanup operation
            _post_cleanup_operations(ui_components, target_dir, module_type)
            
            # Durasi dan pesan sukses
            duration = time.time() - start_time
            success_message = f"{ICONS['success']} Pembersihan selesai. {len(deleted_files)} file dihapus dalam {duration:.1f} detik"
            
            with ui_components['status']:
                display(create_status_indicator("success", success_message))
                
            _update_status_panel(ui_components, "success", success_message)
            
            # Reset UI components related to processed data
            _reset_ui_post_cleanup(ui_components)
            
            # Notifikasi observer
            _notify_cleanup_end(ui_components, module_type, len(deleted_files), duration)
                
        except Exception as e:
            # Tangani error dengan notifikasi yang jelas
            error_message = f"{ICONS['error']} Error saat pembersihan: {str(e)}"
            
            with ui_components['status']:
                display(create_status_indicator("error", error_message))
                
            _update_status_panel(ui_components, "error", error_message)
            
            if logger: logger.error(f"{error_message}")
            
            # Notifikasi observer error
            _notify_cleanup_error(ui_components, module_type, str(e))
        
        finally:
            # Selalu bersihkan UI dan reset state
            _cleanup_ui(ui_components)
    
    # Register handler untuk tombol cleanup
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup,
        'cancel_cleanup': cancel_cleanup,
        'disable_buttons': disable_buttons
    })
    
    return ui_components

def _analyze_files_to_delete(
    ui_components: Dict[str, Any], 
    target_dir: str, 
    temp_dir: Optional[str] = None,
    file_pattern: Optional[str] = None
) -> List[Path]:
    """Analisa dan tentukan file yang akan dihapus."""
    _update_cleanup_progress(ui_components, 1, 4, "Menganalisis file yang akan dihapus...")
    
    files_to_delete = []
    
    # Deteksi module_type
    is_preprocessing = temp_dir is None
    
    if is_preprocessing:
        # Untuk preprocessing, hapus semua file di direktori preprocessed
        target_path = Path(target_dir)
        
        # Cek setiap split
        for split in DEFAULT_SPLITS:
            split_dir = target_path / split
            if not split_dir.exists():
                continue
            
            # Cek direktori images
            images_dir = split_dir / 'images'
            if images_dir.exists():
                files_to_delete.extend(list(images_dir.glob('*.*')))
            
            # Cek direktori labels
            labels_dir = split_dir / 'labels'
            if labels_dir.exists():
                files_to_delete.extend(list(labels_dir.glob('*.*')))
    else:
        # Untuk augmentation, hapus hanya file dengan pattern tertentu
        target_path = Path(target_dir)
        
        # Cek di target_dir (preprocessed) untuk setiap split
        for split in DEFAULT_SPLITS:
            split_dir = target_path / split
            if not split_dir.exists():
                continue
            
            # Cek direktori images untuk file augmentasi
            images_dir = split_dir / 'images'
            if images_dir.exists():
                files_to_delete.extend(list(images_dir.glob(f"{file_pattern}.*")))
            
            # Cek direktori labels untuk file augmentasi
            labels_dir = split_dir / 'labels'
            if labels_dir.exists():
                files_to_delete.extend(list(labels_dir.glob(f"{file_pattern}.*")))
        
        # Jika ada temp_dir, cek juga di sana
        if temp_dir:
            temp_path = Path(temp_dir)
            
            # Cek direktori images
            images_dir = temp_path / 'images'
            if images_dir.exists():
                files_to_delete.extend(list(images_dir.glob(f"{file_pattern}.*")))
            
            # Cek direktori labels
            labels_dir = temp_path / 'labels'
            if labels_dir.exists():
                files_to_delete.extend(list(labels_dir.glob(f"{file_pattern}.*")))
    
    # Step 2 completed
    _update_cleanup_progress(ui_components, 2, 4, f"Ditemukan {len(files_to_delete)} file untuk dihapus")
    
    return files_to_delete

def _delete_files(ui_components: Dict[str, Any], files_to_delete: List[Path]) -> List[Path]:
    """Hapus file dengan progress tracking."""
    _update_cleanup_progress(ui_components, 3, 4, "Menghapus file...")
    
    logger = ui_components.get('logger')
    deleted_files = []
    total_files = len(files_to_delete)
    
    # Jika tidak ada file, return kosong
    if total_files == 0:
        return deleted_files
    
    # Iterate files dengan progress tracking
    for i, file_path in enumerate(files_to_delete):
        try:
            # Hapus file
            os.remove(file_path)
            deleted_files.append(file_path)
            
            # Update progress setiap 10 file atau 10% (mana yang lebih sering)
            update_frequency = max(1, min(10, total_files // 10))
            
            if (i + 1) % update_frequency == 0 or i == len(files_to_delete) - 1:
                # Update progress UI
                progress_percentage = min(100, int((i + 1) / total_files * 100))
                
                if 'progress_bar' in ui_components:
                    ui_components['progress_bar'].value = i + 1
                    ui_components['progress_bar'].description = f"Hapus: {progress_percentage}%"
                
                # Update current progress description dengan nama file
                if 'current_progress' in ui_components:
                    ui_components['current_progress'].description = f"Hapus file {i+1}/{total_files}"
                    
                # Notify observer untuk progress
                _notify_cleanup_progress(ui_components, i + 1, total_files)
        except Exception as e:
            # Log error tapi lanjutkan ke file berikutnya
            if logger:
                logger.warning(f"{ICONS['warning']} Gagal menghapus {file_path.name}: {str(e)}")
    
    return deleted_files

def _post_cleanup_operations(ui_components: Dict[str, Any], target_dir: str, module_type: str) -> None:
    """Lakukan operasi pasca pembersihan."""
    _update_cleanup_progress(ui_components, 4, 4, "Finalisasi pembersihan...")
    
    logger = ui_components.get('logger')
    target_path = Path(target_dir)
    
    # Operasi khusus berdasarkan module_type
    if module_type == 'preprocessing':
        # Untuk preprocessing, hapus direktori metadata jika ada
        metadata_dir = target_path / 'metadata'
        
        if metadata_dir.exists():
            try:
                shutil.rmtree(metadata_dir)
                if logger:
                    logger.info(f"{ICONS['success']} Hapus direktori metadata: {metadata_dir}")
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS['warning']} Gagal hapus metadata: {str(e)}")
    
    # Coba hapus direktori split yang kosong
    for split in DEFAULT_SPLITS:
        split_dir = target_path / split
        if not split_dir.exists():
            continue
        
        try:
            # Cek apakah direktori kosong
            is_empty = True
            for _ in split_dir.iterdir():
                is_empty = False
                break
            
            # Hapus jika kosong
            if is_empty:
                split_dir.rmdir()
                if logger:
                    logger.info(f"{ICONS['success']} Hapus direktori kosong: {split_dir}")
        except Exception:
            pass
    
    # Pastikan direktori utama tetap ada
    target_path.mkdir(parents=True, exist_ok=True)

def _reset_ui_post_cleanup(ui_components: Dict[str, Any]) -> None:
    """Reset komponen UI yang terkait dengan data yang telah diproses."""
    # Sembunyikan container terkait data
    containers = ['summary_container', 'visualization_container', 'visualization_buttons']
    [setattr(ui_components[container].layout, 'display', 'none') 
     for container in containers if container in ui_components]
    
    # Sembunyikan tombol cleanup
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].layout.display = 'none'
    
    # Clear container content
    for container in ['summary_container', 'visualization_container']:
        if container in ui_components:
            with ui_components[container]:
                clear_output()

def _setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking dengan UI yang tepat."""
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].max = 100
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].description = "Cleanup: 0%"
        ui_components['progress_bar'].layout.visibility = 'visible'
    
    if 'current_progress' in ui_components:
        ui_components['current_progress'].max = 4  # Tahapan cleanup
        ui_components['current_progress'].value = 0
        ui_components['current_progress'].description = "Scanning..."
        ui_components['current_progress'].layout.visibility = 'visible'

def _update_cleanup_progress(ui_components: Dict[str, Any], step: int, total_steps: int, message: str) -> None:
    """Update progress untuk proses cleanup."""
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

def _cleanup_ui(ui_components: Dict[str, Any]) -> None:
    """Bersihkan UI setelah selesai operasi."""
    # Reset progress tracking
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar']()
    else:
        # Manual reset
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].layout.visibility = 'hidden'
        
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].layout.visibility = 'hidden'
    
    # Enable tombol yang dinonaktifkan
    if 'disable_buttons' in ui_components and callable(ui_components['disable_buttons']):
        ui_components['disable_buttons'](False)

def _update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """Update status panel dengan pesan."""
    # Validasi ui_components
    if ui_components is None:
        return
    
    logger = ui_components.get('logger')
    
    try:
        # Metode 1: Gunakan fungsi update_status_panel jika tersedia
        if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']):
            try:
                ui_components['update_status_panel'](ui_components, status_type, message)
                return
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat update status panel (metode 1): {str(e)}")
        
        # Metode 2: Update status_panel widget jika tersedia
        if 'status_panel' in ui_components and ui_components['status_panel'] is not None and hasattr(ui_components['status_panel'], 'value'):
            try:
                from smartcash.ui.utils.constants import ALERT_STYLES
                style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
                ui_components['status_panel'].value = f"""
                <div style="padding:10px; background-color:{style['bg_color']}; 
                          color:{style['text_color']}; border-radius:4px; margin:5px 0;
                          border-left:4px solid {style['text_color']};">
                    <p style="margin:5px 0">{style['icon']} {message}</p>
                </div>"""
                return
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat update status panel (metode 2): {str(e)}")
        
        # Metode 3: Update status widget jika tersedia
        if 'status' in ui_components and ui_components['status'] is not None:
            try:
                with ui_components['status']:
                    clear_output(wait=True)
                    display(create_status_indicator(status_type, message))
                return
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS.get('warning', '⚠️')} Error saat update status panel (metode 3): {str(e)}")
        
        # Jika semua metode gagal, log pesan
        if logger:
            logger.debug(f"{ICONS.get('info', 'ℹ️')} Tidak dapat menampilkan status: {message}")
    
    except Exception as e:
        # Tangkap semua error yang mungkin terjadi
        if logger:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Error tidak terduga saat update status panel: {str(e)}")
        # Jangan raise exception agar tidak mengganggu alur program

def _notify_cleanup_start(ui_components: Dict[str, Any], module_type: str) -> None:
    """Notifikasi observer tentang memulai pembersihan."""
    try:
        from smartcash.components.observer import notify
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Gunakan PROGRESS_START sebagai fallback jika event khusus tidak tersedia
        try:
            # Coba dapatkan event type berdasarkan module_type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            event_type = getattr(EventTopics, f"{event_type_prefix}_CLEANUP_START")
        except AttributeError:
            # Fallback ke PROGRESS_START jika event khusus tidak tersedia
            event_type = EventTopics.PROGRESS_START
        
        notify(
            event_type=event_type,
            sender=f"{module_type}_cleanup_handler",
            message=f"Memulai pembersihan data {module_type}"
        )
    except Exception as e:
        # Tangkap semua error dan abaikan
        pass

def _notify_cleanup_progress(ui_components: Dict[str, Any], current: int, total: int) -> None:
    """Notifikasi observer tentang progress pembersihan."""
    try:
        from smartcash.components.observer import notify
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Deteksi module_type
        module_type = 'preprocessing' if 'preprocessing_running' in ui_components else 'augmentation'
        
        # Gunakan PROGRESS_UPDATE sebagai fallback jika event khusus tidak tersedia
        # Ini mencegah error "EventTopics has no attribute PREPROCESSING_CLEANUP_PROGRESS"
        try:
            # Coba dapatkan event type berdasarkan module_type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            event_type = getattr(EventTopics, f"{event_type_prefix}_CLEANUP_PROGRESS")
        except AttributeError:
            # Fallback ke PROGRESS_UPDATE jika event khusus tidak tersedia
            event_type = EventTopics.PROGRESS_UPDATE
        
        notify(
            event_type=event_type,
            sender=f"{module_type}_cleanup_handler",
            message=f"Menghapus file {current}/{total}",
            progress=current,
            total=total
        )
    except Exception as e:
        # Tangkap semua error dan abaikan
        pass

def _notify_cleanup_end(ui_components: Dict[str, Any], module_type: str, files_deleted: int, duration: float) -> None:
    """Notifikasi observer tentang selesai pembersihan."""
    try:
        from smartcash.components.observer import notify
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Gunakan PROGRESS_COMPLETE sebagai fallback jika event khusus tidak tersedia
        try:
            # Coba dapatkan event type berdasarkan module_type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            event_type = getattr(EventTopics, f"{event_type_prefix}_CLEANUP_END")
        except AttributeError:
            # Fallback ke PROGRESS_COMPLETE jika event khusus tidak tersedia
            event_type = EventTopics.PROGRESS_COMPLETE
        
        notify(
            event_type=event_type,
            sender=f"{module_type}_cleanup_handler",
            message=f"Pembersihan data {module_type} selesai",
            files_deleted=files_deleted,
            duration=duration
        )
    except Exception as e:
        # Tangkap semua error dan abaikan
        pass

def _notify_cleanup_error(ui_components: Dict[str, Any], module_type: str, error_message: str) -> None:
    """Notifikasi observer tentang error saat pembersihan."""
    try:
        from smartcash.components.observer import notify
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Gunakan fallback jika event khusus tidak tersedia
        try:
            # Coba dapatkan event type berdasarkan module_type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            event_type = getattr(EventTopics, f"{event_type_prefix}_CLEANUP_ERROR")
        except AttributeError:
            # Fallback ke event error yang tersedia
            if hasattr(EventTopics, f"{event_type_prefix}_ERROR"):
                event_type = getattr(EventTopics, f"{event_type_prefix}_ERROR")
            else:
                # Fallback terakhir jika tidak ada event error yang sesuai
                event_type = "error.cleanup"
        
        notify(
            event_type=event_type,
            sender=f"{module_type}_cleanup_handler",
            message=f"Error saat pembersihan data {module_type}: {error_message}",
            error=error_message
        )
    except Exception as e:
        # Tangkap semua error dan abaikan
        pass