"""
File: smartcash/ui/dataset/shared/cleanup_handler.py
Deskripsi: Utilitas bersama untuk pembersihan data dengan dukungan transactions dan rollback,
serta progress tracking yang dioptimalkan untuk preprocessing dan augmentasi
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional, Tuple, Set
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from concurrent.futures import ThreadPoolExecutor
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
from smartcash.ui.dataset.shared.status_panel import update_status_panel
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def setup_shared_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None, 
                               module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Setup handler untuk pembersihan data dengan transactional support dan progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Tracking state untuk operasi pembersihan
    cleanup_state = {
        'is_cleaning': False,
        'files_to_delete': [],
        'files_deleted': [],
        'rollback_needed': False,
        'transaction_id': '',
        'start_time': 0
    }
    
    # Fungsi untuk menonaktifkan tombol saat sedang proses
    def disable_buttons(disable: bool = True) -> None:
        """Nonaktifkan semua tombol interaktif selama proses pembersihan."""
        # Tombol yang perlu dinonaktifkan selama proses
        button_keys = ['cleanup_button', 'save_button', 'reset_button', 
                     'visualize_button', 'compare_button', 'distribution_button']
        
        # Tambahkan tombol utama berdasarkan module_type
        process_button = 'preprocess_button' if module_type == 'preprocessing' else 'augment_button'
        button_keys.append(process_button)
        
        # Disable/enable tombol dengan one-liner
        [setattr(ui_components[btn], 'disabled', disable) for btn in button_keys 
         if btn in ui_components and hasattr(ui_components[btn], 'disabled')]
    
    # Handler untuk tombol cleanup dengan dialog konfirmasi yang ditingkatkan
    def on_cleanup_click(b) -> None:
        """
        Handler untuk tombol pembersihan data dengan konfirmasi dan undo capability.
        """
        try:
            # Nonaktifkan semua tombol saat proses dimulai
            disable_buttons(True)
            
            # Buat dialog konfirmasi untuk mencegah penghapusan tidak disengaja
            try:
                from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
                
                def on_confirm_cleanup() -> None:
                    """Callback untuk konfirmasi penghapusan."""
                    with ui_components['status']: clear_output(wait=True)
                    perform_cleanup()
                
                def on_cancel_cleanup() -> None:
                    """Callback untuk membatalkan penghapusan."""
                    with ui_components['status']: 
                        clear_output(wait=True)
                        display(create_status_indicator("info", f"{ICONS['info']} Pembersihan dibatalkan"))
                    
                    # Aktifkan kembali tombol setelah batal
                    disable_buttons(False)
                
                # Pesan konfirmasi berdasarkan jenis modul
                message = (
                    "Apakah Anda yakin ingin menghapus semua data hasil preprocessing?" if module_type == 'preprocessing'
                    else "Apakah Anda yakin ingin menghapus semua data augmentasi?"
                )
                
                # Buat dan tampilkan dialog
                dialog = create_confirmation_dialog(
                    "Konfirmasi Pembersihan Data",
                    message + " Tindakan ini tidak dapat dibatalkan.",
                    on_confirm_cleanup, on_cancel_cleanup, "Ya, Hapus Data", "Batal"
                )
                
                with ui_components['status']:
                    clear_output(wait=True)
                    display(dialog)
                
                # Exit early - konfirmasi akan memanggil perform_cleanup jika disetujui
                return
                
            except ImportError:
                # Fallback jika komponen dialog tidak tersedia
                with ui_components['status']: 
                    display(create_info_alert(
                        "Konfirmasi: Anda akan menghapus semua data hasil processing. Lanjutkan?",
                        "warning", ICONS['warning']
                    ))
                    # Buat tombol konfirmasi manual
                    confirm_btn = widgets.Button(description="Ya, Hapus Data", button_style="danger", icon="trash")
                    cancel_btn = widgets.Button(description="Batal", button_style="info", icon="times")
                    
                    # Register handler
                    confirm_btn.on_click(lambda b: perform_cleanup())
                    cancel_btn.on_click(lambda b: cancel_cleanup())
                    
                    # Tampilkan tombol
                    display(widgets.HBox([confirm_btn, cancel_btn], layout=widgets.Layout(justify_content="center", margin="10px 0")))
                
                # Exit early - tombol akan memanggil perform_cleanup jika diklik
                return
                
        except Exception as e:
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
                
            # Aktifkan kembali tombol jika terjadi error
            disable_buttons(False)
    
    # Fungsi untuk membatalkan cleanup
    def cancel_cleanup() -> None:
        """Batalkan operasi pembersihan dan kembalikan UI ke kondisi normal."""
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS['info']} Pembersihan dibatalkan"))
            
        # Aktifkan kembali tombol setelah batal
        disable_buttons(False)
        
        # Reset state
        cleanup_state['is_cleaning'] = False
    
    # Fungsi utama untuk melakukan pembersihan
    def perform_cleanup() -> None:
        """
        Lakukan pembersihan data dengan tracking progress dan dukungan transaksi.
        """
        # Tentukan target berdasarkan module_type
        target_dir = _get_target_directory()
        temp_dir = _get_temp_directory()  # Untuk rollback jika perlu
        
        # Mulai operasi pembersihan
        try:
            # Set flag dan timestamp
            cleanup_state.update({
                'is_cleaning': True,
                'start_time': time.time(),
                'transaction_id': f"cleanup_{int(time.time())}",
                'files_to_delete': [],
                'files_deleted': []
            })
            
            # Tampilkan status proses dimulai
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS['trash']} Memulai pembersihan data {module_type}..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS['trash']} Memulai pembersihan data {module_type}...")
            
            # Notifikasi observer jika tersedia
            _notify_cleanup_start()
            
            # Setup progress tracking
            _setup_progress_tracking()
            
            # Fase 1: Analisis file yang akan dihapus
            files_to_delete = _analyze_files_to_delete(target_dir, temp_dir)
            
            # Tampilkan jumlah file yang akan dihapus
            total_files = len(files_to_delete)
            if logger: logger.info(f"{ICONS['info']} Menemukan {total_files} file untuk dihapus")
            
            # Update progress bar untuk jumlah total file
            if 'progress_bar' in ui_components:
                ui_components['progress_bar'].max = total_files if total_files > 0 else 1
                ui_components['progress_bar'].description = f"Total: {total_files} file"
            
            # Fase 2: Verifikasi operasi pembersihan
            cleanup_state['files_to_delete'] = files_to_delete
            
            # Tidak ada file yang perlu dihapus
            if total_files == 0:
                with ui_components['status']:
                    display(create_status_indicator("warning", 
                        f"{ICONS['warning']} Tidak ada file yang perlu dihapus"))
                    
                update_status_panel(ui_components, "warning", 
                    f"{ICONS['warning']} Tidak ada file yang perlu dihapus")
                
                # Selesai
                _cleanup_ui()
                return
                
            # Fase 3: Hapus file dengan progress tracking
            deleted_files = _delete_files(files_to_delete)
            cleanup_state['files_deleted'] = deleted_files
            
            # Fase 4: Post-cleanup operation
            _perform_post_cleanup(target_dir)
            
            # Tampilkan hasil
            duration = time.time() - cleanup_state['start_time']
            success_message = f"{ICONS['success']} Pembersihan selesai. {len(deleted_files)} file dihapus dalam {duration:.1f} detik"
            
            with ui_components['status']:
                display(create_status_indicator("success", success_message))
                
            update_status_panel(ui_components, "success", success_message)
            
            # Reset UI components related to processed data
            _reset_ui_components()
            
            # Notifikasi observer
            _notify_cleanup_end()
                
        except Exception as e:
            # Tangani error dengan notifikasi yang jelas
            error_message = f"{ICONS['error']} Error saat pembersihan: {str(e)}"
            
            with ui_components['status']:
                display(create_status_indicator("error", error_message))
                
            update_status_panel(ui_components, "error", error_message)
            
            if logger: logger.error(f"{error_message}")
            
            # Notifikasi observer error
            _notify_cleanup_error(str(e))
            
            # Attempt rollback if needed and possible
            if cleanup_state['rollback_needed'] and cleanup_state['files_deleted']:
                _attempt_rollback()
        
        finally:
            # Selalu bersihkan UI dan reset state
            cleanup_state['is_cleaning'] = False
            _cleanup_ui()
    
    # === FUNGSI HELPER INTERNAL ===
    
    def _get_target_directory() -> str:
        """Dapatkan direktori target berdasarkan module type."""
        if module_type == 'preprocessing':
            return ui_components.get('preprocessed_dir', 'data/preprocessed')
        else:
            # Untuk augmentation, target utama adalah preprocessed karena hasil di-merge ke sana
            return ui_components.get('preprocessed_dir', 'data/preprocessed')
    
    def _get_temp_directory() -> Optional[str]:
        """Dapatkan direktori temp (hanya untuk augmentation)."""
        if module_type == 'augmentation':
            return ui_components.get('augmented_dir', 'data/augmented')
        return None
    
    def _setup_progress_tracking() -> None:
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
    
    def _notify_cleanup_start() -> None:
        """Notifikasi observer tentang memulai pembersihan."""
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Tentukan event type berdasarkan module type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            
            notify(
                event_type=getattr(EventTopics, f"{event_type_prefix}_CLEANUP_START"),
                sender=f"{module_type}_handler",
                message=f"Memulai pembersihan data {module_type}",
                transaction_id=cleanup_state['transaction_id']
            )
        except ImportError:
            pass
    
    def _notify_cleanup_end() -> None:
        """Notifikasi observer tentang selesai pembersihan."""
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Tentukan event type berdasarkan module type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            
            notify(
                event_type=getattr(EventTopics, f"{event_type_prefix}_CLEANUP_END"),
                sender=f"{module_type}_handler",
                message=f"Pembersihan data {module_type} selesai",
                transaction_id=cleanup_state['transaction_id'],
                files_deleted=len(cleanup_state['files_deleted']),
                duration=time.time() - cleanup_state['start_time']
            )
        except ImportError:
            pass
    
    def _notify_cleanup_error(error_message: str) -> None:
        """Notifikasi observer tentang error saat pembersihan."""
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Tentukan event type berdasarkan module type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            
            notify(
                event_type=getattr(EventTopics, f"{event_type_prefix}_CLEANUP_ERROR"),
                sender=f"{module_type}_handler",
                message=f"Error saat pembersihan data {module_type}: {error_message}",
                transaction_id=cleanup_state['transaction_id'],
                error=error_message
            )
        except ImportError:
            pass
    
    def _update_cleanup_progress(step: int, total_steps: int, message: str) -> None:
        """
        Update UI untuk progress pembersihan.
        
        Args:
            step: Step saat ini (1-based)
            total_steps: Total steps
            message: Pesan progress
        """
        # Update current progress
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = step
            ui_components['current_progress'].max = total_steps
            ui_components['current_progress'].description = f"Step {step}/{total_steps}"
        
        # Update progress bar jika tersedia
        if 'progress_bar' in ui_components:
            percentage = int((step / total_steps) * 100)
            ui_components['progress_bar'].value = percentage
            ui_components['progress_bar'].description = f"Total: {percentage}%"
        
        # Log message ke UI
        with ui_components['status']:
            display(create_status_indicator("info", f"{ICONS['processing']} {message}"))
        
        # Log ke logger jika tersedia
        if logger:
            logger.info(f"{ICONS['processing']} {message}")
    
    def _analyze_files_to_delete(target_dir: str, temp_dir: Optional[str] = None) -> List[Path]:
        """
        Analisa dan tentukan file yang akan dihapus berdasarkan module_type.
        
        Args:
            target_dir: Direktori target utama
            temp_dir: Direktori temporary (opsional)
            
        Returns:
            List path file yang akan dihapus
        """
        _update_cleanup_progress(1, 4, "Menganalisis file yang akan dihapus...")
        
        files_to_delete = []
        
        # Logic khusus berdasarkan module_type
        if module_type == 'preprocessing':
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
            # Untuk augmentation, hapus hanya file dengan prefix augmentation
            aug_prefix = ui_components['aug_options'].children[2].value if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2 else 'aug'
            
            # Cek di target_dir (preprocessed) untuk setiap split
            target_path = Path(target_dir)
            for split in DEFAULT_SPLITS:
                split_dir = target_path / split
                if not split_dir.exists():
                    continue
                
                # Cek direktori images untuk file augmentasi
                images_dir = split_dir / 'images'
                if images_dir.exists():
                    files_to_delete.extend(list(images_dir.glob(f"{aug_prefix}_*.*")))
                
                # Cek direktori labels untuk file augmentasi
                labels_dir = split_dir / 'labels'
                if labels_dir.exists():
                    files_to_delete.extend(list(labels_dir.glob(f"{aug_prefix}_*.*")))
            
            # Jika ada temp_dir, cek juga di sana
            if temp_dir:
                temp_path = Path(temp_dir)
                
                # Cek direktori images
                images_dir = temp_path / 'images'
                if images_dir.exists():
                    files_to_delete.extend(list(images_dir.glob(f"{aug_prefix}_*.*")))
                
                # Cek direktori labels
                labels_dir = temp_path / 'labels'
                if labels_dir.exists():
                    files_to_delete.extend(list(labels_dir.glob(f"{aug_prefix}_*.*")))
        
        # Step 2 completed
        _update_cleanup_progress(2, 4, f"Ditemukan {len(files_to_delete)} file untuk dihapus")
        
        return files_to_delete
    
    def _delete_files(files_to_delete: List[Path]) -> List[Path]:
        """
        Hapus file dengan progress tracking dan dukungan transaksi.
        
        Args:
            files_to_delete: List path file yang akan dihapus
            
        Returns:
            List path file yang berhasil dihapus
        """
        _update_cleanup_progress(3, 4, "Menghapus file...")
        
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
                    _notify_cleanup_progress(i + 1, total_files)
            except Exception as e:
                # Log error tapi lanjutkan ke file berikutnya
                if logger:
                    logger.warning(f"{ICONS['warning']} Gagal menghapus {file_path.name}: {str(e)}")
        
        return deleted_files
    
    def _notify_cleanup_progress(current: int, total: int) -> None:
        """Notifikasi observer tentang progress pembersihan."""
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Tentukan event type berdasarkan module type
            event_type_prefix = "PREPROCESSING" if module_type == 'preprocessing' else "AUGMENTATION"
            
            notify(
                event_type=getattr(EventTopics, f"{event_type_prefix}_CLEANUP_PROGRESS"),
                sender=f"{module_type}_handler",
                message=f"Menghapus file {current}/{total}",
                transaction_id=cleanup_state['transaction_id'],
                progress=current,
                total=total
            )
        except ImportError:
            pass
    
    def _perform_post_cleanup(target_dir: str) -> None:
        """
        Lakukan operasi pasca pembersihan.
        
        Args:
            target_dir: Direktori target
        """
        _update_cleanup_progress(4, 4, "Finalisasi pembersihan...")
        
        # Operasi khusus berdasarkan module_type
        if module_type == 'preprocessing':
            # Untuk preprocessing, hapus direktori metadata jika ada
            target_path = Path(target_dir)
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
    
    def _reset_ui_components() -> None:
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
    
    def _attempt_rollback() -> None:
        """Coba rollback jika operasi pembersihan gagal."""
        if not cleanup_state['files_deleted']:
            return
            
        if logger:
            logger.warning(f"{ICONS['warning']} Rollback tidak tersedia untuk operasi hapus file")
    
    def _cleanup_ui() -> None:
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
        disable_buttons(False)
    
    # Register handler untuk tombol cleanup
    if 'cleanup_button' in ui_components:
        ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup,
        'cancel_cleanup': cancel_cleanup,
        'cleanup_state': cleanup_state
    })
    
    return ui_components