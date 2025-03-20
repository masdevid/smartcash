"""
File: smartcash/ui/dataset/augmentation_cleanup_handler.py
Deskripsi: Handler untuk membersihkan data hasil augmentasi dengan EventTopics yang diperbarui
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import shutil
import time
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

def setup_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol cleanup augmentasi data.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Handler untuk tombol cleanup
    def on_cleanup_click(b):
        try:
            # Buat dialog konfirmasi jika tersedia
            try:
                from smartcash.ui.helpers.ui_helpers import create_confirmation_dialog
                
                def on_confirm_cleanup():
                    with ui_components['status']: clear_output(wait=True)
                    perform_cleanup()
                
                def on_cancel_cleanup():
                    with ui_components['status']: 
                        display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
                
                dialog = create_confirmation_dialog(
                    "Konfirmasi Pembersihan Data",
                    "Apakah Anda yakin ingin menghapus semua data hasil augmentasi? Tindakan ini tidak dapat dibatalkan.",
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
                        "Konfirmasi: Anda akan menghapus semua data hasil augmentasi. Lanjutkan?",
                        "warning", ICONS['warning']
                    ))
                    # Tambahkan tombol konfirmasi manual
                    import ipywidgets as widgets
                    confirm_btn = widgets.Button(description="Ya, Hapus Data", button_style="danger", icon="trash")
                    cancel_btn = widgets.Button(description="Batal", button_style="info", icon="times")
                    
                    confirm_btn.on_click(lambda b: perform_cleanup())
                    cancel_btn.on_click(lambda b: cancel_cleanup())
                    
                    display(widgets.HBox([confirm_btn, cancel_btn], layout=widgets.Layout(justify_content="center", margin="10px 0")))
                return
                
        except Exception as e:
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
    
    # Fungsi untuk membatalkan cleanup
    def cancel_cleanup():
        with ui_components['status']: 
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
    
    # Fungsi untuk melakukan cleanup sebenarnya
    def perform_cleanup():
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        from smartcash.ui.dataset.augmentation_initialization import update_status_panel
        
        try:
            # Tampilkan status
            with ui_components['status']:
                clear_output(wait=True)
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Membersihkan data augmentasi..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Membersihkan data augmentasi...")
            
            # Notifikasi observer sebelum cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.AUGMENTATION_CLEANUP_START,
                    sender="augmentation_handler",
                    message=f"Memulai pembersihan data augmentasi"
                )
            except ImportError:
                pass
            
            path = Path(augmented_dir)
            if path.exists():
                import time
                start_time = time.time()
                
                # Backup sebelum hapus jika diinginkan
                backup_dir = config.get('cleanup', {}).get('backup_dir', 'data/backup/augmentation')
                backup_path = Path(backup_dir)
                backup_path.mkdir(parents=True, exist_ok=True)
                
                # Timestamp untuk backup
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_target = backup_path / f"augmented_{timestamp}"
                
                # Copy ke backup
                try:
                    shutil.copytree(path, backup_target)
                    with ui_components['status']:
                        display(create_status_indicator(
                            "info", 
                            f"📦 Data augmentasi di-backup ke: {backup_target}"
                        ))
                except Exception as e:
                    with ui_components['status']:
                        display(create_status_indicator(
                            "warning", 
                            f"⚠️ Gagal membuat backup: {str(e)}"
                        ))
                
                # Hapus direktori augmentasi
                shutil.rmtree(path)
                success = True
                duration = time.time() - start_time
                
                if logger: logger.info(f"{ICONS['success']} Pembersihan data selesai dalam {duration:.2f} detik")
            else:
                with ui_components['status']:
                    display(create_status_indicator("warning", 
                        f"{ICONS.get('warning', '⚠️')} Direktori tidak ditemukan: {augmented_dir}"))
                success = False
            
            # Update UI jika sukses
            if success:
                with ui_components['status']:
                    display(create_status_indicator("success", 
                        f"{ICONS.get('success', '✅')} Data augmentasi berhasil dibersihkan"))
                
                update_status_panel(ui_components, "success", 
                    f"{ICONS.get('success', '✅')} Data augmentasi berhasil dibersihkan")
                
                # Sembunyikan elemen UI yang tidak relevan
                ui_components['cleanup_button'].layout.display = 'none'
                if 'summary_container' in ui_components:
                    ui_components['summary_container'].layout.display = 'none'
                if 'visualization_buttons' in ui_components:
                    ui_components['visualization_buttons'].layout.display = 'none'
                if 'visualization_container' in ui_components:
                    ui_components['visualization_container'].layout.display = 'none'
                
                # Notifikasi observer
                try:
                    from smartcash.components.observer import notify
                    from smartcash.components.observer.event_topics_observer import EventTopics
                    notify(
                        event_type=EventTopics.AUGMENTATION_CLEANUP_END,
                        sender="augmentation_handler",
                        message=f"Pembersihan data augmentasi selesai"
                    )
                except ImportError:
                    pass
                
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
    
    # Register handler
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup,
        'cancel_cleanup': cancel_cleanup
    })
    
    return ui_components