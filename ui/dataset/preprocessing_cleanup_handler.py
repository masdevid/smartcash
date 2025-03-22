"""
File: smartcash/ui/dataset/preprocessing_cleanup_handler.py
Deskripsi: Handler yang disederhanakan untuk membersihkan data hasil preprocessing dengan perbaikan dialog konfirmasi
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import shutil
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

def setup_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol cleanup preprocessing data.
    
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
        buttons = [
            'preprocess_button', 'cleanup_button', 'save_button', 
            'visualize_button', 'compare_button', 'distribution_button'
        ]
        
        for btn_name in buttons:
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
                        clear_output(wait=True)  # PERBAIKAN: Clear output sebelum menampilkan pesan batal
                        display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
                    # Aktifkan kembali tombol setelah batal
                    disable_buttons(False)
                
                # Buat dialog konfirmasi
                dialog = create_confirmation_dialog(
                    "Konfirmasi Pembersihan Data",
                    "Apakah Anda yakin ingin menghapus semua data hasil preprocessing? Tindakan ini tidak dapat dibatalkan.",
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
                        "Konfirmasi: Anda akan menghapus semua data hasil preprocessing. Lanjutkan?",
                        "warning", ICONS['warning']
                    ))
                    # Tambahkan tombol konfirmasi manual
                    import ipywidgets as widgets
                    confirm_btn = widgets.Button(description="Ya, Hapus Data", button_style="danger", icon="trash")
                    cancel_btn = widgets.Button(description="Batal", button_style="info", icon="times")
                    
                    # PERBAIKAN: Gunakan lambda dengan parameter untuk menghindari closure yang salah
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
            clear_output(wait=True)  # PERBAIKAN: Clear output untuk menghilangkan dialog
            display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
        # Aktifkan kembali tombol setelah batal
        disable_buttons(False)
    
    # Fungsi untuk melakukan cleanup sebenarnya
    def perform_cleanup():
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
        
        try:
            # Tampilkan status proses dimulai
            with ui_components['status']:
                clear_output(wait=True)  # PERBAIKAN: Clear output untuk menghilangkan dialog
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Membersihkan data preprocessing..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Membersihkan data preprocessing...")
            
            # Notifikasi observer sebelum cleanup
            try:
                from smartcash.components.observer import notify
                from smartcash.components.observer.event_topics_observer import EventTopics
                notify(
                    event_type=EventTopics.PREPROCESSING_CLEANUP_START,
                    sender="preprocessing_handler",
                    message=f"Memulai pembersihan data preprocessing"
                )
            except ImportError:
                pass
            
            # Pembersihan dengan dataset manager jika tersedia
            dataset_manager = ui_components.get('dataset_manager')
            if dataset_manager and hasattr(dataset_manager, 'clean_preprocessed'):
                # Bersihkan semua split sekaligus
                dataset_manager.clean_preprocessed(split='all')
                success = True
            else:
                # Fallback manual tanpa dataset manager
                path = Path(preprocessed_dir)
                if path.exists():
                    import time
                    start_time = time.time()
                    
                    # Coba gunakan storage preprocessed jika tersedia
                    try:
                        from smartcash.dataset.services.preprocessor.storage import PreprocessedStorage
                        storage = PreprocessedStorage(preprocessed_dir, logger=logger)
                        storage.clean_storage()
                        success = True
                    except ImportError:
                        # Fallback: hapus direktori dan buat kembali
                        shutil.rmtree(path)
                        path.mkdir(parents=True, exist_ok=True)
                        success = True
                        
                    duration = time.time() - start_time
                    if logger: logger.info(f"{ICONS['success']} Pembersihan data selesai dalam {duration:.2f} detik")
                else:
                    with ui_components['status']:
                        display(create_status_indicator("warning", 
                            f"{ICONS.get('warning', '⚠️')} Direktori tidak ditemukan: {preprocessed_dir}"))
                    success = False
            
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
                if 'visualize_button' in ui_components:
                    ui_components['visualize_button'].layout.display = 'none'
                if 'compare_button' in ui_components:
                    ui_components['compare_button'].layout.display = 'none'
                if 'distribution_button' in ui_components:
                    ui_components['distribution_button'].layout.display = 'none'
                
                # Notifikasi observer
                try:
                    from smartcash.components.observer import notify
                    from smartcash.components.observer.event_topics_observer import EventTopics
                    notify(
                        event_type=EventTopics.PREPROCESSING_CLEANUP_END,
                        sender="preprocessing_handler",
                        message=f"Pembersihan data preprocessing selesai"
                    )
                except ImportError:
                    pass
                
            # Aktifkan kembali tombol preprocess setelah cleanup selesai
            ui_components['preprocess_button'].disabled = False
                
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
    
    # Register handler
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup,
        'cancel_cleanup': cancel_cleanup,
        'disable_buttons': disable_buttons
    })
    
    return ui_components