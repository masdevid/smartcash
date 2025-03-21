"""
File: smartcash/ui/dataset/augmentation_cleanup_handler.py
Deskripsi: Handler untuk membersihkan data hasil augmentasi dengan pattern yang diperbarui
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import shutil, time, re, glob, os
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert

def setup_cleanup_handler(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk tombol cleanup augmentasi data dengan pattern yang ditingkatkan.
    
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
                    with ui_components['status']: 
                        clear_output(wait=True)
                    perform_cleanup()
                
                def on_cancel_cleanup():
                    with ui_components['status']: 
                        clear_output(wait=True)
                        display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
                
                confirmation_dialog = create_confirmation_dialog(
                    "Konfirmasi Pembersihan Data",
                    "Apakah Anda yakin ingin menghapus semua data hasil augmentasi? Tindakan ini tidak dapat dibatalkan.",
                    on_confirm_cleanup, on_cancel_cleanup, "Ya, Hapus Data", "Batal"
                )
                
                with ui_components['status']:
                    clear_output(wait=True)
                    display(confirmation_dialog)
                return
                
            except ImportError:
                # Lanjutkan tanpa konfirmasi jika fungsi tidak tersedia
                with ui_components['status']: 
                    clear_output(wait=True)
                    display(create_info_alert(
                        "Konfirmasi: Anda akan menghapus semua data hasil augmentasi. Lanjutkan?",
                        "warning", ICONS['warning']
                    ))
                    # Tambahkan tombol konfirmasi manual
                    import ipywidgets as widgets
                    confirm_btn = widgets.Button(description="Ya, Hapus Data", button_style="danger", icon="trash")
                    cancel_btn = widgets.Button(description="Batal", button_style="info", icon="times")
                    
                    # Definisi handler untuk tombol konfirmasi/batal
                    def on_confirm(b):
                        with ui_components['status']: clear_output(wait=True)
                        perform_cleanup()
                    
                    def on_cancel(b):
                        with ui_components['status']: 
                            clear_output(wait=True)
                            display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
                    
                    confirm_btn.on_click(on_confirm)
                    cancel_btn.on_click(on_cancel)
                    
                    buttons_box = widgets.HBox([confirm_btn, cancel_btn], 
                                             layout=widgets.Layout(justify_content="center", margin="10px 0"))
                    display(buttons_box)
                return
                
        except Exception as e:
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
    
    # Fungsi untuk melakukan cleanup sebenarnya dengan proteksi data preprocessing
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
            
            # Dapatkan augmentation pattern dari config
            augmentation_patterns = []
            cleanup_config = config.get('cleanup', {})
            
            # Fallback pattern jika tidak ada dalam config
            default_patterns = ['aug_.*', '.*_augmented.*', '.*_modified.*']
            
            # Cek source config untuk patterns
            if 'augmentation_patterns' in cleanup_config:
                augmentation_patterns = cleanup_config['augmentation_patterns']
            else:
                # Coba ambil dari configurasi lama
                augmentation_patterns = cleanup_config.get('patterns', default_patterns)
                
            # Pastikan selalu ada pattern default
            if not augmentation_patterns:
                augmentation_patterns = default_patterns
            
            # Get current output prefix from UI
            current_prefix = ""
            if 'aug_options' in ui_components and len(ui_components['aug_options'].children) > 2:
                current_prefix = ui_components['aug_options'].children[2].value
                if current_prefix and current_prefix not in augmentation_patterns:
                    augmentation_patterns.append(f"{current_prefix}_.*")
                    
            path = Path(augmented_dir)
            
            if path.exists():
                import time
                start_time = time.time()
                
                # Backup sebelum hapus jika diinginkan
                backup_enabled = cleanup_config.get('backup_enabled', True)
                backup_dir = cleanup_config.get('backup_dir', 'data/backup/augmentation')
                
                if backup_enabled:
                    backup_path = Path(backup_dir)
                    backup_path.mkdir(parents=True, exist_ok=True)
                    
                    # Timestamp untuk backup
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    backup_target = backup_path / f"augmented_{timestamp}"
                    
                    # Copy ke backup (hanya jika direktori memiliki konten)
                    if list(path.glob('**/*')):
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
                
                # Jika ada subdirectory images dan labels, hapus file yang sesuai pattern
                images_dir = path / 'images'
                labels_dir = path / 'labels'
                
                deleted_count = 0
                
                # Hapus files berdasarkan pattern, bukan seluruh direktori
                if images_dir.exists() and labels_dir.exists():
                    for pattern in augmentation_patterns:
                        for img_file in list(images_dir.glob('*')):
                            try:
                                # Gunakan re.match untuk pattern dari awal
                                if re.match(pattern, img_file.name):
                                    # Hapus gambar
                                    img_file.unlink()
                                    deleted_count += 1
                                    
                                    # Hapus label yang sesuai
                                    label_file = labels_dir / f"{img_file.stem}.txt"
                                    if label_file.exists():
                                        label_file.unlink()
                                        deleted_count += 1
                            except Exception as e:
                                if logger:
                                    logger.warning(f"⚠️ Error saat proses file {img_file}: {e}")
                
                    
                    # Log hasil penghapusan
                    with ui_components['status']:
                        display(create_status_indicator(
                            "info", 
                            f"🗑️ {deleted_count} file berhasil dihapus sesuai pattern augmentasi"
                        ))
                else:
                    # Hapus keseluruhan direktori jika tidak memiliki struktur standar
                    shutil.rmtree(path, ignore_errors=True)
                
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
                clear_output(wait=True)
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
    
    # Periksa apakah folder augmentasi sudah ada dan berisi data
    def check_augmentation_exists():
        augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
        path = Path(augmented_dir)
        
        # Cek jika direktori images ada dan memiliki file
        images_dir = path / 'images'
        has_files = images_dir.exists() and any(images_dir.glob('*'))
        
        # Tampilkan/sembunyikan tombol cleanup berdasarkan kondisi
        if 'cleanup_button' in ui_components:
            ui_components['cleanup_button'].layout.display = 'inline-block' if has_files else 'none'
            
        return has_files
    
    # Register handler
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Set tombol cleanup visibility berdasarkan kondisi folder
    check_augmentation_exists()
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup,
        'check_augmentation_exists': check_augmentation_exists
    })
    
    return ui_components