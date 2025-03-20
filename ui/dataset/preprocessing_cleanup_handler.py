"""
File: smartcash/ui/dataset/preprocessing_cleanup_handler.py
Deskripsi: Handler yang disederhanakan untuk membersihkan data hasil preprocessing
"""

from typing import Dict, Any
from IPython.display import display, clear_output
import shutil
from pathlib import Path
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alerts import create_status_indicator, create_info_alert

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
                    "Apakah Anda yakin ingin menghapus semua data hasil preprocessing? Tindakan ini tidak dapat dibatalkan.",
                    on_confirm_cleanup, on_cancel_cleanup, "Ya, Hapus Data", "Batal"
                )
                
                with ui_components['status']:
                    clear_output(wait=True)
                    display(dialog)
                return
                
            except ImportError:
                # Lanjutkan tanpa konfirmasi jika fungsi tidak tersedia
                with ui_components['status']: clear_output(wait=True)
                perform_cleanup()
                
        except Exception as e:
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
    
    # Fungsi untuk melakukan cleanup sebenarnya
    def perform_cleanup():
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
        
        try:
            # Tampilkan status
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Membersihkan data preprocessing..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Membersihkan data preprocessing...")
            
            # Pembersihan dengan dataset manager jika tersedia
            dataset_manager = ui_components.get('dataset_manager')
            if dataset_manager and hasattr(dataset_manager, 'clean_preprocessed'):
                # Bersihkan semua split sekaligus
                for split in ['train', 'valid', 'test']: dataset_manager.clean_preprocessed(split)
                success = True
            else:
                # Fallback manual tanpa dataset manager
                path = Path(preprocessed_dir)
                if path.exists():
                    shutil.rmtree(path)
                    path.mkdir(parents=True, exist_ok=True)
                    success = True
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
                
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
            
            update_status_panel(ui_components, "error", 
                f"{ICONS.get('error', '❌')} Gagal membersihkan data: {str(e)}")
            
            if logger: logger.error(f"{ICONS.get('error', '❌')} Error saat membersihkan data: {str(e)}")
    
    # Register handler
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Tambahkan fungsi ke ui_components
    ui_components.update({
        'on_cleanup_click': on_cleanup_click,
        'perform_cleanup': perform_cleanup
    })
    
    return ui_components