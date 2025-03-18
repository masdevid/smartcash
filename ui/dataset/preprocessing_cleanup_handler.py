"""
File: smartcash/ui/dataset/preprocessing_cleanup_handler.py
Deskripsi: Handler untuk membersihkan data hasil preprocessing
"""

from typing import Dict, Any
from IPython.display import display, HTML, clear_output
import shutil
from pathlib import Path

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
    try:
        from smartcash.ui.components.alerts import create_status_indicator, create_info_alert
        from smartcash.ui.utils.constants import ICONS
        from smartcash.ui.dataset.preprocessing_initialization import update_status_panel
    except ImportError:
        def create_status_indicator(status, message):
            icons = {'success': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️'}
            icon = icons.get(status, 'ℹ️')
            return HTML(f"<div style='padding:8px'>{icon} {message}</div>")
        
        def create_info_alert(message, alert_type='info', icon=None):
            return HTML(f"<div style='padding:8px'>{icon or '🔍'} {message}</div>")
            
        def update_status_panel(ui_components, status_type, message):
            pass
            
        ICONS = {
            'trash': '🗑️',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️'
        }
    
    # Konfirmasi dialog untuk cleanup
    def create_cleanup_confirmation():
        """Buat dialog konfirmasi untuk cleanup."""
        try:
            from smartcash.ui.components.helpers import create_confirmation_dialog
            
            def on_confirm_cleanup():
                with ui_components['status']:
                    clear_output(wait=True)
                perform_cleanup()
            
            def on_cancel_cleanup():
                with ui_components['status']:
                    display(create_status_indicator("info", f"{ICONS.get('info', 'ℹ️')} Cleanup dibatalkan"))
            
            return create_confirmation_dialog(
                "Konfirmasi Pembersihan Data",
                "Apakah Anda yakin ingin menghapus semua data hasil preprocessing? Tindakan ini tidak dapat dibatalkan.",
                on_confirm_cleanup,
                on_cancel_cleanup,
                "Ya, Hapus Data",
                "Batal"
            )
        except ImportError:
            # Fallback jika tidak dapat membuat dialog konfirmasi
            return None
    
    # Perform cleanup
    def perform_cleanup():
        """Bersihkan data hasil preprocessing."""
        try:
            # Dapatkan direktori preprocessed
            preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
            
            # Tampilkan status
            with ui_components['status']:
                display(create_status_indicator("info", f"{ICONS.get('trash', '🗑️')} Membersihkan data preprocessing..."))
            
            # Update status panel
            update_status_panel(ui_components, "info", f"{ICONS.get('trash', '🗑️')} Membersihkan data preprocessing...")
            
            # Pembersihan dengan dataset manager jika tersedia
            dataset_manager = ui_components.get('dataset_manager')
            if dataset_manager and hasattr(dataset_manager, 'clean_preprocessed'):
                # Bersihkan semua split
                splits = ['train', 'valid', 'test']
                for split in splits:
                    dataset_manager.clean_preprocessed(split)
                
                # Status berhasil
                with ui_components['status']:
                    display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Data preprocessing berhasil dibersihkan"))
                
                # Update status panel
                update_status_panel(ui_components, "success", f"{ICONS.get('success', '✅')} Data preprocessing berhasil dibersihkan")
                
                # Sembunyikan tombol cleanup
                ui_components['cleanup_button'].layout.display = 'none'
                
                # Sembunyikan summary container
                if 'summary_container' in ui_components:
                    ui_components['summary_container'].layout.display = 'none'
                
                return
            
            # Fallback manual jika dataset manager tidak tersedia
            try:
                path = Path(preprocessed_dir)
                if path.exists():
                    # Hapus direktori
                    shutil.rmtree(path)
                    
                    # Buat direktori baru
                    path.mkdir(parents=True, exist_ok=True)
                    
                    # Status berhasil
                    with ui_components['status']:
                        display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Data preprocessing berhasil dibersihkan"))
                    
                    # Update status panel
                    update_status_panel(ui_components, "success", f"{ICONS.get('success', '✅')} Data preprocessing berhasil dibersihkan")
                    
                    # Sembunyikan tombol cleanup
                    ui_components['cleanup_button'].layout.display = 'none'
                    
                    # Sembunyikan summary container
                    if 'summary_container' in ui_components:
                        ui_components['summary_container'].layout.display = 'none'
                else:
                    with ui_components['status']:
                        display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Direktori preprocessing tidak ditemukan: {preprocessed_dir}"))
            except Exception as e:
                with ui_components['status']:
                    display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
                
                # Update status panel
                update_status_panel(ui_components, "error", f"{ICONS.get('error', '❌')} Gagal membersihkan data: {str(e)}")
                
                # Log error
                if 'logger' in ui_components:
                    ui_components['logger'].error(f"{ICONS.get('error', '❌')} Error saat membersihkan data preprocessing: {str(e)}")
                
        except Exception as e:
            with ui_components['status']:
                display(create_status_indicator("error", f"{ICONS.get('error', '❌')} Error: {str(e)}"))
            
            # Update status panel
            update_status_panel(ui_components, "error", f"{ICONS.get('error', '❌')} Gagal membersihkan data: {str(e)}")
            
            # Log error
            if 'logger' in ui_components:
                ui_components['logger'].error(f"{ICONS.get('error', '❌')} Error saat membersihkan data preprocessing: {str(e)}")
    
    # Handler untuk tombol cleanup
    def on_cleanup_click(b):
        # Show confirmation dialog if possible
        confirmation_dialog = create_cleanup_confirmation()
        if confirmation_dialog:
            with ui_components['status']:
                clear_output(wait=True)
                display(confirmation_dialog)
        else:
            # No confirmation dialog, proceed directly
            with ui_components['status']:
                clear_output(wait=True)
            perform_cleanup()
    
    # Register button click handler
    ui_components['cleanup_button'].on_click(on_cleanup_click)
    
    # Add reference to handlers in ui_components
    ui_components['on_cleanup_click'] = on_cleanup_click
    ui_components['perform_cleanup'] = perform_cleanup
    
    return ui_components