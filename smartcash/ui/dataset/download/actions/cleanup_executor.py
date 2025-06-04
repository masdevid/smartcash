"""
File: smartcash/ui/dataset/download/actions/cleanup_executor.py
Deskripsi: Cleanup action executor dengan confirmation dialog dan comprehensive cleanup
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.ui.dataset.download.utils.dataset_checker import check_complete_dataset_status
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute cleanup dengan confirmation dialog."""
    logger = ui_components.get('logger')
    
    try:
        logger and logger.info("🧹 Memulai analisis untuk cleanup")
        _clear_ui_outputs(ui_components)
        
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation']('cleanup')
        
        # Analyze current dataset
        _update_cleanup_progress(ui_components, 20, "📊 Menganalisis dataset yang ada...")
        dataset_status = check_complete_dataset_status()
        
        if not _has_data_to_cleanup(dataset_status):
            logger and logger.info("ℹ️ Tidak ada data untuk dihapus")
            _complete_cleanup_progress(ui_components, "Tidak ada data untuk dihapus")
            return
        
        # Show confirmation dialog
        _show_cleanup_confirmation(ui_components, dataset_status)
        
    except Exception as e:
        logger and logger.error(f"❌ Error cleanup preparation: {str(e)}")
        _error_cleanup_progress(ui_components, f"Error preparation: {str(e)}")

def _has_data_to_cleanup(dataset_status: Dict[str, Any]) -> bool:
    """Check apakah ada data yang bisa di-cleanup."""
    final_dataset = dataset_status['final_dataset']
    downloads_folder = dataset_status['downloads_folder']
    
    return ((final_dataset['exists'] and final_dataset['total_images'] > 0) or 
            (downloads_folder['exists'] and downloads_folder['total_files'] > 0))

def _show_cleanup_confirmation(ui_components: Dict[str, Any], dataset_status: Dict[str, Any]) -> None:
    """Show confirmation dialog dengan detail info."""
    final_dataset = dataset_status['final_dataset']
    downloads_folder = dataset_status['downloads_folder']
    storage_info = dataset_status['storage_info']
    
    # Build cleanup info
    cleanup_items = []
    total_files = 0
    
    if final_dataset['exists']:
        cleanup_items.append(f"📁 Dataset Final: {final_dataset['total_images']} gambar, {final_dataset['total_labels']} label")
        total_files += final_dataset['total_images'] + final_dataset['total_labels']
    
    if downloads_folder['exists']:
        cleanup_items.append(f"📥 Downloads: {downloads_folder['total_files']} files ({downloads_folder['total_size_mb']} MB)")
        total_files += downloads_folder['total_files']
    
    storage_warning = "⚠️ Data akan dihapus PERMANEN dari Google Drive!" if storage_info['persistent'] else "ℹ️ Data akan dihapus dari storage lokal"
    
    message = (
        f"🗑️ Konfirmasi Cleanup Dataset\n\n"
        f"Data yang akan dihapus:\n" + '\n'.join(cleanup_items) + 
        f"\n\n📊 Total: ~{total_files} files\n"
        f"📁 Storage: {storage_info['type']}\n"
        f"{storage_warning}\n\n"
        f"❗ Tindakan ini TIDAK DAPAT DIBATALKAN!\n"
        f"Lanjutkan cleanup?"
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        execute_cleanup_confirmed(ui_components, dataset_status)
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        logger = ui_components.get('logger')
        logger and logger.info("❌ Cleanup dibatalkan oleh user")
        if 'hide_container' in ui_components:
            ui_components['hide_container']()
    
    dialog = create_destructive_confirmation(
        title="🗑️ Konfirmasi Cleanup Dataset", message=message,
        on_confirm=on_confirm, on_cancel=on_cancel, item_name="Dataset",
        confirm_text="Ya, Hapus Semua Data", cancel_text="Batal"
    )
    
    # Show dialog
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def execute_cleanup_confirmed(ui_components: Dict[str, Any], dataset_status: Dict[str, Any] = None) -> None:
    """Execute cleanup setelah konfirmasi."""
    logger = ui_components.get('logger')
    
    try:
        logger and logger.info("🗑️ Memulai proses cleanup")
        
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation']('cleanup')
        
        _update_cleanup_progress(ui_components, 10, "🔧 Mempersiapkan cleanup...")
        
        if not dataset_status:
            dataset_status = check_complete_dataset_status()
        
        organizer = DatasetOrganizer(logger=logger)
        
        # Setup progress callback
        def cleanup_callback(step: str, current: int, total: int, message: str):
            if step == 'cleanup':
                step_progress = 20 + int((current / max(total, 1)) * 60)
                _update_cleanup_progress(ui_components, step_progress, f"Cleanup: {message}")
                
                if 'update_progress' in ui_components:
                    current_progress = int((current / max(total, 1)) * 100)
                    ui_components['update_progress']('current', current_progress, f"🗂️ {message}")
        
        organizer.set_progress_callback(cleanup_callback)
        
        # Execute cleanup
        _update_cleanup_progress(ui_components, 70, "🗑️ Menjalankan cleanup...")
        result = organizer.cleanup_all_dataset_folders()
        
        # Verification
        _update_cleanup_progress(ui_components, 90, "✅ Memverifikasi hasil cleanup...")
        
        # Display results
        if result['status'] == 'success':
            total_deleted = result.get('total_files_removed', 0)
            if total_deleted > 0:
                _complete_cleanup_progress(ui_components, f"Cleanup berhasil: {total_deleted} files dihapus")
                logger and logger.success(f"🎉 Cleanup berhasil: {total_deleted} files dihapus")
            else:
                message = result.get('message', 'Tidak ada file untuk dihapus')
                _complete_cleanup_progress(ui_components, message)
                logger and logger.info(f"ℹ️ {message}")
        else:
            error_msg = result.get('message', 'Cleanup gagal')
            _error_cleanup_progress(ui_components, error_msg)
            logger and logger.error(f"❌ {error_msg}")
        
    except Exception as e:
        logger and logger.error(f"❌ Error cleanup: {str(e)}")
        _error_cleanup_progress(ui_components, f"Error cleanup: {str(e)}")

def _update_cleanup_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update cleanup progress."""
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, f"🧹 {message}")

def _complete_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete cleanup progress."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](f"🧹 {message}")

def _error_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](f"🧹 {message}")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except: pass