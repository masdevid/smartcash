"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Enhanced cleanup action dengan confirmation dialog dan comprehensive progress tracking
"""

from typing import Dict, Any
from IPython.display import display
from smartcash.ui.dataset.download.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.download.utils.dataset_checker import check_complete_dataset_status
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute cleanup dengan enhanced confirmation dan progress tracking."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    # Check if operation can start
    can_start, message = button_manager.can_start_operation('cleanup')
    if not can_start:
        logger and logger.warning(f"⚠️ {message}")
        return
    
    try:
        logger and logger.info("🧹 Memulai analisis untuk cleanup")
        _clear_ui_outputs(ui_components)
        
        # Show progress container for cleanup operation
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation']('cleanup')
        
        # Analyze current dataset status
        _update_cleanup_progress(ui_components, 20, "📊 Menganalisis dataset yang ada...")
        dataset_status = check_complete_dataset_status()
        
        if not _has_data_to_cleanup(dataset_status):
            logger and logger.info("ℹ️ Tidak ada data untuk dihapus")
            _complete_cleanup_progress(ui_components, "Tidak ada data untuk dihapus")
            return
        
        # Show enhanced confirmation dialog
        _show_enhanced_cleanup_confirmation(ui_components, dataset_status)
        
    except Exception as e:
        logger and logger.error(f"❌ Error cleanup preparation: {str(e)}")
        _error_cleanup_progress(ui_components, f"Error preparation: {str(e)}")

def _has_data_to_cleanup(dataset_status: Dict[str, Any]) -> bool:
    """Check apakah ada data yang bisa di-cleanup."""
    final_dataset = dataset_status['final_dataset']
    downloads_folder = dataset_status['downloads_folder']
    
    return (final_dataset['exists'] and final_dataset['total_images'] > 0) or \
           (downloads_folder['exists'] and downloads_folder['total_files'] > 0)

def _show_enhanced_cleanup_confirmation(ui_components: Dict[str, Any], 
                                      dataset_status: Dict[str, Any]) -> None:
    """Show enhanced confirmation dialog dengan detailed info."""
    
    final_dataset = dataset_status['final_dataset']
    downloads_folder = dataset_status['downloads_folder']
    storage_info = dataset_status['storage_info']
    
    # Build detailed message
    cleanup_items = []
    total_files = 0
    
    if final_dataset['exists']:
        cleanup_items.append(f"📁 Dataset Final: {final_dataset['total_images']} gambar, {final_dataset['total_labels']} label")
        total_files += final_dataset['total_images'] + final_dataset['total_labels']
        
        for split, split_info in final_dataset['splits'].items():
            if split_info['exists'] and split_info['images'] > 0:
                cleanup_items.append(f"   • {split}: {split_info['images']} gambar")
    
    if downloads_folder['exists']:
        cleanup_items.append(f"📥 Downloads: {downloads_folder['total_files']} files ({downloads_folder['total_size_mb']} MB)")
        total_files += downloads_folder['total_files']
    
    storage_warning = "⚠️ Data akan dihapus PERMANEN dari Google Drive!" if storage_info['persistent'] else \
                     "ℹ️ Data akan dihapus dari storage lokal"
    
    message = (
        f"🗑️ Konfirmasi Cleanup Dataset\n\n"
        f"Data yang akan dihapus:\n" + '\n'.join(cleanup_items) + 
        f"\n\n📊 Total: ~{total_files} files\n"
        f"📁 Storage: {storage_info['type']}\n"
        f"{storage_warning}\n\n"
        f"❗ Tindakan ini TIDAK DAPAT DIBATALKAN!\n"
        f"Pastikan Anda telah mem-backup data penting.\n\n"
        f"Lanjutkan cleanup?"
    )
    
    def on_confirm(b):
        ui_components['confirmation_area'].clear_output()
        try:
            execute_cleanup_confirmed(ui_components, dataset_status)
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Error execute cleanup: {str(e)}")
            _error_cleanup_progress(ui_components, f"Error execute cleanup: {str(e)}")
    
    def on_cancel(b):
        ui_components['confirmation_area'].clear_output()
        logger = ui_components.get('logger')
        logger and logger.info("❌ Cleanup dibatalkan oleh user")
        
        # Hide progress and reset
        if 'hide_container' in ui_components:
            ui_components['hide_container']()
        
        # No need to manually enable buttons as operation_context will handle it
    
    dialog = create_destructive_confirmation(
        title="🗑️ Konfirmasi Cleanup Dataset",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        item_name="Dataset",
        confirm_text="Ya, Hapus Semua Data",
        cancel_text="Batal"
    )
    
    # Show dialog
    ui_components['confirmation_area'].clear_output()
    with ui_components['confirmation_area']:
        display(dialog)

def execute_cleanup_confirmed(ui_components: Dict[str, Any], 
                            dataset_status: Dict[str, Any] = None) -> None:
    """Execute cleanup dengan comprehensive progress tracking."""
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('cleanup'):
        try:
            logger and logger.info("🗑️ Memulai proses cleanup komprehensif")
            
            # Ensure progress is shown for cleanup operation
            if 'show_for_operation' in ui_components:
                ui_components['show_for_operation']('cleanup')
            
            _update_cleanup_progress(ui_components, 10, "🔧 Mempersiapkan cleanup...")
            
            # Get fresh status jika tidak disediakan
            if not dataset_status:
                dataset_status = check_complete_dataset_status()
            
            organizer = DatasetOrganizer(logger=logger)
            
            # Enhanced progress callback dengan step dan current tracking
            cleanup_results = {'total_deleted': 0, 'errors': [], 'completed_steps': []}
            
            def enhanced_cleanup_callback(step: str, current: int, total: int, message: str):
                """Enhanced callback dengan multi-level progress tracking."""
                # Update current progress untuk detail per folder/file
                if 'update_progress' in ui_components:
                    current_progress = int((current / max(total, 1)) * 100)
                    ui_components['update_progress']('current', current_progress, f"🗂️ {message}")
                
                # Update step progress
                if step == 'cleanup':
                    step_progress = 20 + int((current / max(total, 1)) * 60)  # 20-80% range
                    _update_cleanup_progress(ui_components, step_progress, f"Cleanup: {message}")
                    
                    # Update step bar
                    if 'update_progress' in ui_components:
                        ui_components['update_progress']('step', current_progress, f"🧹 {message}")
                
                # Track progress untuk final summary
                cleanup_results['total_deleted'] = current
            
            organizer.set_progress_callback(enhanced_cleanup_callback)
            
            # Execute cleanup phases
            _execute_cleanup_phases(ui_components, organizer, dataset_status, cleanup_results)
            
            # Final summary and completion
            _display_cleanup_summary(ui_components, cleanup_results)
            
            # Complete progress tracking
            if cleanup_results.get('success', False):
                total_deleted = cleanup_results.get('total_files_removed', 0)
                if total_deleted > 0:
                    _complete_cleanup_progress(ui_components, f"Cleanup berhasil: {total_deleted} files dihapus")
                else:
                    message = cleanup_results.get('message', 'Tidak ada file untuk dihapus')
                    _complete_cleanup_progress(ui_components, message)
            else:
                errors = cleanup_results.get('errors', [])
                error_msg = errors[0] if errors else "Cleanup gagal"
                _error_cleanup_progress(ui_components, error_msg)
            
        except Exception as e:
            logger and logger.error(f"❌ Error cleanup: {str(e)}")
            _error_cleanup_progress(ui_components, f"Error cleanup: {str(e)}")
            raise

def _execute_cleanup_phases(ui_components: Dict[str, Any], 
                          organizer: DatasetOrganizer,
                          dataset_status: Dict[str, Any],
                          cleanup_results: Dict[str, Any]) -> None:
    """Execute cleanup dalam phases dengan detailed tracking."""
    
    _update_cleanup_progress(ui_components, 20, "📋 Menghitung files yang akan dihapus...")
    
    # Phase 1: Dataset final structure
    if dataset_status['final_dataset']['exists']:
        _update_cleanup_progress(ui_components, 30, "🗂️ Membersihkan dataset struktur final...")
        if 'update_progress' in ui_components:
            ui_components['update_progress']('step', 25, "🗂️ Membersihkan train/valid/test folders")
        
        cleanup_results['completed_steps'].append("Dataset final structure")
    
    # Phase 2: Downloads folder  
    if dataset_status['downloads_folder']['exists']:
        _update_cleanup_progress(ui_components, 50, "📥 Membersihkan downloads folder...")
        if 'update_progress' in ui_components:
            ui_components['update_progress']('step', 50, "📥 Membersihkan downloads folder")
        
        cleanup_results['completed_steps'].append("Downloads folder")
    
    # Phase 3: Execute actual cleanup
    _update_cleanup_progress(ui_components, 70, "🗑️ Menjalankan cleanup...")
    result = organizer.cleanup_all_dataset_folders()
    
    # Phase 4: Verification
    _update_cleanup_progress(ui_components, 90, "✅ Memverifikasi hasil cleanup...")
    if 'update_progress' in ui_components:
        ui_components['update_progress']('step', 100, "✅ Verifikasi cleanup selesai")
    
    # Process results
    if result['status'] == 'success':
        cleanup_results.update(result['stats'])
        cleanup_results['success'] = True
    elif result['status'] == 'empty':
        cleanup_results['message'] = result['message']
        cleanup_results['success'] = True
    else:
        cleanup_results['errors'].append(result.get('message', 'Cleanup gagal'))
        cleanup_results['success'] = False

def _display_cleanup_summary(ui_components: Dict[str, Any], 
                           cleanup_results: Dict[str, Any]) -> None:
    """Display comprehensive cleanup summary."""
    logger = ui_components.get('logger')
    if not logger:
        return
    
    if cleanup_results.get('success', False):
        total_deleted = cleanup_results.get('total_files_removed', 0)
        
        if total_deleted > 0:
            logger.success(f"🎉 Cleanup berhasil: {total_deleted} files dihapus")
            
            folders_cleaned = cleanup_results.get('folders_cleaned', [])
            if folders_cleaned:
                logger.info(f"📁 Folders yang dibersihkan:")
                for folder in folders_cleaned:
                    logger.info(f"   • {folder}")
        else:
            message = cleanup_results.get('message', 'Tidak ada file untuk dihapus')
            logger.info(f"ℹ️ {message}")
        
        # Show completed steps
        completed_steps = cleanup_results.get('completed_steps', [])
        if completed_steps:
            logger.info(f"✅ Steps completed: {', '.join(completed_steps)}")
    
    # Show errors if any
    errors = cleanup_results.get('errors', [])
    if errors:
        logger.warning(f"⚠️ Errors encountered:")
        for error in errors[:3]:  # Show first 3 errors
            logger.info(f"   • {error}")
        if len(errors) > 3:
            logger.info(f"   • ... dan {len(errors) - 3} errors lainnya")

def _update_cleanup_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update cleanup progress dengan progress tracking system."""
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, f"🧹 {message}")

def _complete_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete cleanup progress dengan success state."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](f"🧹 {message}")

def _error_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state untuk cleanup progress."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](f"🧹 {message}")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass