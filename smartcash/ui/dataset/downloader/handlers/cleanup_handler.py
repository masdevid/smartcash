"""
File: smartcash/ui/dataset/downloader/handlers/cleanup_handler.py
Deskripsi: Handler untuk cleanup operation dengan konfirmasi dan progress tracking
"""

from typing import Dict, Any, Callable
import threading
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
from smartcash.dataset.utils.path_validator import get_path_validator

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Callable:
    """Setup cleanup handler dengan konfirmasi destruktif"""
    
    def handle_cleanup(button):
        """Handle cleanup operation dengan konfirmasi"""
        button.disabled = True
        
        try:
            # Check existing data untuk konfirmasi
            cleanup_info = _get_cleanup_info()
            
            if cleanup_info['total_files'] == 0:
                show_status_safe("ℹ️ Tidak ada file untuk dibersihkan", "info", ui_components)
                return
            
            # Show confirmation dialog
            _show_cleanup_confirmation(ui_components, cleanup_info, logger)
            
        except Exception as e:
            logger.error(f"❌ Error cleanup handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            button.disabled = False
    
    return handle_cleanup

def _get_cleanup_info() -> Dict[str, Any]:
    """Get informasi tentang file yang akan dibersihkan"""
    try:
        path_validator = get_path_validator()
        dataset_paths = path_validator.get_dataset_paths()
        
        # Count files in each directory
        from pathlib import Path
        
        cleanup_targets = {
            'train': dataset_paths['train'],
            'valid': dataset_paths['valid'], 
            'test': dataset_paths['test'],
            'downloads': dataset_paths['downloads'],
            'preprocessed': dataset_paths.get('preprocessed', ''),
            'augmented': dataset_paths.get('augmented', '')
        }
        
        total_files = 0
        target_info = {}
        
        for target, path_str in cleanup_targets.items():
            if not path_str:
                continue
                
            path = Path(path_str)
            if path.exists():
                file_count = sum(1 for f in path.rglob('*') if f.is_file())
                total_files += file_count
                target_info[target] = {
                    'path': str(path),
                    'files': file_count,
                    'exists': True
                }
            else:
                target_info[target] = {
                    'path': str(path),
                    'files': 0,
                    'exists': False
                }
        
        return {
            'total_files': total_files,
            'targets': target_info,
            'has_data': total_files > 0
        }
        
    except Exception as e:
        return {
            'total_files': 0,
            'targets': {},
            'has_data': False,
            'error': str(e)
        }

def _show_cleanup_confirmation(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Show destructive confirmation untuk cleanup"""
    
    total_files = cleanup_info['total_files']
    targets_with_data = [name for name, info in cleanup_info['targets'].items() if info['files'] > 0]
    
    confirmation_dialog = create_destructive_confirmation(
        title="Konfirmasi Cleanup Dataset",
        message=f"""⚠️ OPERASI DESTRUKTIF ⚠️

Akan menghapus {total_files:,} file dari:
{chr(10).join([f'• {target}: {cleanup_info["targets"][target]["files"]:,} file' for target in targets_with_data])}

❌ Operasi ini TIDAK DAPAT DIBATALKAN
💾 Pastikan Anda sudah backup data penting

Lanjutkan cleanup?""",
        on_confirm=lambda b: _execute_cleanup(ui_components, cleanup_info, logger),
        on_cancel=lambda b: _clear_confirmation_area(ui_components),
        item_name="Dataset",
        confirm_text="Ya, Hapus Semua",
        cancel_text="Batal"
    )
    
    _show_in_confirmation_area(ui_components, confirmation_dialog)

def _execute_cleanup(ui_components: Dict[str, Any], cleanup_info: Dict[str, Any], logger) -> None:
    """Execute cleanup operation dengan progress tracking"""
    
    def cleanup_thread():
        """Cleanup thread dengan progress tracking"""
        try:
            # Clear confirmation area
            _clear_confirmation_area(ui_components)
            
            # Show progress
            progress_tracker = ui_components.get('tracker')
            if progress_tracker:
                progress_tracker.show('cleanup')
                progress_tracker.update('overall', 0, "🧹 Memulai cleanup dataset...")
            
            # Create organizer untuk cleanup
            organizer = DatasetOrganizer(logger)
            
            # Setup progress callback
            if progress_tracker:
                organizer.set_progress_callback(_create_cleanup_progress_callback(progress_tracker, logger))
            
            # Execute cleanup
            result = organizer.cleanup_all_dataset_folders()
            
            # Handle result
            if result['status'] == 'success':
                files_removed = result['stats']['total_files_removed']
                folders_cleaned = len(result['stats']['folders_cleaned'])
                
                if progress_tracker:
                    progress_tracker.complete(f"✅ Cleanup selesai: {files_removed:,} file dihapus")
                
                success_msg = f"✅ Dataset berhasil dibersihkan: {files_removed:,} file dari {folders_cleaned} folder"
                show_status_safe(success_msg, "success", ui_components)
                logger.success(success_msg)
                
            elif result['status'] == 'empty':
                if progress_tracker:
                    progress_tracker.complete("ℹ️ Tidak ada file untuk dihapus")
                show_status_safe("ℹ️ Tidak ada file dataset untuk dibersihkan", "info", ui_components)
                
            else:
                error_msg = f"❌ Cleanup gagal: {result.get('message', 'Unknown error')}"
                if progress_tracker:
                    progress_tracker.error(error_msg)
                show_status_safe(error_msg, "error", ui_components)
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"❌ Error saat cleanup: {str(e)}"
            if progress_tracker:
                progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            logger.error(error_msg)
    
    # Start cleanup thread
    thread = threading.Thread(target=cleanup_thread, daemon=True)
    thread.start()

def _create_cleanup_progress_callback(progress_tracker, logger) -> Callable:
    """Create progress callback untuk cleanup operations"""
    
    def progress_callback(step: str, current: int, total: int, message: str):
        """Progress callback untuk cleanup dengan step mapping"""
        try:
            # Map cleanup steps
            if step == 'cleanup':
                progress_tracker.update('overall', current, message)
            elif 'Menghitung' in message:
                progress_tracker.update('overall', current, f"🔍 {message}")
            elif 'Menghapus' in message:
                progress_tracker.update('overall', current, f"🗑️ {message}")
            else:
                progress_tracker.update('overall', current, message)
                
        except Exception as e:
            logger.debug(f"🔍 Cleanup progress callback error: {str(e)}")
    
    return progress_callback

def _show_in_confirmation_area(ui_components: Dict[str, Any], dialog_widget) -> None:
    """Show dialog dalam confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        with confirmation_area:
            confirmation_area.clear_output(wait=True)
            from IPython.display import display
            display(dialog_widget)

def _clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Clear confirmation area dengan one-liner"""
    confirmation_area = ui_components.get('confirmation_area')
    confirmation_area and confirmation_area.clear_output(wait=True)