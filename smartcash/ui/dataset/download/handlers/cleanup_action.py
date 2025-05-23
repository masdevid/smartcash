"""
File: smartcash/ui/dataset/download/handlers/cleanup_action.py
Deskripsi: Fixed cleanup action dengan proper progress bar integration dan observer notifications
"""

import shutil
from pathlib import Path
from typing import Dict, Any
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_cleanup_confirmation
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons

def execute_cleanup_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi cleanup dataset dengan proper progress tracking."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🧹 Memulai cleanup dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        # Clear outputs sebelum mulai
        _clear_ui_outputs(ui_components)
        
        output_dir = ui_components.get('output_dir', {}).value or 'data'
        output_path = Path(output_dir)
        
        if not output_path.exists():
            if logger:
                logger.warning(f"⚠️ Direktori tidak ditemukan: {output_dir}")
            disable_download_buttons(ui_components, False)
            return
        
        # Hitung file untuk konfirmasi
        total_files = sum(1 for _ in output_path.rglob('*') if _.is_file())
        
        if total_files == 0:
            if logger:
                logger.info("ℹ️ Tidak ada file untuk dihapus")
            disable_download_buttons(ui_components, False)
            return
        
        # Tampilkan konfirmasi
        show_cleanup_confirmation(ui_components, output_dir, total_files)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error persiapan cleanup: {str(e)}")
        disable_download_buttons(ui_components, False)

def execute_cleanup_confirmed(ui_components: Dict[str, Any], output_dir: str) -> None:
    """Eksekusi cleanup setelah konfirmasi dengan proper progress tracking."""
    logger = ui_components.get('logger')
    output_path = Path(output_dir)
    
    try:
        # Initialize progress tracking
        _start_cleanup_progress(ui_components, "Memulai cleanup dataset")
        
        if logger:
            logger.info(f"🗑️ Menghapus dataset: {output_dir}")
        
        # Step 1: Count files untuk accurate progress (10%)
        _update_cleanup_progress(ui_components, 10, "Menghitung file...")
        total_files = sum(1 for _ in output_path.rglob('*') if _.is_file())
        
        if total_files == 0:
            _complete_cleanup_progress(ui_components, "Tidak ada file untuk dihapus")
            if logger:
                logger.info("ℹ️ Direktori sudah kosong")
            return
        
        # Step 2: Delete files dengan progress tracking (10% - 90%)
        deleted_files = _delete_files_with_progress(ui_components, output_path, total_files)
        
        # Step 3: Remove directory structure (90% - 95%)
        _update_cleanup_progress(ui_components, 90, "Menghapus direktori...")
        if output_path.exists():
            shutil.rmtree(output_path, ignore_errors=True)
        
        # Step 4: Verify deletion (95% - 100%)
        _update_cleanup_progress(ui_components, 95, "Memverifikasi penghapusan...")
        
        if output_path.exists():
            # Force removal jika masih ada
            try:
                shutil.rmtree(output_path, ignore_errors=True)
                if output_path.exists():
                    raise Exception("Direktori masih ada setelah penghapusan")
            except Exception as e:
                _error_cleanup_progress(ui_components, f"Gagal menghapus direktori: {str(e)}")
                if logger:
                    logger.error(f"❌ Error cleanup: {str(e)}")
                return
        
        # Complete
        _complete_cleanup_progress(ui_components, f"Cleanup berhasil: {deleted_files} file dihapus")
        
        if logger:
            logger.success(f"✅ Dataset berhasil dihapus: {deleted_files} file")
        
    except Exception as e:
        _error_cleanup_progress(ui_components, f"Cleanup gagal: {str(e)}")
        if logger:
            logger.error(f"❌ Error cleanup: {str(e)}")
    finally:
        disable_download_buttons(ui_components, False)

def _delete_files_with_progress(ui_components: Dict[str, Any], dataset_path: Path, total_files: int) -> int:
    """Delete files dengan detailed progress tracking."""
    deleted_count = 0
    processed_count = 0
    
    try:
        # Walk through directory dan delete files
        for root, dirs, files in os.walk(str(dataset_path), topdown=False):
            # Delete files first
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception:
                    pass  # Continue dengan file lainnya
                
                processed_count += 1
                
                # Update progress setiap 5% atau setiap 10 files
                if processed_count % max(1, total_files // 20) == 0 or processed_count % 10 == 0:
                    # Progress dari 10% sampai 90%
                    progress = 10 + int((processed_count / total_files) * 80)
                    _update_cleanup_progress(ui_components, progress, 
                                           f"Menghapus file: {processed_count}/{total_files}")
            
            # Delete empty directories
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except Exception:
                    pass
    
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Error saat delete files: {str(e)}")
    
    return deleted_count

def _start_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start cleanup progress tracking."""
    # Show progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'
    
    # Reset progress widgets
    _update_progress_widgets(ui_components, 0, message)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_START', ui_components, 
               progress=0, message=message, namespace="cleanup")
    except Exception:
        pass

def _update_cleanup_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update cleanup progress."""
    # Clamp progress
    progress = max(0, min(100, progress))
    
    # Update UI widgets directly
    _update_progress_widgets(ui_components, progress, message)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_PROGRESS', ui_components,
               progress=progress, message=message, namespace="cleanup")
    except Exception:
        pass

def _complete_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete cleanup progress."""
    # Update ke 100%
    _update_progress_widgets(ui_components, 100, message)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_COMPLETE', ui_components,
               progress=100, message=message, namespace="cleanup")
    except Exception:
        pass

def _error_cleanup_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set cleanup progress ke error state."""
    # Reset progress ke 0 untuk indicate error
    _update_progress_widgets(ui_components, 0, f"❌ {message}", error=True)
    
    # Send observer notification
    try:
        from smartcash.components.observer import notify
        notify('DOWNLOAD_ERROR', ui_components,
               progress=0, message=message, namespace="cleanup")
    except Exception:
        pass

def _update_progress_widgets(ui_components: Dict[str, Any], progress: int, message: str, error: bool = False) -> None:
    """Update progress widgets directly untuk immediate feedback."""
    # Update main progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = progress
        description = "Error" if error else f"Progress: {progress}%"
        ui_components['progress_bar'].description = description
        
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
    
    # Update current progress (step progress) 
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = progress
        ui_components['current_progress'].description = f"Cleanup: {progress}%"
        
        if hasattr(ui_components['current_progress'], 'layout'):
            ui_components['current_progress'].layout.visibility = 'visible'
    
    # Update labels
    if 'overall_label' in ui_components:
        ui_components['overall_label'].value = message
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'visible'
    
    if 'step_label' in ui_components:
        ui_components['step_label'].value = f"Cleanup: {message}"
        if hasattr(ui_components['step_label'], 'layout'):
            ui_components['step_label'].layout.visibility = 'visible'

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs sebelum cleanup."""
    # Clear log output
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    # Clear confirmation area
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()

# Import os yang diperlukan
import os