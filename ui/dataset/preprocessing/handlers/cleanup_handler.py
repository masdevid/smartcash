"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_handler.py
Deskripsi: Fixed cleanup handler dengan progress tracking yang konsisten
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from pathlib import Path

from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.components.progress_tracking import (
    update_overall_progress, update_step_progress, update_current_progress
)
from smartcash.dataset.utils.path_validator import get_path_validator


def setup_cleanup_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk cleanup dengan fixed progress tracking."""
    logger = ui_components.get('logger')
    path_validator = get_path_validator(logger)
    
    ui_components['cleanup_running'] = False
    
    def _check_preprocessed_data() -> tuple[bool, list, str, int]:
        """Check data preprocessing dengan detail count."""
        preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
        validation_result = path_validator.validate_preprocessed_structure(preprocessed_dir)
        
        if not validation_result['valid']:
            return False, [], "Tidak ada data preprocessing", 0
        
        available_splits = []
        total_files = validation_result['total_processed']
        
        for split, info in validation_result['splits'].items():
            if info['processed'] > 0:
                available_splits.append(f"{split} ({info['processed']} files)")
        
        if not available_splits:
            return False, [], "Tidak ada data valid untuk cleanup", 0
        
        return True, available_splits, f"Total {total_files} files", total_files
    
    def _on_cleanup_click(b):
        """Handler untuk tombol cleanup."""
        if ui_components['cleanup_running']:
            logger.warning("⚠️ Cleanup sedang berjalan")
            return
        
        if ui_components.get('processing_running', False):
            logger.warning("⚠️ Tidak dapat cleanup saat preprocessing berjalan")
            return
        
        # Check data exists
        has_data, splits_info, summary, total_files = _check_preprocessed_data()
        if not has_data:
            logger.info(f"ℹ️ {summary}")
            update_status_panel(ui_components['status_panel'], summary, "info")
            return
        
        # Show confirmation dialog
        splits_text = '\n'.join([f"• {split}" for split in splits_info])
        confirmation_msg = f"""Akan menghapus data preprocessing berikut:

{splits_text}

{summary}

⚠️ Aksi ini tidak dapat dibatalkan!"""
        
        dialog = create_destructive_confirmation(
            title="Konfirmasi Cleanup Preprocessing",
            message=confirmation_msg,
            on_confirm=lambda b: _start_cleanup_confirmed(total_files),
            on_cancel=lambda b: _hide_dialog(),
            item_name="Data Preprocessing"
        )
        
        ui_components['cleanup_dialog'] = dialog
        _show_dialog(dialog)
    
    def _start_cleanup_confirmed(total_files: int):
        """Mulai cleanup setelah konfirmasi."""
        if 'cleanup_dialog' in ui_components:
            _hide_dialog()
        
        ui_components['cleanup_running'] = True
        
        # Update UI state
        ui_components['cleanup_button'].disabled = True
        ui_components['cleanup_button'].description = "Cleaning..."
        
        logger.info("🧹 Memulai cleanup data preprocessing")
        update_status_panel(ui_components['status_panel'], "Memulai cleanup preprocessing...", "info")
        
        # Show progress container
        if 'progress_components' in ui_components:
            ui_components['progress_components']['show_container']()
            ui_components['progress_components']['reset_all']()
        
        # Start cleanup dalam thread
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_cleanup, total_files)
        ui_components['cleanup_future'] = future
        ui_components['cleanup_executor'] = executor
        
        # Monitor completion
        def check_completion():
            if future.done():
                try:
                    result = future.result()
                    _on_cleanup_complete(result)
                except Exception as e:
                    _on_cleanup_error(e)
                finally:
                    executor.shutdown(wait=False)
            else:
                import threading
                threading.Timer(1.0, check_completion).start()
        
        check_completion()
    
    def _run_cleanup(total_files: int):
        """Jalankan cleanup dengan progress tracking."""
        try:
            preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
            
            # Update progress callback
            def progress_callback(step: str, current: int, total: int, message: str):
                try:
                    # Map step to numeric for overall progress
                    step_mapping = {'scan': 1, 'cleanup': 2, 'verify': 3}
                    step_num = step_mapping.get(step, 1)
                    
                    # Overall progress (3 steps total)
                    overall_progress = ((step_num - 1) * 100 + (current / max(total, 1)) * 100) / 3
                    update_overall_progress(
                        ui_components['progress_components'],
                        int(overall_progress), 100, f"Cleanup: {message}"
                    )
                    
                    # Step progress
                    update_step_progress(
                        ui_components['progress_components'],
                        step_num, 3, f"Step: {step.title()}"
                    )
                    
                    # Current progress
                    update_current_progress(
                        ui_components['progress_components'],
                        current, total, message
                    )
                    
                    # Log progress
                    if logger:
                        logger.info(f"🧹 {message} ({current}/{total})")
                        
                except Exception:
                    pass
            
            # Step 1: Scan files
            progress_callback('scan', 0, 100, "Scanning files")
            
            splits_to_clean = ['train', 'valid', 'test']
            files_removed = 0
            
            for i, split in enumerate(splits_to_clean):
                split_path = preprocessed_dir / split
                if not split_path.exists():
                    continue
                
                # Count files in split
                files_in_split = list(split_path.rglob('*'))
                files_in_split = [f for f in files_in_split if f.is_file()]
                
                if not files_in_split:
                    continue
                
                # Step 2: Cleanup files
                step_start = 33 + (i * 33)  # Distribute across splits
                
                for j, file_path in enumerate(files_in_split):
                    try:
                        file_path.unlink()
                        files_removed += 1
                        
                        # Update progress every 10 files or at end
                        if j % 10 == 0 or j == len(files_in_split) - 1:
                            progress = step_start + ((j + 1) / len(files_in_split)) * 33
                            progress_callback('cleanup', int(progress), 100, 
                                            f"Removing {split} files")
                    except Exception:
                        pass
                
                # Remove empty directories
                try:
                    if split_path.exists():
                        import shutil
                        shutil.rmtree(split_path, ignore_errors=True)
                except Exception:
                    pass
            
            # Step 3: Verify cleanup
            progress_callback('verify', 100, 100, "Verifying cleanup")
            
            return {
                'success': True,
                'files_removed': files_removed,
                'message': f"Berhasil menghapus {files_removed} file"
            }
            
        except Exception as e:
            logger.error(f"❌ Error cleanup: {str(e)}")
            raise
    
    def _on_cleanup_complete(result):
        """Handler saat cleanup selesai."""
        ui_components['cleanup_running'] = False
        ui_components['cleanup_button'].disabled = False
        ui_components['cleanup_button'].description = "Cleanup Dataset"
        
        if result and result.get('success', False):
            files_removed = result.get('files_removed', 0)
            success_msg = f"✅ Cleanup selesai: {files_removed} file dihapus"
            logger.success(success_msg)
            
            # Update progress to 100%
            if 'progress_components' in ui_components:
                update_overall_progress(
                    ui_components['progress_components'],
                    100, 100, "Cleanup selesai"
                )
            
            update_status_panel(ui_components['status_panel'], success_msg, "success")
        else:
            error_msg = "❌ Cleanup gagal"
            logger.error(error_msg)
            update_status_panel(ui_components['status_panel'], error_msg, "error")
        
        # Hide progress setelah delay
        import threading
        def hide_progress():
            if 'progress_components' in ui_components:
                ui_components['progress_components']['hide_container']()
        threading.Timer(3.0, hide_progress).start()
    
    def _on_cleanup_error(error):
        """Handler saat cleanup error."""
        ui_components['cleanup_running'] = False
        ui_components['cleanup_button'].disabled = False
        ui_components['cleanup_button'].description = "Cleanup Dataset"
        
        error_msg = f"❌ Error cleanup: {str(error)}"
        logger.error(error_msg)
        update_status_panel(ui_components['status_panel'], error_msg, "error")
        
        # Reset progress
        if 'progress_components' in ui_components:
            ui_components['progress_components']['reset_all']()
            ui_components['progress_components']['hide_container']()
    
    def _show_dialog(dialog):
        """Show confirmation dialog."""
        from IPython.display import display
        display(dialog)
    
    def _hide_dialog():
        """Hide confirmation dialog."""
        if 'cleanup_dialog' in ui_components:
            try:
                ui_components['cleanup_dialog'].close()
            except:
                pass
            del ui_components['cleanup_dialog']
    
    # Setup event handlers
    ui_components['cleanup_button'].on_click(_on_cleanup_click)
    
    if logger:
        logger.debug("✅ Cleanup handler preprocessing setup selesai")
    
    return ui_components