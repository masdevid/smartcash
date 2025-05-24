"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_executor.py
Deskripsi: Handler khusus untuk execute cleanup dengan 3-level progress tracking
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from pathlib import Path
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.dataset.preprocessing.utils import (
    get_validation_helper, get_dialog_manager, 
    get_ui_state_manager, get_progress_bridge
)

def setup_cleanup_executor(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk execute cleanup dengan comprehensive management."""
    logger = ui_components.get('logger')
    
    # Get utilities
    validation_helper = get_validation_helper(ui_components, logger)
    dialog_manager = get_dialog_manager(ui_components)
    ui_state = get_ui_state_manager(ui_components)
    progress_bridge = get_progress_bridge(ui_components, logger)
    
    def _on_cleanup_click(b):
        """Handler untuk tombol cleanup dengan comprehensive validation."""
        # Check operation state
        can_start, message = ui_state.can_start_operation('cleanup', exclude_operations=['validation'])
        if not can_start:
            logger and logger.warning(f"⚠️ {message}")
            update_status_panel(ui_components['status_panel'], message, "warning")
            return
        
        # Check data exists
        has_data, splits_info, summary, total_files = validation_helper.check_preprocessed_data()
        if not has_data:
            logger and logger.info(f"ℹ️ {summary}")
            update_status_panel(ui_components['status_panel'], summary, "info")
            return
        
        # Show destructive confirmation dialog
        splits_text = '\n'.join([f"• {split}" for split in splits_info])
        confirmation_msg = f"""Akan menghapus data preprocessing berikut:

{splits_text}

{summary}

⚠️ Aksi ini tidak dapat dibatalkan!
Data yang dihapus tidak dapat dikembalikan."""
        
        dialog_manager.show_destructive_confirmation(
            title="Konfirmasi Cleanup Preprocessing",
            message=confirmation_msg,
            on_confirm=lambda: _start_cleanup_confirmed(total_files),
            item_name="Data Preprocessing"
        )
    
    def _start_cleanup_confirmed(total_files: int):
        """Start cleanup setelah konfirmasi."""
        # Setup progress
        progress_bridge.setup_for_operation('cleanup')
        
        # Set UI state
        ui_state.set_button_processing('cleanup_button', True, "Cleaning...")
        
        logger and logger.info("🧹 Memulai cleanup data preprocessing")
        update_status_panel(ui_components['status_panel'], "Memulai cleanup preprocessing...", "info")
        
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
        """Execute cleanup dengan 3-level progress tracking."""
        try:
            preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
            
            def progress_update(overall_pct: int, step_pct: int, current_pct: int, 
                              overall_msg: str, step_msg: str, current_msg: str):
                """Helper untuk update 3-level progress."""
                try:
                    progress_bridge.update_progress(
                        overall_progress=overall_pct,
                        step=int(step_pct/33.33) + 1 if step_pct > 0 else 1,
                        current_progress=current_pct,
                        current_total=100,
                        message=overall_msg,
                        split_step=step_msg,
                        status='info'
                    )
                except Exception:
                    pass
            
            # Phase 1: Scan dan persiapan (0-20%)
            progress_update(5, 25, 0, "Memulai cleanup...", "📋 Persiapan", "Scanning directories")
            
            splits_to_clean = ['train', 'valid', 'test']
            files_removed = 0
            splits_cleaned = 0
            
            # Count total files untuk progress calculation
            total_actual_files = 0
            split_file_counts = {}
            
            for split in splits_to_clean:
                split_path = preprocessed_dir / split
                if split_path.exists():
                    files_in_split = [f for f in split_path.rglob('*') if f.is_file()]
                    split_file_counts[split] = len(files_in_split)
                    total_actual_files += len(files_in_split)
                else:
                    split_file_counts[split] = 0
            
            progress_update(20, 100, 100, f"Scan selesai: {total_actual_files} files ditemukan", 
                          "✅ Persiapan Selesai", "Ready to clean")
            
            # Phase 2: Cleanup per split (20-90%)
            current_overall = 20
            
            for i, split in enumerate(splits_to_clean):
                split_path = preprocessed_dir / split
                file_count = split_file_counts[split]
                
                if file_count == 0:
                    continue
                
                # Progress range untuk split ini
                split_progress_range = 70 / len([s for s in splits_to_clean if split_file_counts[s] > 0])
                split_start = current_overall
                
                progress_update(int(split_start), 0, 0, 
                              f"Membersihkan split {split}...", f"🗑️ Cleanup {split.title()}", 
                              f"Starting {split}")
                
                # Delete files dalam split
                files_in_split = [f for f in split_path.rglob('*') if f.is_file()]
                
                for j, file_path in enumerate(files_in_split):
                    try:
                        file_path.unlink()
                        files_removed += 1
                        
                        # Update progress setiap 10 files atau di akhir
                        if j % 10 == 0 or j == len(files_in_split) - 1:
                            current_file_pct = int(((j + 1) / len(files_in_split)) * 100)
                            current_split_pct = int(((j + 1) / len(files_in_split)) * 100)
                            overall_pct = int(split_start + ((j + 1) / len(files_in_split)) * split_progress_range)
                            
                            progress_update(overall_pct, current_split_pct, current_file_pct,
                                          f"Cleanup {split}: {j+1}/{len(files_in_split)}", 
                                          f"🗑️ {split.title()}: {current_split_pct}%",
                                          f"File {j+1}/{len(files_in_split)}")
                    except Exception:
                        pass
                
                # Remove empty directories
                try:
                    if split_path.exists():
                        import shutil
                        shutil.rmtree(split_path, ignore_errors=True)
                        splits_cleaned += 1
                except Exception:
                    pass
                
                current_overall += split_progress_range
                progress_update(int(current_overall), 100, 100, 
                              f"Split {split} selesai dibersihkan", f"✅ {split.title()} Done", "Complete")
            
            # Phase 3: Finalisasi (90-100%)
            progress_update(95, 0, 0, "Finalisasi cleanup...", "🔄 Finalisasi", "Cleaning metadata")
            
            # Clean metadata if exists
            metadata_dir = preprocessed_dir / 'metadata'
            if metadata_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(metadata_dir, ignore_errors=True)
                except Exception:
                    pass
            
            progress_update(100, 100, 100, f"Cleanup selesai: {files_removed} files dihapus", 
                          "✅ Finalisasi Selesai", "All clean")
            
            return {
                'success': True,
                'files_removed': files_removed,
                'splits_cleaned': splits_cleaned,
                'message': f"Berhasil menghapus {files_removed:,} file dari {splits_cleaned} split"
            }
            
        except Exception as e:
            logger and logger.error(f"❌ Error cleanup: {str(e)}")
            raise
    
    def _on_cleanup_complete(result):
        """Handler saat cleanup selesai."""
        # Reset UI state
        ui_state.set_button_processing('cleanup_button', False, 
                                     success_text="Cleanup Dataset")
        
        if result and result.get('success', False):
            files_removed = result.get('files_removed', 0)
            splits_cleaned = result.get('splits_cleaned', 0)
            success_msg = f"✅ Cleanup selesai: {files_removed:,} file dari {splits_cleaned} split dihapus"
            
            logger and logger.success(success_msg)
            progress_bridge.complete_operation("Cleanup selesai")
            update_status_panel(ui_components['status_panel'], success_msg, "success")
        else:
            error_msg = result.get('message', 'Cleanup gagal') if result else "Cleanup gagal"
            logger and logger.error(f"❌ {error_msg}")
            progress_bridge.error_operation("Cleanup gagal")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    def _on_cleanup_error(error):
        """Handler saat cleanup error."""
        # Reset UI state
        ui_state.set_button_processing('cleanup_button', False)
        
        error_msg = f"❌ Error cleanup: {str(error)}"
        logger and logger.error(error_msg)
        progress_bridge.error_operation("Error cleanup")
        update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['cleanup_button'].on_click(_on_cleanup_click)
    
    logger and logger.debug("✅ Cleanup executor setup selesai")
    
    return ui_components