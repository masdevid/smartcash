"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_executor.py
Deskripsi: Fixed cleanup executor dengan proper button state management dan progress tracking
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
    
    # State management
    ui_components['cleanup_executor'] = None
    ui_components['cleanup_future'] = None
    
    def _on_cleanup_click(b):
        """Handler untuk tombol cleanup dengan proper state management."""
        # Check operation state - exclude validation karena bisa concurrent
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
        """Start cleanup setelah konfirmasi dengan proper state management."""
        # Set UI states
        ui_state.set_button_processing('cleanup_button', True, "Cleaning...")
        
        # Disable other action buttons during cleanup
        for button_key in ['preprocess_button', 'check_button']:
            if button_key in ui_components and ui_components[button_key]:
                ui_components[button_key].disabled = True
        
        # Setup progress tracking
        progress_bridge.setup_for_operation('cleanup')
        
        logger and logger.info("🧹 Memulai cleanup data preprocessing")
        update_status_panel(ui_components['status_panel'], "Memulai cleanup preprocessing...", "info")
        
        # Start cleanup dalam thread dengan proper executor management
        if ui_components['cleanup_executor']:
            ui_components['cleanup_executor'].shutdown(wait=False)
        
        ui_components['cleanup_executor'] = ThreadPoolExecutor(max_workers=1)
        ui_components['cleanup_future'] = ui_components['cleanup_executor'].submit(_run_cleanup, total_files)
        
        # Monitor completion
        _monitor_cleanup_completion()
    
    def _run_cleanup(total_files: int):
        """Execute cleanup dengan 3-level progress tracking."""
        try:
            preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
            
            # Phase 1: Scan dan persiapan (0-20%)
            progress_bridge.update_progress(
                overall_progress=5,
                step=1,
                current_progress=0,
                message="Memulai cleanup...",
                split_step="Scanning directories",
                status='info'
            )
            
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
            
            progress_bridge.update_progress(
                overall_progress=20,
                step=1,
                current_progress=100,
                message=f"Scan selesai: {total_actual_files} files ditemukan",
                split_step="Ready to clean",
                status='info'
            )
            
            # Phase 2: Cleanup per split (20-90%)
            current_overall = 20
            active_splits = [s for s in splits_to_clean if split_file_counts[s] > 0]
            
            if active_splits:
                split_progress_range = 70 / len(active_splits)
                
                for i, split in enumerate(active_splits):
                    split_path = preprocessed_dir / split
                    file_count = split_file_counts[split]
                    
                    # Progress range untuk split ini
                    split_start = current_overall
                    
                    progress_bridge.update_progress(
                        overall_progress=int(split_start),
                        step=2,
                        current_progress=0,
                        message=f"Membersihkan split {split}...",
                        split_step=f"Cleanup {split.title()}",
                        status='info'
                    )
                    
                    # Delete files dalam split
                    files_in_split = [f for f in split_path.rglob('*') if f.is_file()]
                    
                    for j, file_path in enumerate(files_in_split):
                        try:
                            file_path.unlink()
                            files_removed += 1
                            
                            # Update progress setiap 10 files atau di akhir
                            if j % 10 == 0 or j == len(files_in_split) - 1:
                                current_file_pct = int(((j + 1) / len(files_in_split)) * 100)
                                overall_pct = int(split_start + ((j + 1) / len(files_in_split)) * split_progress_range)
                                
                                progress_bridge.update_progress(
                                    overall_progress=overall_pct,
                                    step=2,
                                    current_progress=current_file_pct,
                                    message=f"Cleanup {split}: {j+1}/{len(files_in_split)}",
                                    split_step=f"{split.title()}: {current_file_pct}%",
                                    status='info'
                                )
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
                    progress_bridge.update_progress(
                        overall_progress=int(current_overall),
                        step=2,
                        current_progress=100,
                        message=f"Split {split} selesai dibersihkan",
                        split_step=f"{split.title()} Done",
                        status='success'
                    )
            
            # Phase 3: Finalisasi (90-100%)
            progress_bridge.update_progress(
                overall_progress=95,
                step=3,
                current_progress=0,
                message="Finalisasi cleanup...",
                split_step="Cleaning metadata",
                status='info'
            )
            
            # Clean metadata if exists
            metadata_dir = preprocessed_dir / 'metadata'
            if metadata_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(metadata_dir, ignore_errors=True)
                except Exception:
                    pass
            
            progress_bridge.update_progress(
                overall_progress=100,
                step=3,
                current_progress=100,
                message=f"Cleanup selesai: {files_removed} files dihapus",
                split_step="All clean",
                status='success'
            )
            
            return {
                'success': True,
                'files_removed': files_removed,
                'splits_cleaned': splits_cleaned,
                'message': f"Berhasil menghapus {files_removed:,} file dari {splits_cleaned} split"
            }
            
        except Exception as e:
            logger and logger.error(f"❌ Error cleanup: {str(e)}")
            raise
    
    def _monitor_cleanup_completion():
        """Monitor cleanup completion dengan proper error handling."""
        future = ui_components['cleanup_future']
        
        if not future:
            return
        
        if future.done():
            try:
                result = future.result()
                _on_cleanup_complete(result)
            except Exception as e:
                _on_cleanup_error(e)
            finally:
                # Cleanup executor
                if ui_components['cleanup_executor']:
                    ui_components['cleanup_executor'].shutdown(wait=False)
                    ui_components['cleanup_executor'] = None
                ui_components['cleanup_future'] = None
        else:
            # Colab-safe monitoring dengan simple approach
            import time
            time.sleep(1)
            _monitor_cleanup_completion()
    
    def _on_cleanup_complete(result):
        """Handler saat cleanup selesai dengan proper cleanup."""
        # Reset UI states
        ui_state.set_button_processing('cleanup_button', False, 
                                     success_text="Cleanup Dataset")
        
        # Re-enable other action buttons
        for button_key in ['preprocess_button', 'check_button']:
            if button_key in ui_components and ui_components[button_key]:
                ui_components[button_key].disabled = False
        
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
        """Handler saat cleanup error dengan proper cleanup."""
        # Reset UI states
        ui_state.set_button_processing('cleanup_button', False)
        
        # Re-enable other action buttons
        for button_key in ['preprocess_button', 'check_button']:
            if button_key in ui_components and ui_components[button_key]:
                ui_components[button_key].disabled = False
        
        error_msg = f"❌ Error cleanup: {str(error)}"
        logger and logger.error(error_msg)
        progress_bridge.error_operation("Error cleanup")
        update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['cleanup_button'].on_click(_on_cleanup_click)
    
    logger and logger.debug("✅ Fixed cleanup executor setup selesai")
    
    return ui_components