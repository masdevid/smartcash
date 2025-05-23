"""
File: smartcash/ui/dataset/preprocessing/handlers/cleanup_handler.py
Deskripsi: Handler untuk operasi cleanup preprocessing dengan safety checks dan confirmation
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from pathlib import Path

from smartcash.ui.components.confirmation_dialog import create_destructive_confirmation
from smartcash.ui.components.status_panel import update_status_panel


def setup_cleanup_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk operasi cleanup preprocessing."""
    logger = ui_components.get('logger')
    
    ui_components['cleanup_running'] = False
    
    def _check_preprocessed_data() -> tuple[bool, list, str]:
        """Check data preprocessing yang tersedia."""
        preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
        
        if not preprocessed_dir.exists():
            return False, [], "Tidak ada data preprocessing yang ditemukan"
        
        # Check splits
        available_splits = []
        total_files = 0
        
        for split in ['train', 'val', 'test']:
            split_path = preprocessed_dir / split
            if split_path.exists():
                files = list(split_path.glob('**/*.jpg'))
                if files:
                    available_splits.append(f"{split} ({len(files)} files)")
                    total_files += len(files)
        
        if not available_splits:
            return False, [], "Tidak ada data preprocessing yang valid"
        
        return True, available_splits, f"Total {total_files} files"
    
    def _check_symlink_safety() -> tuple[bool, list]:
        """Check apakah ada symlink yang berbahaya untuk dihapus."""
        preprocessed_dir = Path(ui_components.get('preprocessed_dir', 'data/preprocessed'))
        dangerous_symlinks = []
        
        if preprocessed_dir.is_symlink():
            dangerous_symlinks.append(str(preprocessed_dir))
        
        # Check subdirectories
        if preprocessed_dir.exists():
            for split in ['train', 'val', 'test']:
                split_path = preprocessed_dir / split
                if split_path.is_symlink():
                    dangerous_symlinks.append(str(split_path))
        
        return len(dangerous_symlinks) == 0, dangerous_symlinks
    
    def _on_cleanup_click(b):
        """Handler untuk tombol cleanup."""
        if ui_components['cleanup_running']:
            logger.warning("⚠️ Cleanup sedang berjalan")
            return
        
        if ui_components.get('processing_running', False):
            logger.warning("⚠️ Tidak dapat cleanup saat preprocessing berjalan")
            return
        
        # Check data exists
        has_data, splits_info, summary = _check_preprocessed_data()
        if not has_data:
            logger.info(f"ℹ️ {summary}")
            update_status_panel(ui_components['status_panel'], summary, "info")
            return
        
        # Check symlink safety
        is_safe, dangerous_links = _check_symlink_safety()
        if not is_safe:
            logger.error(f"❌ Tidak dapat cleanup: symlink berbahaya ditemukan: {', '.join(dangerous_links)}")
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
            on_confirm=lambda b: _start_cleanup_confirmed(),
            on_cancel=lambda b: _hide_dialog(),
            item_name="Data Preprocessing"
        )
        
        ui_components['cleanup_dialog'] = dialog
        _show_dialog(dialog)
    
    def _start_cleanup_confirmed():
        """Mulai cleanup setelah konfirmasi."""
        if 'cleanup_dialog' in ui_components:
            _hide_dialog()
        
        ui_components['cleanup_running'] = True
        
        # Disable button dan update status
        ui_components['cleanup_button'].disabled = True
        ui_components['cleanup_button'].description = "Cleaning..."
        
        logger.info("🧹 Memulai cleanup data preprocessing")
        update_status_panel(ui_components['status_panel'], "Memulai cleanup data preprocessing...", "info")
        
        # Start cleanup dalam thread
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_cleanup)
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
                # Check again in 1 second
                import threading
                threading.Timer(1.0, check_completion).start()
        
        check_completion()
    
    def _run_cleanup():
        """Jalankan cleanup dalam thread terpisah."""
        try:
            # Get preprocessing manager
            if not ui_components.get('preprocessing_manager'):
                from smartcash.dataset.services.preprocessing_manager import PreprocessingManager
                config = {
                    'preprocessing': {
                        'preprocessed_dir': ui_components.get('preprocessed_dir', 'data/preprocessed')
                    },
                    'data': {'dir': ui_components.get('data_dir', 'data')}
                }
                ui_components['preprocessing_manager'] = PreprocessingManager(config, logger)
            
            preprocessing_manager = ui_components['preprocessing_manager']
            
            # Clean all preprocessed data
            preprocessing_manager.clean_preprocessed(split='all')
            
            return {'success': True, 'message': 'Cleanup berhasil'}
            
        except Exception as e:
            logger.error(f"❌ Error cleanup: {str(e)}")
            raise
    
    def _on_cleanup_complete(result):
        """Handler saat cleanup selesai."""
        ui_components['cleanup_running'] = False
        ui_components['cleanup_button'].disabled = False
        ui_components['cleanup_button'].description = "Cleanup Dataset"
        
        if result and result.get('success', False):
            success_msg = "✅ Cleanup data preprocessing berhasil"
            logger.success(success_msg)
            update_status_panel(ui_components['status_panel'], success_msg, "success")
        else:
            error_msg = "❌ Cleanup gagal"
            logger.error(error_msg)
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    def _on_cleanup_error(error):
        """Handler saat cleanup error."""
        ui_components['cleanup_running'] = False
        ui_components['cleanup_button'].disabled = False
        ui_components['cleanup_button'].description = "Cleanup Dataset"
        
        error_msg = f"❌ Error cleanup: {str(error)}"
        logger.error(error_msg)
        update_status_panel(ui_components['status_panel'], error_msg, "error")
    
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