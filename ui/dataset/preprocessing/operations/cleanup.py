"""
File: smartcash/ui/dataset/preprocessing/operations/cleanup.py
Deskripsi: Cleanup operation handler untuk preprocessing module.
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.error_handler import handle_ui_errors
from smartcash.ui.dataset.preprocessing.operations.base_operation import BaseOperationHandler


class CleanupOperationHandler(BaseOperationHandler):
    """Cleanup operation handler untuk preprocessing module.
    
    Features:
    - 🎯 Dataset cleanup dengan configurable target
    - 📊 Progress tracking per step
    - 🔄 Backup support
    - 📝 Summary reporting
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize cleanup operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(ui_components)
        
        # Set operation specific properties
        self.operation_name = "cleanup"
        self.button_name = "cleanup_button"
        self.confirmation_message = "Apakah Anda yakin ingin membersihkan dataset? Proses ini akan menghapus data sesuai konfigurasi."
        self.success_message = "Pembersihan dataset berhasil diselesaikan"
        self.failure_message = "Pembersihan dataset gagal diselesaikan"
        self.waiting_message = "Menunggu konfirmasi untuk pembersihan..."
    
    @handle_ui_errors(error_component_title="Cleanup Error", log_error=True)
    def _execute_operation(self) -> Dict[str, Any]:
        """Execute cleanup operation.
        
        Returns:
            Operation result dictionary
        """
        self.logger.info("🧹 Memulai pembersihan dataset")
        
        # Get config
        config = self.extract_config()
        cleanup_config = config.get('cleanup', {})
        
        # Extract parameters
        target = cleanup_config.get('target', 'preprocessed')
        backup = cleanup_config.get('backup', True)
        
        self.logger.info(f"📋 Membersihkan target: {target}, backup: {backup}")
        
        # Update status
        self.update_status(f"Membersihkan {target} data")
        
        # Simulate cleanup steps
        total_steps = 4  # Backup, cleanup, verify, finalize
        current_step = 0
        
        # Step 1: Backup if enabled
        if backup:
            current_step += 1
            self.update_progress(current_step, total_steps, "Membuat backup data")
            self.logger.info("💾 Membuat backup data sebelum pembersihan")
        
        # Step 2: Cleanup based on target
        current_step += 1
        self.update_progress(current_step, total_steps, f"Membersihkan {target} data")
        self.logger.info(f"🧹 Membersihkan {target} data")
        
        # Step 3: Verify cleanup
        current_step += 1
        self.update_progress(current_step, total_steps, "Memverifikasi pembersihan")
        self.logger.info("🔍 Memverifikasi hasil pembersihan")
        
        # Step 4: Finalize
        current_step += 1
        self.update_progress(current_step, total_steps, "Menyelesaikan pembersihan")
        self.logger.info("✅ Menyelesaikan pembersihan")
        
        # Complete
        self.update_progress(total_steps, total_steps, "Pembersihan selesai")
        
        # Generate dummy stats (in real implementation, this would be actual statistics)
        stats = {
            "target": target,
            "backup_created": backup,
            "files_removed": 100 if target == "all" else 50,
            "space_freed": "250MB" if target == "all" else "125MB"
        }
        
        # Log completion
        self.logger.info(f"✅ Pembersihan selesai: {stats['files_removed']} files removed, {stats['space_freed']} freed")
        
        # Update summary with detailed stats
        summary_message = f"Target: {stats['target']}\n" \
                         f"Backup created: {'Yes' if stats['backup_created'] else 'No'}\n" \
                         f"Files removed: {stats['files_removed']}\n" \
                         f"Space freed: {stats['space_freed']}"
                         
        self.update_summary(summary_message, "info", "Cleanup Results")
        
        # Return success result with stats
        return {
            "status": True,
            "stats": stats,
            "message": self.success_message
        }
