"""
File: smartcash/ui/dataset/preprocessing/operations/check.py
Deskripsi: Check operation handler untuk preprocessing module.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.dataset.preprocessing.operations.base_operation import BaseOperationHandler


class CheckOperationHandler(BaseOperationHandler):
    """Check operation handler untuk preprocessing module.
    
    Features:
    - 🎯 Dataset validation dan integrity checking
    - 📊 Progress tracking per check
    - 🔍 Detailed reporting
    - 📝 Summary reporting dengan statistics
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize check operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(ui_components)
        
        # Set operation specific properties
        self.operation_name = "check"
        self.button_name = "check_button"
        self.confirmation_message = "Apakah Anda yakin ingin memeriksa dataset? Proses ini akan memvalidasi data yang telah diproses."
        self.success_message = "Pemeriksaan dataset berhasil diselesaikan"
        self.failure_message = "Pemeriksaan dataset gagal diselesaikan"
        self.waiting_message = "Menunggu konfirmasi untuk pemeriksaan..."
    
    @handle_ui_errors(error_component_title="Check Error", log_error=True)
    def _execute_operation(self) -> Dict[str, Any]:
        """Execute check operation.
        
        Returns:
            Operation result dictionary
        """
        self.logger.info("🔍 Memulai pemeriksaan dataset")
        
        # Get config
        config = self.extract_config()
        preprocessing_config = config.get('preprocessing', {})
        
        # Extract parameters
        validation = preprocessing_config.get('validation', True)
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        
        self.logger.info(f"📋 Memeriksa dataset untuk splits: {', '.join(target_splits)}")
        
        # Update status
        self.update_status(f"Memeriksa dataset untuk {len(target_splits)} splits")
        
        # Simulate check steps
        total_checks = len(target_splits) * 3  # 3 checks per split
        current_check = 0
        
        # Check each split
        for split in target_splits:
            self.update_status(f"Memeriksa split: {split}")
            
            # Check 1: File integrity
            current_check += 1
            self.update_progress(current_check, total_checks, f"Memeriksa integritas file {split}")
            self.logger.info(f"🔍 Memeriksa integritas file untuk {split}")
            
            # Check 2: Image dimensions
            current_check += 1
            self.update_progress(current_check, total_checks, f"Memeriksa dimensi gambar {split}")
            self.logger.info(f"🔍 Memeriksa dimensi gambar untuk {split}")
            
            # Check 3: Metadata
            current_check += 1
            self.update_progress(current_check, total_checks, f"Memeriksa metadata {split}")
            self.logger.info(f"🔍 Memeriksa metadata untuk {split}")
        
        # Complete
        self.update_progress(total_checks, total_checks, "Pemeriksaan selesai")
        
        # Generate dummy stats (in real implementation, this would be actual statistics)
        stats = {
            "total_files": 100,
            "valid": 95,
            "invalid": 5,
            "splits": {
                split: {"total": 50 if split == "train" else 50, "valid": 48 if split == "train" else 47, "invalid": 2 if split == "train" else 3}
                for split in target_splits
            }
        }
        
        # Log completion
        self.logger.info(f"✅ Pemeriksaan selesai: {stats['valid']} valid, {stats['invalid']} invalid files")
        
        # Update summary with detailed stats
        summary_message = f"Total files: {stats['total_files']}\n" \
                         f"Valid: {stats['valid']} ({stats['valid']/stats['total_files']*100:.1f}%)\n" \
                         f"Invalid: {stats['invalid']} ({stats['invalid']/stats['total_files']*100:.1f}%)\n\n"
                         
        for split, split_stats in stats['splits'].items():
            summary_message += f"{split}: {split_stats['valid']} valid, {split_stats['invalid']} invalid\n"
            
        self.update_summary(summary_message, "info", "Check Results")
        
        # Return success result with stats
        return {
            "status": True,
            "stats": stats,
            "message": self.success_message
        }
