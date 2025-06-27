"""
File: smartcash/ui/dataset/preprocessing/handlers/confirmation_handlers.py
Deskripsi: Confirmation handlers menggunakan BasePreprocessingHandler untuk DRY implementation
"""

from typing import Dict, Any
from .base_handler import BasePreprocessingHandler

class ConfirmationHandler(BasePreprocessingHandler):
    """Handler untuk dialog confirmations menggunakan base class"""
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup confirmation management functions"""
        handlers = {
            'show_preprocessing_confirmation': self.show_preprocessing_confirmation,
            'show_cleanup_confirmation': self.show_cleanup_confirmation,
            'set_preprocessing_confirmed': self.set_preprocessing_confirmed,
            'set_cleanup_confirmed': self.set_cleanup_confirmed,
            'handle_preprocessing_cancel': self.handle_preprocessing_cancel,
            'handle_cleanup_cancel': self.handle_cleanup_cancel,
            'is_confirmation_pending': self.is_confirmation_pending,
            'clear_confirmations': self.clear_all_confirmations
        }
        
        self.log_debug("✅ Confirmation handlers setup completed")
        return handlers
    
    def show_preprocessing_confirmation(self) -> None:
        """Show preprocessing confirmation dialog"""
        self.show_confirmation_dialog(
            title="Konfirmasi Preprocessing",
            message="Apakah Anda yakin ingin memproses dataset dengan YOLO normalization?",
            on_confirm=self.set_preprocessing_confirmed,
            on_cancel=self.handle_preprocessing_cancel,
            confirm_text="Ya, Proses",
            cancel_text="Batal"
        )
        self.log_info("⏳ Menunggu konfirmasi preprocessing...")
    
    def show_cleanup_confirmation(self) -> None:
        """Show cleanup confirmation dialog dengan target info"""
        config = self.extract_config()
        cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
        
        target_descriptions = {
            'preprocessed': 'file preprocessing (pre_*.npy + pre_*.txt)',
            'samples': 'sample images (sample_*.jpg)',
            'both': 'file preprocessing dan sample images'
        }
        target_desc = target_descriptions.get(cleanup_target, cleanup_target)
        
        self.show_confirmation_dialog(
            title="🧹 Konfirmasi Cleanup",
            message=f"Hapus {target_desc}?\n\nTindakan ini akan menghapus file-file yang sudah diproses.",
            on_confirm=self.set_cleanup_confirmed,
            on_cancel=self.handle_cleanup_cancel,
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
        self.log_info(f"⏳ Menunggu konfirmasi cleanup untuk: {target_desc}...")
    
    def set_preprocessing_confirmed(self) -> None:
        """Set preprocessing confirmation dan trigger execution"""
        self.log_info("✅ Konfirmasi diterima, memulai preprocessing...")
        self.ui_components['_preprocessing_confirmed'] = True
        
        # Import dan execute dari operation handler
        from .operation_handlers import OperationHandler
        operation_handler = OperationHandler(self.ui_components)
        operation_handler._execute_preprocessing_with_api()
    
    def set_cleanup_confirmed(self) -> None:
        """Set cleanup confirmation dan trigger execution"""
        self.log_info("✅ Konfirmasi cleanup diterima, memulai pembersihan...")
        self.ui_components['_cleanup_confirmed'] = True
        
        # Import dan execute dari operation handler
        from .operation_handlers import OperationHandler
        operation_handler = OperationHandler(self.ui_components)
        operation_handler._execute_cleanup_with_api()
        
        # Clear dialog area
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        clear_dialog_area(self.ui_components)
    
    def handle_preprocessing_cancel(self) -> None:
        """Handle preprocessing cancellation"""
        self.handle_operation_cancel('preprocessing', '_preprocessing_confirmed')
    
    def handle_cleanup_cancel(self) -> None:
        """Handle cleanup cancellation dengan dialog cleanup"""
        self.handle_operation_cancel('cleanup', '_cleanup_confirmed')
        
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        clear_dialog_area(self.ui_components)
    
    def clear_all_confirmations(self) -> None:
        """Clear semua confirmation flags"""
        from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
        
        confirmation_flags = ['_preprocessing_confirmed', '_cleanup_confirmed']
        for flag in confirmation_flags:
            self.ui_components.pop(flag, None)
        
        clear_dialog_area(self.ui_components)
        self.log_debug("🧹 Semua confirmation flags dibersihkan")

# Factory function untuk backward compatibility
def setup_confirmation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Factory function untuk setup confirmation handlers"""
    handler = ConfirmationHandler(ui_components)
    return handler.setup_handlers()