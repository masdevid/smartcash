"""
File: smartcash/ui/dataset/preprocessing/handlers/confirmation_handlers.py
Deskripsi: Enhanced confirmation handlers dengan optimized glass morphism dialog integration
"""

from typing import Dict, Any
from .base_handler import BasePreprocessingHandler

class ConfirmationHandler(BasePreprocessingHandler):
    """Enhanced confirmation handler dengan robust dialog management"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        super().__init__(ui_components)
        self._ensure_dialog_area()
    
    def _ensure_dialog_area(self) -> None:
        """Ensure confirmation area exists dan properly configured"""
        from smartcash.ui.components.dialog.confirmation_dialog import create_confirmation_area
        
        if 'confirmation_area' not in self.ui_components:
            self.ui_components['confirmation_area'] = create_confirmation_area(self.ui_components)
            self.log_debug("🎨 Dialog area berhasil dibuat")
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup confirmation management functions dengan enhanced error handling"""
        handlers = {
            'show_preprocessing_confirmation': self._show_preprocessing_confirmation_safe,
            'show_cleanup_confirmation': self._show_cleanup_confirmation_safe,
            'set_preprocessing_confirmed': self._set_preprocessing_confirmed_safe,
            'set_cleanup_confirmed': self._set_cleanup_confirmed_safe,
            'handle_preprocessing_cancel': self._handle_preprocessing_cancel_safe,
            'handle_cleanup_cancel': self._handle_cleanup_cancel_safe,
            'is_confirmation_pending': self.is_confirmation_pending,
            'clear_confirmations': self._clear_all_confirmations_safe,
            'reset_dialog_state': self._reset_dialog_state
        }
        
        self.log_debug("✅ Enhanced confirmation handlers setup completed")
        return handlers
    
    def _show_preprocessing_confirmation_safe(self) -> None:
        """Show preprocessing confirmation dengan enhanced validation"""
        try:
            # Validate state sebelum showing dialog
            if self.is_confirmation_pending():
                self.log_warning("⚠️ Dialog confirmation lain masih aktif, clearing state...")
                self._reset_dialog_state()
            
            config = self.extract_config()
            preprocessing_config = config.get('preprocessing', {})
            
            # Build confirmation message dengan config details
            message_parts = [
                "Apakah Anda yakin ingin memproses dataset dengan YOLO normalization?",
                "",
                "📋 Konfigurasi yang akan digunakan:"
            ]
            
            # Add key config details
            if preprocessing_config.get('resize_enabled'):
                size = preprocessing_config.get('target_size', [640, 640])
                message_parts.append(f"• Resize ke: {size[0]}x{size[1]}px")
            
            if preprocessing_config.get('augmentation_enabled'):
                message_parts.append("• Augmentasi: Diaktifkan")
            
            normalization = preprocessing_config.get('normalization', 'yolo')
            message_parts.append(f"• Format: {normalization.upper()}")
            
            message = "\n".join(message_parts)
            
            self.show_confirmation_dialog(
                title="🚀 Konfirmasi Preprocessing Dataset",
                message=message,
                on_confirm=self._set_preprocessing_confirmed_safe,
                on_cancel=self._handle_preprocessing_cancel_safe,
                confirm_text="Ya, Proses Dataset",
                cancel_text="Batal"
            )
            
            self.log_info("⏳ Menunggu konfirmasi preprocessing...")
            
        except Exception as e:
            self.log_error(f"❌ Error showing preprocessing confirmation: {str(e)}")
            self._handle_dialog_error("preprocessing", e)
    
    def _show_cleanup_confirmation_safe(self) -> None:
        """Show cleanup confirmation dengan enhanced target info"""
        try:
            if self.is_confirmation_pending():
                self.log_warning("⚠️ Dialog confirmation lain masih aktif, clearing state...")
                self._reset_dialog_state()
            
            config = self.extract_config()
            cleanup_config = config.get('preprocessing', {}).get('cleanup', {})
            cleanup_target = cleanup_config.get('target', 'preprocessed')
            
            # Enhanced target descriptions dengan file counts
            target_info = self._get_cleanup_target_info(cleanup_target)
            
            message = f"""Hapus {target_info['description']}?

📁 Target cleanup: {target_info['files']}

⚠️ Tindakan ini akan menghapus file-file yang sudah diproses.
Pastikan Anda sudah backup data penting sebelum melanjutkan."""
            
            self.show_confirmation_dialog(
                title="🧹 Konfirmasi Cleanup Dataset",
                message=message,
                on_confirm=self._set_cleanup_confirmed_safe,
                on_cancel=self._handle_cleanup_cancel_safe,
                confirm_text="Ya, Hapus File",
                cancel_text="Batal",
                danger_mode=True
            )
            
            self.log_info(f"⏳ Menunggu konfirmasi cleanup untuk: {target_info['description']}")
            
        except Exception as e:
            self.log_error(f"❌ Error showing cleanup confirmation: {str(e)}")
            self._handle_dialog_error("cleanup", e)
    
    def _get_cleanup_target_info(self, target: str) -> Dict[str, str]:
        """Get detailed cleanup target information"""
        target_map = {
            'preprocessed': {
                'description': 'file preprocessing hasil normalisasi',
                'files': 'pre_*.npy + pre_*.txt files'
            },
            'samples': {
                'description': 'sample images yang sudah di-generate',
                'files': 'sample_*.jpg files'
            },
            'both': {
                'description': 'semua file preprocessing dan sample images',
                'files': 'pre_*.npy + pre_*.txt + sample_*.jpg'
            }
        }
        
        return target_map.get(target, {
            'description': f'target "{target}"',
            'files': f'{target} files'
        })
    
    def _set_preprocessing_confirmed_safe(self) -> None:
        """Confirm dan execute preprocessing dengan enhanced flow control"""
        try:
            self.ui_components['_preprocessing_confirmed'] = True
            self.log_info("✅ Konfirmasi preprocessing diterima, memulai proses...")
            
            # Clear dialog state
            self._reset_dialog_state()
            
            # Execute preprocessing via operation handler
            from .operation_handlers import OperationHandler
            operation_handler = OperationHandler(self.ui_components)
            operation_handler._execute_preprocessing_with_api()
            
        except Exception as e:
            self.log_error(f"❌ Error executing preprocessing: {str(e)}")
            self._handle_execution_error("preprocessing", e)
    
    def _set_cleanup_confirmed_safe(self) -> None:
        """Confirm dan execute cleanup dengan enhanced flow control"""
        try:
            self.ui_components['_cleanup_confirmed'] = True
            self.log_info("✅ Konfirmasi cleanup diterima, memulai pembersihan...")
            
            # Clear dialog state
            self._reset_dialog_state()
            
            # Execute cleanup via operation handler
            from .operation_handlers import OperationHandler
            operation_handler = OperationHandler(self.ui_components)
            operation_handler._execute_cleanup_with_api()
            
        except Exception as e:
            self.log_error(f"❌ Error executing cleanup: {str(e)}")
            self._handle_execution_error("cleanup", e)
    
    def _handle_preprocessing_cancel_safe(self) -> None:
        """Handle preprocessing cancellation dengan enhanced cleanup"""
        try:
            self._handle_operation_cancel_safe('preprocessing', '_preprocessing_confirmed')
        except Exception as e:
            self.log_error(f"❌ Error handling preprocessing cancel: {str(e)}")
    
    def _handle_cleanup_cancel_safe(self) -> None:
        """Handle cleanup cancellation dengan enhanced dialog cleanup"""
        try:
            self._handle_operation_cancel_safe('cleanup', '_cleanup_confirmed')
            self._reset_dialog_state()
        except Exception as e:
            self.log_error(f"❌ Error handling cleanup cancel: {str(e)}")
    
    def _handle_operation_cancel_safe(self, operation: str, flag_key: str) -> None:
        """Enhanced operation cancellation handler dengan proper state management"""
        try:
            # Reset confirmation flag
            self.ui_components[flag_key] = False
            
            # Log cancellation
            self.log_info(f"❌ {operation.title()} dibatalkan oleh user")
            
            # Reset dialog state
            self._reset_dialog_state()
            
            # Re-enable UI buttons jika ada
            if hasattr(self, '_enable_ui_buttons'):
                self._enable_ui_buttons()
            
            # Update status panel
            from smartcash.ui.utils.status_utils import update_status_panel_enhanced
            update_status_panel_enhanced(
                self.ui_components, 
                f"{operation.title()} dibatalkan", 
                'warning', 
                force_update=True
            )
            
        except Exception as e:
            self.log_error(f"❌ Error in cancel handler for {operation}: {str(e)}")
    
    def _reset_dialog_state(self) -> None:
        """Reset dialog state untuk prevent show/hide issues"""
        try:
            from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
            clear_dialog_area(self.ui_components)
            
            # Clear any pending confirmation flags
            confirmation_flags = ['_preprocessing_confirmed', '_cleanup_confirmed']
            for flag in confirmation_flags:
                if flag in self.ui_components and self.ui_components[flag] is not True:
                    self.ui_components.pop(flag, None)
            
            self.log_debug("🔄 Dialog state berhasil direset")
            
        except Exception as e:
            self.log_error(f"❌ Error resetting dialog state: {str(e)}")
    
    def _clear_all_confirmations_safe(self) -> None:
        """Clear semua confirmation flags dengan enhanced cleanup"""
        try:
            self._reset_dialog_state()
            
            # Clear all confirmation related keys
            keys_to_clear = [k for k in self.ui_components.keys() 
                           if k.endswith('_confirmed') or k.startswith('_dialog_')]
            
            for key in keys_to_clear:
                self.ui_components.pop(key, None)
            
            self.log_debug("🧹 Semua confirmation flags berhasil dibersihkan")
            
        except Exception as e:
            self.log_error(f"❌ Error clearing confirmations: {str(e)}")
    
    def _handle_dialog_error(self, operation: str, error: Exception) -> None:
        """Handle dialog error dengan fallback ke console confirmation"""
        try:
            self.log_warning(f"⚠️ Dialog error untuk {operation}, fallback ke console input")
            
            # Simple console fallback
            response = input(f"Konfirmasi {operation}? (y/N): ").lower().strip()
            
            if response in ['y', 'yes', 'ya']:
                if operation == 'preprocessing':
                    self._set_preprocessing_confirmed_safe()
                elif operation == 'cleanup':
                    self._set_cleanup_confirmed_safe()
            else:
                if operation == 'preprocessing':
                    self._handle_preprocessing_cancel_safe()
                elif operation == 'cleanup':
                    self._handle_cleanup_cancel_safe()
                    
        except Exception as fallback_error:
            self.log_error(f"❌ Fallback error: {str(fallback_error)}")
    
    def _handle_execution_error(self, operation: str, error: Exception) -> None:
        """Handle execution error dengan proper cleanup"""
        try:
            # Reset confirmation state
            flag_key = f'_{operation}_confirmed'
            self.ui_components[flag_key] = False
            
            # Reset dialog state
            self._reset_dialog_state()
            
            # Update status dengan error
            from smartcash.ui.utils.status_utils import update_status_panel_enhanced
            update_status_panel_enhanced(
                self.ui_components,
                f"Error {operation}: {str(error)}",
                'error',
                force_update=True
            )
            
        except Exception as cleanup_error:
            self.log_error(f"❌ Error in execution error handler: {str(cleanup_error)}")
    
    def is_confirmation_pending(self) -> bool:
        """Check apakah ada confirmation yang sedang pending"""
        try:
            # Check dialog visibility
            from smartcash.ui.components.dialog.confirmation_dialog import is_dialog_visible
            if is_dialog_visible(self.ui_components):
                return True
            
            # Check confirmation flags yang belum resolved
            pending_flags = [
                '_preprocessing_confirmed',
                '_cleanup_confirmed'
            ]
            
            for flag in pending_flags:
                if flag in self.ui_components and self.ui_components[flag] is None:
                    return True
            
            return False
            
        except Exception as e:
            self.log_error(f"❌ Error checking confirmation pending: {str(e)}")
            return False
    
    def _enable_ui_buttons(self) -> None:
        """Re-enable UI buttons setelah operation cancel"""
        try:
            # Import dan panggil UI utils untuk enable buttons
            from smartcash.ui.dataset.preprocessing.utils import ui_utils
            ui_utils.enable_buttons(self.ui_components)
            
        except Exception as e:
            self.log_debug(f"⚠️ Could not re-enable buttons: {str(e)}")

# Factory function untuk backward compatibility dengan enhanced error handling
def setup_confirmation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Factory function untuk setup enhanced confirmation handlers
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary handlers yang telah disetup
        
    Raises:
        ValueError: Jika setup gagal
    """
    try:
        handler = ConfirmationHandler(ui_components)
        handlers = handler.setup_handlers()
        
        # Validate handlers setup
        required_handlers = [
            'show_preprocessing_confirmation',
            'show_cleanup_confirmation',
            'clear_confirmations'
        ]
        
        for required_handler in required_handlers:
            if required_handler not in handlers:
                raise ValueError(f"Required handler {required_handler} not setup")
        
        return handlers
        
    except Exception as e:
        print(f"❌ Error setup confirmation handlers: {str(e)}")
        raise ValueError(f"Gagal setup confirmation handlers: {str(e)}") from e