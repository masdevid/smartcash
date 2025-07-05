"""
File: smartcash/ui/dataset/preprocessing/handlers/operation/cleanup.py
Deskripsi: Cleanup operation handler dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.decorators.ui_decorators import safe_ui_operation
from .base_operation import BaseOperationHandler

class CleanupOperationHandler(BaseOperationHandler):
    """Cleanup operation handler dengan centralized error handling.
    
    Handles cleanup operation:
    - Confirmation dialog
    - API integration
    - Progress tracking
    - Result processing
    - Summary reporting
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize cleanup operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(ui_components=ui_components)
        self._operation_name = "cleanup"
        self._operation_title = "Cleanup"
        self._confirmed = False
    
    @handle_ui_errors(error_component_title="Cleanup Error", log_error=True)
    def execute(self) -> Dict[str, Any]:
        """Execute cleanup operation dengan proper error handling.
        
        Returns:
            Cleanup result
        """
        # Check if confirmation is pending
        if self._confirmation_pending:
            self.logger.warning("⚠️ Confirmation dialog masih pending")
            return {'status': False, 'message': 'Confirmation dialog masih pending'}
        
        # Prepare operation
        if not self._prepare_operation():
            return {'status': False, 'message': 'Failed to prepare operation'}
        
        # Show confirmation dialog
        self._show_confirmation()
        
        # If confirmed, execute cleanup
        if self._confirmed:
            return self._execute_cleanup()
        
        return {'status': False, 'message': 'Waiting for confirmation'}
    
    @safe_ui_operation(error_title="Cleanup Confirmation Error")
    def _show_confirmation(self) -> None:
        """Show cleanup confirmation dialog."""
        try:
            # Extract config for confirmation message
            config = self._extract_config()
            cleanup_target = config.get('preprocessing', {}).get('cleanup', {}).get('target', 'preprocessed')
            
            target_descriptions = {
                'preprocessed': '<strong>preprocessing files</strong> (pre_*.npy + pre_*.txt)',
                'samples': '<strong>sample images</strong> (sample_*.jpg)',
                'both': '<strong>preprocessing files dan sample images</strong>'
            }
            
            target_desc = target_descriptions.get(cleanup_target, cleanup_target)
            
            # Show confirmation dialog
            self._confirmation_pending = True
            self.show_confirmation_dialog(
                self.ui_components,
                title="⚠️ Konfirmasi Cleanup",
                message=f"Hapus {target_desc} dari dataset?<br><br><span style='color:#dc3545;'>⚠️ <strong>Tindakan ini tidak dapat dibatalkan!</strong></span><br><br>🗑️ Files akan dihapus permanent<br>📊 Progress tracking tersedia",
                callback=self._handle_confirmation_result,
                confirm_text="🗑️ Ya, Hapus",
                cancel_text="❌ Batal",
                danger_mode=True
            )
        except Exception as e:
            self.logger.error(f"❌ Error showing confirmation: {str(e)}")
            self._confirmation_pending = False
            self._complete_operation(False, f"Error showing confirmation: {str(e)}")
    
    @safe_ui_operation(error_title="Cleanup Confirmation Result Error")
    def _handle_confirmation_result(self, confirmed: bool) -> None:
        """Handle cleanup confirmation result.
        
        Args:
            confirmed: Whether cleanup was confirmed
        """
        self._confirmation_pending = False
        
        if confirmed:
            self._confirmed = True
            self.logger.info("✅ Cleanup dikonfirmasi, memulai...")
            self._execute_cleanup()
        else:
            self._confirmed = False
            self.logger.info("🚫 Cleanup dibatalkan oleh user")
            self._complete_operation(False, "Cleanup dibatalkan oleh user")
    
    @handle_ui_errors(error_component_title="Cleanup Execution Error", log_error=True)
    def _execute_cleanup(self) -> Dict[str, Any]:
        """Execute cleanup dengan API integration.
        
        Returns:
            Cleanup result
        """
        try:
            # Extract config
            config = self._extract_config()
            
            # Create progress callback
            progress_callback = self._create_progress_callback()
            
            # Execute cleanup API
            from smartcash.api.dataset import cleanup_preprocessing
            result = cleanup_preprocessing(
                config=config,
                progress_callback=progress_callback
            )
            
            # Process result
            if result.get('status', False):
                self._process_cleanup_result(result)
                self._complete_operation(True, "Cleanup berhasil")
            else:
                error_message = result.get('message', 'Unknown error')
                self.logger.error(f"❌ Cleanup gagal: {error_message}")
                self._complete_operation(False, f"Cleanup gagal: {error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing cleanup: {str(e)}")
            self._complete_operation(False, f"Error executing cleanup: {str(e)}")
            return {'status': False, 'message': str(e)}
    
    @handle_ui_errors(error_component_title="Cleanup Result Processing Error", log_error=True)
    def _process_cleanup_result(self, result: Dict[str, Any]) -> None:
        """Process cleanup result.
        
        Args:
            result: Cleanup result
        """
        # Log success
        self.logger.info("✅ Cleanup berhasil")
        
        # Update summary with details
        summary_message = self._create_summary_message(result)
        self.update_summary(summary_message, "success", "Cleanup Berhasil")
    
    def _create_summary_message(self, result: Dict[str, Any]) -> str:
        """Create summary message dari cleanup result.
        
        Args:
            result: Cleanup result
            
        Returns:
            Summary message
        """
        # Extract stats
        stats = result.get('stats', {})
        deleted_files = stats.get('deleted_files', 0)
        target = stats.get('target', 'unknown')
        
        # Create target description
        target_descriptions = {
            'preprocessed': 'preprocessing files (pre_*.npy + pre_*.txt)',
            'samples': 'sample images (sample_*.jpg)',
            'both': 'preprocessing files dan sample images'
        }
        
        target_desc = target_descriptions.get(target, target)
        
        # Create summary message
        message = f"""
        <div>
            <p><strong>✅ Cleanup Berhasil</strong></p>
            <ul>
                <li>Target: {target_desc}</li>
                <li>Deleted files: {deleted_files}</li>
            </ul>
            <p>Dataset telah dibersihkan</p>
        </div>
        """
        
        return message
