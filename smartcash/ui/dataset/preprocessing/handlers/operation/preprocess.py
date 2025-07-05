"""
File: smartcash/ui/dataset/preprocessing/handlers/operation/preprocess.py
Deskripsi: Preprocessing operation handler dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.core.decorators.ui_decorators import safe_ui_operation
from .base_operation import BaseOperationHandler

class PreprocessOperationHandler(BaseOperationHandler):
    """Preprocessing operation handler dengan centralized error handling.
    
    Handles preprocessing operation:
    - Confirmation dialog
    - API integration
    - Progress tracking
    - Result processing
    - Summary reporting
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize preprocessing operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(ui_components=ui_components)
        self._operation_name = "preprocessing"
        self._operation_title = "Preprocessing"
        self._confirmed = False
    
    @handle_ui_errors(error_component_title="Preprocessing Error", log_error=True)
    def execute(self) -> Dict[str, Any]:
        """Execute preprocessing operation dengan proper error handling.
        
        Returns:
            Preprocessing result
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
        
        # If confirmed, execute preprocessing
        if self._confirmed:
            return self._execute_preprocessing()
        
        return {'status': False, 'message': 'Waiting for confirmation'}
    
    @safe_ui_operation(error_title="Preprocessing Confirmation Error")
    def _show_confirmation(self) -> None:
        """Show preprocessing confirmation dialog."""
        try:
            # Extract config for confirmation message
            config = self._extract_config()
            target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
            target_splits_str = ', '.join(target_splits)
            
            # Show confirmation dialog
            self._confirmation_pending = True
            self.show_confirmation_dialog(
                self.ui_components,
                title="⚠️ Konfirmasi Preprocessing",
                message=f"Preprocess dataset untuk splits: <strong>{target_splits_str}</strong>?<br><br>🔄 Preprocessing akan membuat file pre_*.npy dan pre_*.txt<br>📊 Progress tracking tersedia",
                callback=self._handle_confirmation_result,
                confirm_text="✅ Ya, Preprocess",
                cancel_text="❌ Batal"
            )
        except Exception as e:
            self.logger.error(f"❌ Error showing confirmation: {str(e)}")
            self._confirmation_pending = False
            self._complete_operation(False, f"Error showing confirmation: {str(e)}")
    
    @safe_ui_operation(error_title="Preprocessing Confirmation Result Error")
    def _handle_confirmation_result(self, confirmed: bool) -> None:
        """Handle preprocessing confirmation result.
        
        Args:
            confirmed: Whether preprocessing was confirmed
        """
        self._confirmation_pending = False
        
        if confirmed:
            self._confirmed = True
            self.logger.info("✅ Preprocessing dikonfirmasi, memulai...")
            self._execute_preprocessing()
        else:
            self._confirmed = False
            self.logger.info("🚫 Preprocessing dibatalkan oleh user")
            self._complete_operation(False, "Preprocessing dibatalkan oleh user")
    
    @handle_ui_errors(error_component_title="Preprocessing Execution Error", log_error=True)
    def _execute_preprocessing(self) -> Dict[str, Any]:
        """Execute preprocessing dengan API integration.
        
        Returns:
            Preprocessing result
        """
        try:
            # Extract config
            config = self._extract_config()
            
            # Create progress callback
            progress_callback = self._create_progress_callback()
            
            # Execute preprocessing API
            from smartcash.api.dataset import preprocess_dataset
            result = preprocess_dataset(
                config=config,
                progress_callback=progress_callback
            )
            
            # Process result
            if result.get('status', False):
                self._process_success_result(result)
                self._complete_operation(True, "Preprocessing berhasil")
            else:
                error_message = result.get('message', 'Unknown error')
                self.logger.error(f"❌ Preprocessing gagal: {error_message}")
                self._complete_operation(False, f"Preprocessing gagal: {error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing preprocessing: {str(e)}")
            self._complete_operation(False, f"Error executing preprocessing: {str(e)}")
            return {'status': False, 'message': str(e)}
    
    @handle_ui_errors(error_component_title="Result Processing Error", log_error=True)
    def _process_success_result(self, result: Dict[str, Any]) -> None:
        """Process successful preprocessing result.
        
        Args:
            result: Preprocessing result
        """
        # Log success
        self.logger.info("✅ Preprocessing berhasil")
        
        # Update summary with details
        summary_message = self._create_summary_message(result)
        self.update_summary(summary_message, "success", "Preprocessing Berhasil")
    
    def _create_summary_message(self, result: Dict[str, Any]) -> str:
        """Create summary message dari preprocessing result.
        
        Args:
            result: Preprocessing result
            
        Returns:
            Summary message
        """
        # Extract stats
        stats = result.get('stats', {})
        total_files = stats.get('total_files', 0)
        processed_files = stats.get('processed_files', 0)
        skipped_files = stats.get('skipped_files', 0)
        
        # Create summary message
        message = f"""
        <div>
            <p><strong>✅ Preprocessing Berhasil</strong></p>
            <ul>
                <li>Total files: {total_files}</li>
                <li>Processed files: {processed_files}</li>
                <li>Skipped files: {skipped_files}</li>
            </ul>
            <p>Preprocessing files tersimpan di direktori data</p>
        </div>
        """
        
        return message
