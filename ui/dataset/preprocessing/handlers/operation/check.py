"""
File: smartcash/ui/dataset/preprocessing/handlers/operation/check.py
Deskripsi: Check operation handler dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.error_handler import handle_ui_errors
from smartcash.ui.core.decorators.ui_decorators import safe_ui_operation
from .base_operation import BaseOperationHandler

class CheckOperationHandler(BaseOperationHandler):
    """Check operation handler dengan centralized error handling.
    
    Handles dataset check operation:
    - API integration
    - Progress tracking
    - Result processing
    - Summary reporting
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize check operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(ui_components=ui_components)
        self._operation_name = "check"
        self._operation_title = "Dataset Check"
    
    @handle_ui_errors(error_component_title="Check Error", log_error=True)
    def execute(self) -> Dict[str, Any]:
        """Execute check operation dengan proper error handling.
        
        Returns:
            Check result
        """
        # Prepare operation
        if not self._prepare_operation():
            return {'status': False, 'message': 'Failed to prepare operation'}
        
        # Execute check
        return self._execute_check()
    
    @handle_ui_errors(error_component_title="Check Execution Error", log_error=True)
    def _execute_check(self) -> Dict[str, Any]:
        """Execute check dengan API integration.
        
        Returns:
            Check result
        """
        try:
            # Extract config
            config = self._extract_config()
            
            # Create progress callback
            progress_callback = self._create_progress_callback()
            
            # Execute check API
            from smartcash.api.dataset import check_dataset_status
            result = check_dataset_status(
                config=config,
                progress_callback=progress_callback
            )
            
            # Process result
            self._process_status_result(result)
            
            # Complete operation
            if result.get('status', False):
                self._complete_operation(True, "Dataset check berhasil")
            else:
                error_message = result.get('message', 'Unknown error')
                self._complete_operation(False, f"Dataset check gagal: {error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing check: {str(e)}")
            self._complete_operation(False, f"Error executing check: {str(e)}")
            return {'status': False, 'message': str(e)}
    
    @handle_ui_errors(error_component_title="Status Result Processing Error", log_error=True)
    def _process_status_result(self, result: Dict[str, Any]) -> None:
        """Process dataset status result.
        
        Args:
            result: Status result
        """
        # Check if result is valid
        if not isinstance(result, dict):
            self.logger.error("❌ Invalid status result: not a dictionary")
            return
            
        # Check status
        if not result.get('status', False):
            error_message = result.get('message', 'Unknown error')
            self.logger.error(f"❌ Dataset check gagal: {error_message}")
            self.update_summary(f"Dataset check gagal: {error_message}", "error", "Dataset Check Error")
            return
            
        # Process stats
        stats = result.get('stats', {})
        if not stats:
            self.logger.warning("⚠️ No stats in status result")
            self.update_summary("Dataset check berhasil tapi tidak ada statistik", "warning", "Dataset Check")
            return
            
        # Create summary message
        summary_message = self._create_summary_message(stats)
        self.update_summary(summary_message, "success", "Dataset Check Berhasil")
    
    def _create_summary_message(self, stats: Dict[str, Any]) -> str:
        """Create summary message dari status result.
        
        Args:
            stats: Status statistics
            
        Returns:
            Summary message
        """
        # Extract stats
        raw_files = stats.get('raw_files', 0)
        preprocessed_files = stats.get('preprocessed_files', 0)
        sample_files = stats.get('sample_files', 0)
        splits = stats.get('splits', {})
        
        # Create splits summary
        splits_summary = ""
        for split, count in splits.items():
            splits_summary += f"<li>{split}: {count} files</li>"
        
        # Create summary message
        message = f"""
        <div>
            <p><strong>✅ Dataset Check Berhasil</strong></p>
            <ul>
                <li>Raw files: {raw_files}</li>
                <li>Preprocessed files: {preprocessed_files}</li>
                <li>Sample files: {sample_files}</li>
            </ul>
            <p><strong>Splits:</strong></p>
            <ul>
                {splits_summary}
            </ul>
        </div>
        """
        
        return message
