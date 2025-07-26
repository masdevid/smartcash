"""
File: smartcash/ui/dataset/preprocessing/operations/check_operation.py
Description: Operation handler for validating the data and configuration.
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
from .preprocessing_operation_base import BasePreprocessingOperation


class CheckOperationHandler(BasePreprocessingOperation):
    """
    Orchestrates the data and configuration validation check by calling the backend.
    """

    def execute(self) -> Dict[str, Any]:
        """Executes the validation check by calling the backend service."""
        self.log_info("🔍 Menghubungkan ke backend untuk memeriksa status...")
        try:
            status = get_preprocessing_status(config=self.config)
            summary = self._format_status_summary(status)

            if status.get('service_ready'):
                self.log_success("✅ Pemeriksaan berhasil. Ringkasan status dibuat.")
                self._execute_callback('on_success', summary)
                return {'success': True, 'message': 'Pemeriksaan berhasil diselesaikan'}
            else:
                error_message = f"❌ Backend melaporkan layanan belum siap."
                self.log_error(error_message)
                self._execute_callback('on_failure', summary) # Still show summary on failure
                return {'success': False, 'message': 'Backend tidak siap'}

        except Exception as e:
            error_message = f"Gagal memanggil backend pemeriksaan status: {e}"
            self.log_error(f"❌ {error_message}")
            self._execute_callback('on_failure', error_message)
            return {'success': False, 'message': f'Error: {e}'}
        finally:
            self._execute_callback('on_complete')

    def _format_status_summary(self, status: Dict[str, Any]) -> str:
        """Formats the backend status into a user-friendly markdown summary."""
        from smartcash.ui.core.utils.summary_formatter import UnifiedSummaryFormatter
        
        # Transform status data to match formatter expectations
        formatted_result = {
            'success': status.get('service_ready', False),
            'message': status.get('message', 'Pemeriksaan selesai.'),
            'statistics': {
                'files_processed': status.get('file_statistics', {}).get('train', {}).get('preprocessed_files', 0),
                'files_missing': status.get('file_statistics', {}).get('train', {}).get('missing_files', 0),
                'raw_images': status.get('file_statistics', {}).get('train', {}).get('raw_images', 0)
            },
            'dataset_path': status.get('paths', {}).get('raw_data_path', 'N/A'),
            'output_path': status.get('paths', {}).get('preprocessed_data_path', 'N/A')
        }
        
        return UnifiedSummaryFormatter.format_dataset_summary(
            module_name="preprocessing",
            operation_type="status check", 
            result=formatted_result,
            include_paths=True
        )

# Alias for compatibility
CheckOperation = CheckOperationHandler
