"""
File: smartcash/ui/dataset/preprocessing/operations/cleanup_operation.py
Description: Operation handler for cleaning up generated data.
"""

import time
from typing import Dict, Any
from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files

from .preprocessing_operation_base import BasePreprocessingOperation


class CleanupOperationHandler(BasePreprocessingOperation):
    """
    Orchestrates the cleanup of generated data by calling the backend service.
    """

    def execute(self) -> Dict[str, Any]:
        """Executes the cleanup workflow by calling the backend service."""
        self.log_info("🧹 Menghubungkan ke backend untuk pembersihan...")
        
        # Initialize progress tracking
        self.update_progress(0, "Memulai proses pembersihan...")
        
        try:
            # Update progress during cleanup preparation
            self.update_progress(10, "Membaca konfigurasi pembersihan...")
            
            data_dir = self.config.get('data', {}).get('dir')
            preproc_conf = self.config.get('preprocessing', {})
            target = preproc_conf.get('cleanup_target', 'preprocessed')
            splits = preproc_conf.get('target_splits')

            self.update_progress(25, "Menghubungkan ke backend...")
            
            # Create a progress callback for the cleanup operation
            def cleanup_progress_callback(step: str, current: int, total: int, message: str = ''):
                """Progress callback for cleanup operation."""
                # Map cleanup progress to 25-90% range
                progress = 25 + int(((current / total) * 65)) if total > 0 else 25
                self.update_progress(
                    progress=progress,
                    message=f"Membersihkan: {message}",
                    secondary_progress=int((current / total) * 100) if total > 0 else 0,
                    secondary_message=f"{step}: {current}/{total}"
                )

            result = cleanup_preprocessing_files(
                config=self.config
            )
            
            self.update_progress(90, "Memformat ringkasan pembersihan...")
            
            summary = self._format_cleanup_summary(result)

            if result.get('success'):
                files_deleted = result.get('files_deleted', 0)
                self.update_progress(100, f"Pembersihan berhasil - {files_deleted} file dihapus")
                self.log_success(f"✅ Pembersihan berhasil. {files_deleted} file telah dihapus.")
                self._execute_callback('on_success', summary)
                return {'success': True, 'message': f'Pembersihan berhasil, {files_deleted} file dihapus'}
            else:
                error_msg = result.get('message', 'Alasan tidak diketahui.')
                self.update_progress(100, f"Pembersihan gagal: {error_msg}")
                self.log_error(f"❌ Gagal melakukan pembersihan: {error_msg}")
                self._execute_callback('on_failure', summary) # Still show summary on failure
                return {'success': False, 'message': f'Pembersihan gagal: {error_msg}'}

        except Exception as e:
            error_message = f"Gagal memanggil backend pembersihan: {e}"
            self.update_progress(100, f"Error: {e}")
            self.log_error(f"❌ {error_message}")
            self._execute_callback('on_failure', error_message)
            return {'success': False, 'message': f'Error: {e}'}
        finally:
            self._execute_callback('on_complete')

    def _format_cleanup_summary(self, result: Dict[str, Any]) -> str:
        """Formats the cleanup result into a user-friendly markdown summary."""
        from smartcash.ui.core.utils.summary_formatter import UnifiedSummaryFormatter
        
        # Transform cleanup data to match formatter expectations
        formatted_result = {
            'success': result.get('success', False),
            'message': result.get('message', 'Pembersihan selesai.'),
            'statistics': {
                'files_deleted': result.get('files_deleted', 0),
                'space_reclaimed_mb': result.get('space_reclaimed_mb', 0)
            }
        }
        
        return UnifiedSummaryFormatter.format_dataset_summary(
            module_name="preprocessing",
            operation_type="cleanup", 
            result=formatted_result,
            include_paths=False
        )
