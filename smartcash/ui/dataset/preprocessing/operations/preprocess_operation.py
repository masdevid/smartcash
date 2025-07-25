"""
File: smartcash/ui/dataset/preprocessing/operations/preprocess_operation.py
Description: Operation handler for the main data preprocessing workflow.
"""

from typing import Any, Dict
from smartcash.dataset.preprocessor.api.preprocessing_api import preprocess_dataset
from .preprocessing_operation_base import BasePreprocessingOperation


class PreprocessOperation(BasePreprocessingOperation):
    """
    Orchestrates the main data preprocessing workflow by calling the backend service.
    """

    def execute(self) -> Dict[str, Any]:
        """Executes the preprocessing workflow by calling the backend service."""
        self.log_info("🚀 Menghubungkan ke backend untuk pra-pemrosesan...")

        try:
            result = preprocess_dataset(
                config=self.config,
                progress_callback=self._progress_adapter
            )
            summary = self._format_preprocess_summary(result)

            if result.get('success'):
                self.log_success(f"✅ Pra-pemrosesan berhasil.")
                self._execute_callback('on_success', summary)
                return {'success': True, 'message': 'Preprocessing berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Alasan tidak diketahui.')
                self.log_error(f"❌ Gagal melakukan pra-pemrosesan: {error_msg}")
                self._execute_callback('on_failure', summary) # Still show summary on failure
                return {'success': False, 'message': f'Preprocessing gagal: {error_msg}'}

        except Exception as e:
            error_message = f"Gagal memanggil backend pra-pemrosesan: {e}"
            self.log_error(f"❌ {error_message}")
            self._execute_callback('on_failure', error_message)
            return {'success': False, 'message': f'Error: {e}'}
        finally:
            self._execute_callback('on_complete')

    def _format_preprocess_summary(self, result: Dict[str, Any]) -> str:
        """Formats the preprocessing result into a user-friendly markdown summary."""
        success = result.get('success', False)
        status_icon = "✅" if success else "❌"
        status_text = "Berhasil" if success else "Gagal"

        stats = result.get('statistics', {})
        files_processed = stats.get('files_processed', 0)
        files_skipped = stats.get('files_skipped', 0)
        files_failed = stats.get('files_failed', 0)
        total_time = result.get('total_time_seconds', 0)

        return f"""
### Ringkasan Operasi Pra-pemrosesan

| Kategori | Status |
| :--- | :--- |
| **Status Operasi** | {status_icon} {status_text} |
| **File Diproses** | ✔️ {files_processed} |
| **File Dilewati** | ⏭️ {files_skipped} |
| **File Gagal** | ❌ {files_failed} |
| **Total Waktu** | ⏱️ {total_time:.2f} detik |

---

**Pesan dari Backend:** *{result.get('message', 'Pra-pemrosesan selesai.')}*
"""

