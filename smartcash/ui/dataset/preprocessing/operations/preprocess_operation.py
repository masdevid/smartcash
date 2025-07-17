"""
File: smartcash/ui/dataset/preprocessing/operations/preprocess_operation.py
Description: Operation handler for the main data preprocessing workflow.
"""

import time
from typing import Any, Dict, Optional, Callable
from smartcash.dataset.preprocessor.api.preprocessing_api import preprocess_dataset
from .preprocessing_operation_base import BasePreprocessingOperation


class PreprocessOperationHandler(BasePreprocessingOperation):
    """
    Orchestrates the main data preprocessing workflow by calling the backend service.
    """

    def execute(self) -> None:
        """Executes the preprocessing workflow by calling the backend service."""
        self.log_operation("🚀 Menghubungkan ke backend untuk pra-pemrosesan...", level='info')

        try:
            result = preprocess_dataset(
                config=self.config,
                progress_callback=self._progress_adapter
            )
            summary = self._format_preprocess_summary(result)

            if result.get('success'):
                self.log_operation(f"✅ Pra-pemrosesan berhasil.", level='success')
                self._execute_callback('on_success', summary)
            else:
                self.log_operation(f"❌ Gagal melakukan pra-pemrosesan: {result.get('message', 'Alasan tidak diketahui.')}", level='error')
                self._execute_callback('on_failure', summary) # Still show summary on failure

        except Exception as e:
            error_message = f"Gagal memanggil backend pra-pemrosesan: {e}"
            self.log_operation(f"❌ {error_message}", level='error')
            self._execute_callback('on_failure', error_message)
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
