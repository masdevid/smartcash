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

    def execute(self) -> None:
        """Executes the cleanup workflow by calling the backend service."""
        self.log_operation("🧹 Menghubungkan ke backend untuk pembersihan...", level='info')
        try:
            data_dir = self.config.get('data', {}).get('dir')
            preproc_conf = self.config.get('preprocessing', {})
            target = preproc_conf.get('cleanup_target', 'preprocessed')
            splits = preproc_conf.get('target_splits')

            result = cleanup_preprocessing_files(
                config=self.config
            )
            summary = self._format_cleanup_summary(result)

            if result.get('success'):
                self.log_operation(f" Pembersihan berhasil. {result.get('files_deleted', 0)} file telah dihapus.", level='success')
                self._execute_callback('on_success', summary)
            else:
                self.log_operation(f" Gagal melakukan pembersihan: {result.get('message', 'Alasan tidak diketahui.')}", level='error')
                self._execute_callback('on_failure', summary) # Still show summary on failure

        except Exception as e:
            error_message = f"Gagal memanggil backend pembersihan: {e}"
            self.log_operation(f"❌ {error_message}", level='error')
            self._execute_callback('on_failure', error_message)
        finally:
            self._execute_callback('on_complete')

    def _format_cleanup_summary(self, result: Dict[str, Any]) -> str:
        """Formats the cleanup result into a user-friendly markdown summary."""
        success = result.get('success', False)
        status_icon = "✅" if success else "❌"
        status_text = "Berhasil" if success else "Gagal"

        files_deleted = result.get('files_deleted', 0)
        space_reclaimed = result.get('space_reclaimed_mb', 0)

        return f"""
### Ringkasan Operasi Pembersihan

| Kategori | Status |
| :--- | :--- |
| **Status Operasi** | {status_icon} {status_text} |
| **File Dihapus** | 🗑️ {files_deleted} |
| **Ruang Kosong** | 💾 {space_reclaimed:.2f} MB |

---

**Pesan dari Backend:** *{result.get('message', 'Pembersihan selesai.')}*
"""
