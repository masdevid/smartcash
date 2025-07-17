"""
File: smartcash/ui/dataset/preprocessing/operations/check_operation.py
Description: Operation handler for validating the data and configuration.
"""

import time
from typing import Dict, Any
from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
from .preprocessing_operation_base import BasePreprocessingOperation


class CheckOperationHandler(BasePreprocessingOperation):
    """
    Orchestrates the data and configuration validation check by calling the backend.
    """

    def execute(self) -> None:
        """Executes the validation check by calling the backend service."""
        self.log_operation("🔍 Menghubungkan ke backend untuk memeriksa status...", level='info')
        try:
            status = get_preprocessing_status(config=self.config)
            summary = self._format_status_summary(status)

            if status.get('service_ready'):
                self.log_operation("✅ Pemeriksaan berhasil. Ringkasan status dibuat.", level='success')
                self._execute_callback('on_success', summary)
            else:
                error_message = f"❌ Backend melaporkan layanan belum siap."
                self.log_operation(error_message, level='error')
                self._execute_callback('on_failure', summary) # Still show summary on failure

        except Exception as e:
            error_message = f"Gagal memanggil backend pemeriksaan status: {e}"
            self.log_operation(f"❌ {error_message}", level='error')
            self._execute_callback('on_failure', error_message)
        finally:
            self._execute_callback('on_complete')

    def _format_status_summary(self, status: Dict[str, Any]) -> str:
        """Formats the backend status into a user-friendly markdown summary."""
        service_ready = status.get('service_ready', False)
        service_status_icon = "✅" if service_ready else "❌"
        service_status_text = "Siap" if service_ready else "Tidak Siap"

        stats = status.get('file_statistics', {}).get('train', {})
        raw_images = stats.get('raw_images', 0)
        preprocessed_files = stats.get('preprocessed_files', 0)
        missing_files = stats.get('missing_files', 0)

        paths = status.get('paths', {})
        raw_path = paths.get('raw_data_path', 'N/A')
        preprocessed_path = paths.get('preprocessed_data_path', 'N/A')

        return f"""
### Ringkasan Status Pra-pemrosesan

| Kategori | Status |
| :--- | :--- |
| **Status Layanan** | {service_status_icon} {service_status_text} |
| **Gambar Mentah** | 🖼️ {raw_images} |
| **File Diproses** | ✨ {preprocessed_files} |
| **File Hilang** | ❓ {missing_files} |

---

#### Lokasi Dataset
- **Data Mentah:** `{raw_path}`
- **Data Diproses:** `{preprocessed_path}`

---

**Pesan dari Backend:** *{status.get('message', 'Pemeriksaan selesai.')}*
"""
