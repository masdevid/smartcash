"""
File: smartcash/ui/dataset/preprocessing/tests/__init__.py
Deskripsi: Package untuk tests preprocessing dataset
"""

# Export tests
from smartcash.ui.dataset.preprocessing.tests.test_preprocessing_utils import TestPreprocessingUtils
from smartcash.ui.dataset.preprocessing.tests.test_cell import TestPreprocessingCell

# Export test runner
from smartcash.ui.dataset.preprocessing.tests.run_tests import run_all_tests

__all__ = [
    'TestPreprocessingUtils',
    'TestPreprocessingCell',
    'run_all_tests'
]

# Informasi tentang tests yang tersedia
"""
Tests yang tersedia:

1. TestPreprocessingUtils - Unit tests untuk utilitas preprocessing
   * test_is_preprocessing_running_true - Cek status preprocessing running (true)
   * test_is_preprocessing_running_false - Cek status preprocessing running (false)
   * test_set_preprocessing_state - Tes set state preprocessing
   * test_toggle_widgets - Tes toggle status widget
   * test_update_status_panel - Tes update panel status
   * test_get_widget_value - Tes mendapatkan nilai widget
   * test_log_message - Tes logging pesan
   * test_initialize_preprocessing_ui - Tes inisialisasi UI preprocessing

2. TestPreprocessingCell - Test untuk cell preprocessing
   * test_cell_execution - Tes eksekusi cell preprocessing

Untuk menjalankan semua test:
python -m smartcash.ui.dataset.preprocessing.tests.run_tests
"""
