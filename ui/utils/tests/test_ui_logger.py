"""
File: smartcash/ui/utils/tests/test_ui_logger.py
Deskripsi: Test untuk UI logger dengan fokus pada buffer log awal
"""

import unittest
from unittest.mock import MagicMock, patch
import threading
import ipywidgets as widgets
import sys
import io
import importlib

class TestUILogger(unittest.TestCase):
    """Test case untuk ui_logger.py dengan fokus pada buffer log awal"""
    
    def setUp(self):
        """Setup untuk test"""
        # Reset variabel global di ui_logger
        from smartcash.ui.utils import ui_logger
        importlib.reload(ui_logger)  # Reload modul untuk memastikan variabel global di-reset
        ui_logger._EARLY_LOG_BUFFER = []
        ui_logger._UI_READY = False
        
        # Simpan stdout asli
        self.original_stdout = sys.stdout
        # Buat StringIO untuk menangkap output
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
        
        # Simpan sys.modules asli
        self.original_modules = sys.modules.copy()
    
    def tearDown(self):
        """Cleanup setelah test"""
        # Kembalikan stdout asli
        sys.stdout = self.original_stdout
        
        # Kembalikan sys.modules asli
        sys.modules = self.original_modules
    
    def test_early_log_buffer(self):
        """Test buffer untuk log awal sebelum UI terender"""
        from smartcash.ui.utils.ui_logger import log_to_ui, _EARLY_LOG_BUFFER, _UI_READY
        
        # Buat mock UI components tanpa status (UI belum terender)
        ui_components = {}
        
        # Log beberapa pesan sebelum UI terender
        log_to_ui(ui_components, "Pesan log awal 1", "info")
        log_to_ui(ui_components, "Pesan log awal 2", "warning")
        log_to_ui(ui_components, "Pesan log awal 3", "error")
        
        # Verifikasi pesan disimpan di buffer
        self.assertEqual(len(_EARLY_LOG_BUFFER), 3)
        self.assertEqual(_EARLY_LOG_BUFFER[0][0], "info")
        self.assertEqual(_EARLY_LOG_BUFFER[0][1], "Pesan log awal 1")
        self.assertEqual(_EARLY_LOG_BUFFER[1][0], "warning")
        self.assertEqual(_EARLY_LOG_BUFFER[1][1], "Pesan log awal 2")
        self.assertEqual(_EARLY_LOG_BUFFER[2][0], "error")
        self.assertEqual(_EARLY_LOG_BUFFER[2][1], "Pesan log awal 3")
        
        # Verifikasi UI belum siap
        self.assertFalse(_UI_READY)
        
        # Verifikasi pesan juga ditampilkan di console
        output = self.captured_output.getvalue()
        self.assertIn("[INFO] Pesan log awal 1", output)
        self.assertIn("[WARNING] Pesan log awal 2", output)
        self.assertIn("[ERROR] Pesan log awal 3", output)
    
    def test_flush_early_logs(self):
        """Test flush log awal setelah UI terender dengan mode test"""
        # Import modul yang akan diuji
        from smartcash.ui.utils.ui_logger import log_to_ui, flush_early_logs, _EARLY_LOG_BUFFER, _UI_READY
        
        # Reset variabel global
        _EARLY_LOG_BUFFER.clear()
        _UI_READY = False
        
        # Buat mock UI components tanpa status (UI belum terender)
        ui_components = {}
        
        # Log beberapa pesan sebelum UI terender
        log_to_ui(ui_components, "Pesan log awal 1", "info")
        log_to_ui(ui_components, "Pesan log awal 2", "warning")
        
        # Verifikasi pesan disimpan di buffer
        self.assertEqual(len(_EARLY_LOG_BUFFER), 2)
        
        # Buat mock UI components dengan status (UI sudah terender)
        ui_components = {
            'status': MagicMock()
        }
        ui_components['status'].clear_output = MagicMock()
        
        # Simpan salinan buffer sebelum flush
        buffer_before = _EARLY_LOG_BUFFER.copy()
        self.assertEqual(len(buffer_before), 2)
        
        # Gunakan test_mode=True untuk menghindari masalah dengan import
        flush_early_logs(ui_components, test_mode=True)
        
        # Verifikasi buffer dikosongkan
        self.assertEqual(len(_EARLY_LOG_BUFFER), 0)
        
        # Verifikasi UI sudah siap
        self.assertTrue(_UI_READY)
            
    def test_flush_early_logs_with_exception(self):
        """Test flush log awal dengan exception"""
        # Import modul yang akan diuji
        from smartcash.ui.utils.ui_logger import log_to_ui, flush_early_logs, _EARLY_LOG_BUFFER, _UI_READY
        
        # Reset variabel global
        _EARLY_LOG_BUFFER.clear()
        _UI_READY = False
        
        # Buat mock UI components tanpa status (UI belum terender)
        ui_components = {}
        
        # Log beberapa pesan sebelum UI terender
        log_to_ui(ui_components, "Pesan log awal 1", "info")
        log_to_ui(ui_components, "Pesan log awal 2", "warning")
        
        # Verifikasi pesan disimpan di buffer
        self.assertEqual(len(_EARLY_LOG_BUFFER), 2)
        
        # Buat mock UI components dengan status yang akan raise exception
        ui_components = {
            'status': MagicMock()
        }
        ui_components['status'].__enter__ = MagicMock(side_effect=Exception('Test exception'))
        ui_components['status'].__exit__ = MagicMock(return_value=None)
        ui_components['status'].clear_output = MagicMock()
        
        # Simpan salinan buffer sebelum flush
        buffer_before = _EARLY_LOG_BUFFER.copy()
        
        # Flush log awal dengan exception (tanpa test_mode)
        flush_early_logs(ui_components)
        
        # Verifikasi buffer dikembalikan karena terjadi exception
        self.assertEqual(len(_EARLY_LOG_BUFFER), 2)
        
        # Verifikasi UI belum siap karena terjadi exception
        self.assertFalse(_UI_READY)
    
    def test_intercept_stdout_to_ui(self):
        """Test intercept stdout dengan buffer log awal"""
        from smartcash.ui.utils.ui_logger import intercept_stdout_to_ui, _EARLY_LOG_BUFFER
        
        # Tambahkan beberapa log awal ke buffer
        _EARLY_LOG_BUFFER.append(("info", "Log dari stdout 1"))
        _EARLY_LOG_BUFFER.append(("warning", "Log dari stdout 2"))
        
        # Buat mock UI components dengan status (UI sudah terender)
        ui_components = {
            'status': MagicMock()
        }
        ui_components['status'].__enter__ = MagicMock(return_value=None)
        ui_components['status'].__exit__ = MagicMock(return_value=None)
        ui_components['status'].clear_output = MagicMock()
        
        # Mock fungsi flush_early_logs
        with patch('smartcash.ui.utils.ui_logger.flush_early_logs') as mock_flush:
            # Intercept stdout
            intercept_stdout_to_ui(ui_components)
            
            # Verifikasi flush_early_logs dipanggil
            mock_flush.assert_called_once_with(ui_components)

if __name__ == '__main__':
    unittest.main()
