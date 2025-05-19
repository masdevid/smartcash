"""
File: smartcash/ui/setup/env_config/tests/test_env_config_initializer.py
Deskripsi: Test untuk initializer konfigurasi environment
"""

import unittest
from unittest.mock import MagicMock, patch
from smartcash.ui.setup.env_config.tests.test_helper import WarningTestCase, ignore_layout_warnings

class TestEnvConfigInitializer(WarningTestCase):
    """Test case untuk env_config_initializer.py"""
    
    def test_initializer_import(self):
        """Test import initialize_env_config_ui berhasil"""
        from smartcash.ui.setup.env_config.env_config_initializer import initialize_env_config_ui
        self.assertTrue(callable(initialize_env_config_ui))
    
    def test_disable_ui_during_processing(self):
        """Test fungsi untuk menonaktifkan UI selama processing"""
        from smartcash.ui.setup.env_config.env_config_initializer import _disable_ui_during_processing
        
        # Buat mock UI components sederhana
        ui_components = {
            'drive_button': MagicMock(disabled=False),
            'directory_button': MagicMock(disabled=False)
        }
        
        # Test disable UI
        _disable_ui_during_processing(ui_components, True)
        
        # Verifikasi komponen dinonaktifkan
        self.assertTrue(ui_components['drive_button'].disabled)
        self.assertTrue(ui_components['directory_button'].disabled)
        
        # Test enable UI
        _disable_ui_during_processing(ui_components, False)
        
        # Verifikasi komponen diaktifkan kembali
        self.assertFalse(ui_components['drive_button'].disabled)
        self.assertFalse(ui_components['directory_button'].disabled)
    
    def test_cleanup_ui(self):
        """Test fungsi untuk membersihkan UI"""
        from smartcash.ui.setup.env_config.env_config_initializer import _cleanup_ui
        
        # Buat mock UI components sederhana
        ui_components = {
            'drive_button': MagicMock(disabled=True),
            'progress_bar': MagicMock(layout=MagicMock(visibility='visible')),
            'progress_message': MagicMock(layout=MagicMock(visibility='visible'))
        }
        
        # Panggil fungsi cleanup
        _cleanup_ui(ui_components)
        
        # Verifikasi tombol diaktifkan kembali
        self.assertFalse(ui_components['drive_button'].disabled)
        
        # Verifikasi progress bar dan message disembunyikan
        self.assertEqual(ui_components['progress_bar'].layout.visibility, 'hidden')
        self.assertEqual(ui_components['progress_message'].layout.visibility, 'hidden')

if __name__ == '__main__':
    unittest.main()
