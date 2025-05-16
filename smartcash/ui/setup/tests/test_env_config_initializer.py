"""
File: smartcash/ui/setup/tests/test_env_config_initializer.py
Deskripsi: Test untuk initializer konfigurasi environment
"""

import unittest
from unittest.mock import MagicMock, patch
from smartcash.ui.setup.tests.test_helper import WarningTestCase, ignore_layout_warnings

class TestEnvConfigInitializer(WarningTestCase):
    """Test case untuk env_config_initializer.py"""
    
    def test_disable_ui_during_processing(self):
        """Test fungsi untuk menonaktifkan UI selama processing"""
        from smartcash.ui.setup.env_config_initializer import _disable_ui_during_processing
        
        # Buat mock UI components
        ui_components = {
            'drive_button': MagicMock(disabled=False),
            'directory_button': MagicMock(disabled=False),
            'check_button': MagicMock(disabled=False),
            'save_button': MagicMock(disabled=False),
            'other_button': MagicMock(disabled=False)  # Tidak dalam daftar disable
        }
        
        # Test disable UI
        _disable_ui_during_processing(ui_components, True)
        
        # Verifikasi komponen yang seharusnya dinonaktifkan
        self.assertTrue(ui_components['drive_button'].disabled)
        self.assertTrue(ui_components['directory_button'].disabled)
        self.assertTrue(ui_components['check_button'].disabled)
        self.assertTrue(ui_components['save_button'].disabled)
        
        # Verifikasi komponen yang seharusnya tidak dinonaktifkan
        self.assertFalse(ui_components['other_button'].disabled)
        
        # Test enable UI
        _disable_ui_during_processing(ui_components, False)
        
        # Verifikasi semua komponen diaktifkan kembali
        self.assertFalse(ui_components['drive_button'].disabled)
        self.assertFalse(ui_components['directory_button'].disabled)
        self.assertFalse(ui_components['check_button'].disabled)
        self.assertFalse(ui_components['save_button'].disabled)
    
    def test_cleanup_ui(self):
        """Test fungsi untuk membersihkan UI"""
        from smartcash.ui.setup.env_config_initializer import _cleanup_ui
        
        # Buat mock UI components
        ui_components = {
            'drive_button': MagicMock(disabled=True),
            'directory_button': MagicMock(disabled=True),
            'progress_bar': MagicMock(layout=MagicMock(visibility='visible')),
            'progress_message': MagicMock(layout=MagicMock(visibility='visible'))
        }
        
        # Panggil fungsi cleanup
        _cleanup_ui(ui_components)
        
        # Verifikasi tombol diaktifkan kembali
        self.assertFalse(ui_components['drive_button'].disabled)
        self.assertFalse(ui_components['directory_button'].disabled)
        
        # Verifikasi progress bar dan message disembunyikan
        self.assertEqual(ui_components['progress_bar'].layout.visibility, 'hidden')
        self.assertEqual(ui_components['progress_message'].layout.visibility, 'hidden')
    
    def test_initialize_module_integration(self):
        """Test integrasi dengan initialize_module_ui"""
        # Import modul yang diperlukan
        from smartcash.ui.setup.env_config_initializer import _disable_ui_during_processing, _cleanup_ui
        
        # Setup mock untuk ui_components
        ui_components = {
            'drive_button': MagicMock(disabled=False),
            'directory_button': MagicMock(disabled=False),
            'check_button': MagicMock(disabled=False),
            'save_button': MagicMock(disabled=False),
            'progress_bar': MagicMock(layout=MagicMock(visibility='visible')),
            'progress_message': MagicMock(layout=MagicMock(visibility='visible'))
        }
        
        # Simulasi alur penggunaan fungsi helper dalam initialize_module_ui
        # 1. Disable UI selama proses
        _disable_ui_during_processing(ui_components, True)
        
        # Verifikasi UI dinonaktifkan
        self.assertTrue(ui_components['drive_button'].disabled)
        self.assertTrue(ui_components['directory_button'].disabled)
        self.assertTrue(ui_components['check_button'].disabled)
        self.assertTrue(ui_components['save_button'].disabled)
        
        # 2. Cleanup UI setelah proses selesai
        _cleanup_ui(ui_components)
        
        # Verifikasi UI diaktifkan kembali dan progress disembunyikan
        self.assertFalse(ui_components['drive_button'].disabled)
        self.assertFalse(ui_components['directory_button'].disabled)
        self.assertFalse(ui_components['check_button'].disabled)
        self.assertFalse(ui_components['save_button'].disabled)
        self.assertEqual(ui_components['progress_bar'].layout.visibility, 'hidden')
        self.assertEqual(ui_components['progress_message'].layout.visibility, 'hidden')

if __name__ == '__main__':
    unittest.main()
