"""
File: smartcash/ui/setup/tests/test_dependency_installer_progress.py
Deskripsi: Test untuk memverifikasi progress bar reporting pada instalasi dependencies
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import ipywidgets as widgets
from typing import Dict, Any

# Mock untuk create_status_indicator yang digunakan dalam run_batch_installation
class MockHTML:
    def __init__(self, html_content):
        self.html_content = html_content

# Mock untuk ALERT_STYLES dan COLORS
MOCK_ALERT_STYLES = {
    'info': {'icon': 'ℹ️', 'bg_color': '#e3f2fd', 'text_color': '#0d47a1', 'border_color': '#2196f3'},
    'success': {'icon': '✅', 'bg_color': '#e8f5e9', 'text_color': '#1b5e20', 'border_color': '#4caf50'},
    'warning': {'icon': '⚠️', 'bg_color': '#fff8e1', 'text_color': '#ff8f00', 'border_color': '#ffc107'},
    'error': {'icon': '❌', 'bg_color': '#ffebee', 'text_color': '#c62828', 'border_color': '#f44336'}
}

MOCK_COLORS = {
    'primary': '#2196f3',
    'secondary': '#607d8b',
    'success': '#4caf50',
    'warning': '#ff9800',
    'danger': '#f44336',
    'info': '#00bcd4',
    'light': '#f5f5f5',
    'dark': '#212121',
    'muted': '#757575',
    'border': '#e0e0e0',
    'background': '#ffffff'
}

# Patch untuk alert_utils.create_status_indicator
@patch('smartcash.ui.utils.alert_utils.create_status_indicator', return_value=MockHTML(''))
# Patch untuk constants
@patch('smartcash.ui.utils.constants.ALERT_STYLES', MOCK_ALERT_STYLES)
@patch('smartcash.ui.utils.constants.COLORS', MOCK_COLORS)
# Patch untuk display
@patch('IPython.display.display')
class TestDependencyInstallerProgress(unittest.TestCase):
    """Test untuk progress bar reporting pada dependency installer."""
    
    def setUp(self):
        """Setup untuk test."""
        # Buat mock UI components
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Installing:',
            layout=widgets.Layout(visibility='hidden')
        )
        
        self.progress_label = widgets.HTML(
            value="",
            layout=widgets.Layout(visibility='hidden')
        )
        
        self.status = widgets.Output()
        self.logger = MagicMock()
        
        # Buat ui_components dictionary
        self.ui_components = {
            'install_progress': self.progress_bar,
            'progress_label': self.progress_label,
            'status': self.status,
            'logger': self.logger
        }
    
    @patch('smartcash.ui.setup.package_installer.install_single_package')
    def test_progress_bar_visibility(self, mock_install, mock_display, mock_colors, mock_alert_styles, mock_status_indicator):
        """Test bahwa progress bar visibility diatur dengan benar selama instalasi."""
        from smartcash.ui.setup.package_installer import run_batch_installation
        
        # Setup mock untuk install_single_package
        mock_install.return_value = (True, "")
        
        # Pastikan progress bar tersembunyi sebelum instalasi
        self.assertEqual(self.progress_bar.layout.visibility, 'hidden')
        self.assertEqual(self.progress_label.layout.visibility, 'hidden')
        
        # Daftar package untuk diinstall
        packages = ['package1', 'package2', 'package3']
        
        # Jalankan instalasi
        success, stats = run_batch_installation(packages, self.ui_components)
        
        # Verifikasi bahwa progress bar terlihat setelah instalasi
        self.assertEqual(self.progress_bar.layout.visibility, 'visible')
        self.assertEqual(self.progress_label.layout.visibility, 'visible')
    
    @patch('smartcash.ui.setup.package_installer.install_single_package')
    def test_progress_bar_updates(self, mock_install, mock_display, mock_colors, mock_alert_styles, mock_status_indicator):
        """Test bahwa progress bar diperbarui dengan benar selama instalasi."""
        from smartcash.ui.setup.package_installer import run_batch_installation
        
        # Setup mock untuk install_single_package
        mock_install.return_value = (True, "")
        
        # Daftar package untuk diinstall
        packages = ['package1', 'package2', 'package3', 'package4', 'package5']
        
        # Jalankan instalasi
        success, stats = run_batch_installation(packages, self.ui_components)
        
        # Verifikasi bahwa progress bar diperbarui dengan benar
        self.assertEqual(self.progress_bar.value, 5)
        self.assertEqual(self.progress_bar.max, 5)
        
        # Verifikasi bahwa progress label diperbarui dengan benar
        self.assertIn("selesai", self.progress_label.value)
        
        # Verifikasi bahwa install_single_package dipanggil untuk setiap package
        self.assertEqual(mock_install.call_count, 5)
    
    @patch('smartcash.ui.setup.package_installer.install_single_package')
    def test_progress_reporting_with_errors(self, mock_install, mock_display, mock_colors, mock_alert_styles, mock_status_indicator):
        """Test progress reporting saat ada error instalasi."""
        from smartcash.ui.setup.package_installer import run_batch_installation
        
        # Setup mock untuk install_single_package
        # Package 1 dan 3 berhasil, package 2 gagal
        mock_install.side_effect = [
            (True, ""),
            (False, "Error installing package2"),
            (True, "")
        ]
        
        # Daftar package untuk diinstall
        packages = ['package1', 'package2', 'package3']
        
        # Jalankan instalasi
        success, stats = run_batch_installation(packages, self.ui_components)
        
        # Verifikasi bahwa progress bar diperbarui dengan benar
        self.assertEqual(self.progress_bar.value, 3)
        self.assertEqual(self.progress_bar.max, 3)
        
        # Verifikasi bahwa stats diperbarui dengan benar
        self.assertEqual(stats['success'], 2)
        self.assertEqual(stats['failed'], 1)
        self.assertEqual(len(stats['errors']), 1)
        
        # Verifikasi bahwa progress label diperbarui dengan benar
        self.assertIn("selesai", self.progress_label.value)
    
    @patch('smartcash.ui.setup.package_installer.install_single_package')
    def test_tqdm_package_skipped(self, mock_install, mock_display, mock_colors, mock_alert_styles, mock_status_indicator):
        """Test bahwa package tqdm dilewati sesuai konfigurasi."""
        from smartcash.ui.setup.package_installer import run_batch_installation
        
        # Setup mock untuk install_single_package (tidak seharusnya dipanggil untuk tqdm)
        mock_install.return_value = (True, "")
        
        # Daftar package untuk diinstall termasuk tqdm
        packages = ['package1', 'tqdm', 'package2']
        
        # Jalankan instalasi
        success, stats = run_batch_installation(packages, self.ui_components)
        
        # Verifikasi bahwa tqdm dilewati (mock_install hanya dipanggil 2 kali)
        self.assertEqual(mock_install.call_count, 2)
        
        # Verifikasi bahwa stats mencatat package yang dilewati
        self.assertEqual(stats['skipped'], 1)
        
        # Verifikasi bahwa progress bar diperbarui dengan benar
        self.assertEqual(self.progress_bar.value, 2)
        self.assertEqual(self.progress_bar.max, 2)

if __name__ == '__main__':
    unittest.main()
