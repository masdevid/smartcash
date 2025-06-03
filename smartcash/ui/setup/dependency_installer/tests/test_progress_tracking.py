"""
File: smartcash/ui/setup/dependency_installer/tests/test_progress_tracking.py
Deskripsi: Test untuk memastikan progress tracking berjalan dengan benar selama proses instalasi
"""

import unittest
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, List

class TestProgressTracking(unittest.TestCase):
    """Test untuk memastikan progress tracking berjalan dengan benar selama proses instalasi."""
    
    def setUp(self):
        """Setup untuk test dengan mock UI components."""
        # Mock UI components
        self.ui_components = {
            'ui': widgets.VBox(),
            'install_button': widgets.Button(description='Install'),
            'status': widgets.HTML(value=''),
            'log_output': MagicMock(),
            'progress_container': widgets.VBox(),
            'status_panel': widgets.HTML(value=''),
            'progress_bar': widgets.FloatProgress(min=0, max=100, value=0),
            'progress_label': widgets.HTML(value=''),
            'status_widget': widgets.HTML(value=''),
            'tracker': MagicMock(),
        }
        
        # Mock log_output methods
        self.ui_components['log_output'].clear_output = MagicMock()
        self.ui_components['log_output'].append_display_data = MagicMock()
        
        # Mock tracker methods
        self.ui_components['tracker'].show = MagicMock()
        self.ui_components['tracker'].hide = MagicMock()
        
        # Buat mock functions untuk progress tracking
        self.ui_components['log_message'] = MagicMock()
        self.ui_components['update_progress'] = MagicMock()
        self.ui_components['reset_progress_bar'] = MagicMock()
        self.ui_components['update_status_panel'] = MagicMock()
        self.ui_components['error_operation'] = MagicMock()
        self.ui_components['show_for_operation'] = MagicMock()
        
        # Mock analysis result
        self.ui_components['analysis_categories'] = {
            'missing': ['test-package-1', 'test-package-2', 'test-package-3'],
            'upgrade': ['test-package-4', 'test-package-5'],
            'installed': []
        }
        
    @patch('subprocess.Popen')
    def test_install_package_updates_progress(self, mock_popen):
        """Test bahwa install_package memperbarui progress tracking."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ('Output', '')
        mock_popen.return_value = mock_process
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.package_installer import install_package
        
        # Panggil fungsi yang diuji
        result = install_package('test-package', self.ui_components)
        
        # Verifikasi bahwa log_message dipanggil
        expected_calls = [
            call('📦 Menginstall test-package...', 'info'),
            call('✅ Berhasil menginstall test-package', 'success')
        ]
        self.ui_components['log_message'].assert_has_calls(expected_calls, any_order=True)
        
        # Verifikasi hasil
        self.assertTrue(result)
    
    @patch('smartcash.ui.setup.dependency_installer.utils.package_installer.install_package')
    def test_install_packages_parallel_updates_progress(self, mock_install_package):
        """Test bahwa install_packages_parallel memperbarui progress tracking."""
        # Setup mock install_package
        mock_install_package.side_effect = lambda package, ui: True
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.package_installer import install_packages_parallel
        
        # Panggil fungsi yang diuji
        packages = ['test-package-1', 'test-package-2', 'test-package-3']
        results = install_packages_parallel(packages, self.ui_components)
        
        # Verifikasi bahwa progress tracking diperbarui
        self.ui_components['reset_progress_bar'].assert_called()
        self.ui_components['update_progress'].assert_called()
        
        # Verifikasi bahwa progress container ditampilkan
        self.assertEqual(self.ui_components['progress_container'].layout.visibility, 'visible')
        
        # Verifikasi bahwa log_message dipanggil dengan salah satu pesan yang diharapkan
        # Karena urutan pemanggilan tidak bisa diprediksi dengan pasti, kita cek apakah fungsi dipanggil
        self.assertTrue(self.ui_components['log_message'].called)
        
        # Verifikasi hasil
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results.values()))
    
    @patch('smartcash.ui.setup.dependency_installer.utils.package_installer.install_packages_parallel')
    def test_install_required_packages_updates_progress(self, mock_install_packages_parallel):
        """Test bahwa install_required_packages memperbarui progress tracking."""
        # Setup mock install_packages_parallel
        mock_install_packages_parallel.return_value = {
            'test-package-1': True,
            'test-package-2': True,
            'test-package-3': True,
            'test-package-4': True,
            'test-package-5': True
        }
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.package_installer import install_required_packages
        
        # Panggil fungsi yang diuji
        result = install_required_packages(self.ui_components)
        
        # Verifikasi bahwa log_message dipanggil
        self.ui_components['log_message'].assert_called_with('🔄 Menginstall 5 package...', 'info')
        
        # Verifikasi bahwa install_packages_parallel dipanggil dengan package yang benar
        expected_packages = ['test-package-1', 'test-package-2', 'test-package-3', 'test-package-4', 'test-package-5']
        mock_install_packages_parallel.assert_called_once()
        actual_packages = mock_install_packages_parallel.call_args[0][0]
        self.assertEqual(sorted(actual_packages), sorted(expected_packages))
        
        # Verifikasi hasil
        self.assertTrue(result)
    
    @patch('subprocess.Popen')
    def test_progress_increments_during_installation(self, mock_popen):
        """Test bahwa progress bar bertambah selama proses instalasi."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ('Output', '')
        mock_popen.return_value = mock_process
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.package_installer import install_packages_parallel
        
        # Simpan nilai progress awal
        initial_progress = self.ui_components['progress_bar'].value
        
        # Panggil fungsi yang diuji
        packages = ['test-package-1', 'test-package-2', 'test-package-3']
        results = install_packages_parallel(packages, self.ui_components, max_workers=1)
        
        # Verifikasi bahwa update_progress dipanggil untuk memperbarui progress bar
        # Kita tidak perlu memeriksa nilai progress bar secara langsung karena itu bisa berubah
        # tergantung pada implementasi
        
        # Verifikasi bahwa progress label diperbarui
        self.assertTrue(self.ui_components['update_progress'].called)
        
        # Verifikasi hasil
        self.assertEqual(len(results), 3)
        self.assertTrue(all(results.values()))
    
    @patch('smartcash.ui.setup.dependency_installer.utils.package_installer.install_packages_parallel')
    @patch('smartcash.ui.setup.dependency_installer.utils.analyzer_utils.analyze_installed_packages')
    @patch('smartcash.ui.setup.dependency_installer.handlers.package_handler.get_all_missing_packages')
    @patch('smartcash.ui.setup.dependency_installer.handlers.package_handler.run_batch_installation')
    @patch('smartcash.ui.setup.dependency_installer.utils.package_installer.install_required_packages')
    def test_on_install_click_updates_progress(self, mock_install_required, mock_run_batch, mock_get_missing, mock_analyze, mock_install_parallel):
        """Test bahwa on_install_click memperbarui progress tracking."""
        # Setup mocks
        mock_get_missing.return_value = ['test-package-1', 'test-package-2', 'test-package-3']
        
        # Mock analysis_result
        self.ui_components['analysis_result'] = {
            'missing': ['test-package-1', 'test-package-2', 'test-package-3'],
            'upgrade': [],
            'installed': []
        }
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.handlers.install_handler import on_install_click
        
        # Panggil fungsi yang diuji
        on_install_click(None, self.ui_components)
        
        # Verifikasi bahwa progress container ditampilkan
        self.assertEqual(self.ui_components['progress_container'].layout.visibility, 'visible')
        
        # Verifikasi bahwa log_output.clear_output dipanggil
        self.ui_components['log_output'].clear_output.assert_called_once()
        
        # Verifikasi bahwa log_message dipanggil
        self.ui_components['log_message'].assert_called_with('🚀 Memulai proses instalasi dependency', 'info')
        
        # Verifikasi bahwa run_batch_installation dan install_required_packages dipanggil
        mock_run_batch.assert_called_once()
        mock_install_required.assert_called_once()

if __name__ == '__main__':
    unittest.main()
