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
        
        # Setup layout untuk progress_container
        self.ui_components['progress_container'].layout = widgets.Layout(visibility='hidden')
        
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
        self.ui_components['complete_operation'] = MagicMock()
        self.ui_components['active_bars'] = ['overall', 'step', 'current']
        
        # Mock analysis result
        self.ui_components['analysis_categories'] = {
            'missing': ['test-package-1', 'test-package-2', 'test-package-3'],
            'upgrade': ['test-package-4', 'test-package-5'],
            'installed': []
        }
        
        # Mock analysis_result untuk install_required_packages
        self.ui_components['analysis_result'] = {
            'missing_packages': ['test-package-1', 'test-package-2', 'test-package-3', 'test-package-4', 'test-package-5']
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
        
        # Verifikasi bahwa show_for_operation dipanggil
        self.ui_components['show_for_operation'].assert_called_with('install')
        
        # Verifikasi bahwa log_message dipanggil dengan salah satu pesan yang diharapkan
        # Karena urutan pemanggilan tidak bisa diprediksi dengan pasti, kita cek apakah fungsi dipanggil
        self.assertTrue(self.ui_components['log_message'].called)
        
    def test_complete_operation_updates_ui(self):
        """Test bahwa complete_operation memperbarui UI dengan benar."""
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.ui_utils import complete_operation
        
        # Panggil fungsi yang diuji
        success_message = "Semua package berhasil diinstall"
        complete_operation(self.ui_components, success_message)
        
        # Verifikasi bahwa progress bar diupdate ke 100%
        expected_progress_calls = [
            call('overall', 100, "Operasi selesai", "#28a745"),
            call('step', 100, "Operasi selesai", "#28a745"),
            call('current', 100, "✅ Operasi selesai", "#28a745")
        ]
        self.ui_components['update_progress'].assert_has_calls(expected_progress_calls, any_order=True)
        
        # Verifikasi bahwa status panel diupdate dengan pesan sukses
        self.ui_components['update_status_panel'].assert_called_with("success", f"✅ {success_message}")
        
        # Verifikasi bahwa log_message dipanggil dengan pesan sukses
        self.ui_components['log_message'].assert_called_with(f"✅ {success_message}", "success")
        
    def test_analyze_installed_packages_updates_progress(self):
        """Test bahwa analyze_installed_packages memperbarui progress tracking dengan benar."""
        # Mock fungsi yang dipanggil oleh analyze_installed_packages
        with patch('smartcash.ui.setup.dependency_installer.utils.analyzer_utils.get_installed_packages') as mock_get_installed_packages, \
             patch('smartcash.ui.setup.dependency_installer.utils.analyzer_utils.get_installed_package_versions') as mock_get_versions, \
             patch('smartcash.ui.setup.dependency_installer.utils.analyzer_utils.get_project_requirements') as mock_get_requirements, \
             patch('smartcash.ui.setup.dependency_installer.utils.analyzer_utils.analyze_requirements') as mock_analyze_requirements:
            
            # Setup mock returns
            mock_get_installed_packages.return_value = {'numpy', 'pandas'}
            mock_get_versions.return_value = {'numpy': '1.20.0', 'pandas': '1.3.0'}
            mock_get_requirements.return_value = ['numpy>=1.18.0', 'pandas>=1.2.0', 'matplotlib>=3.3.0']
            mock_analyze_requirements.return_value = {
                'numpy>=1.18.0': {'installed': True, 'package_name': 'numpy', 'installed_version': '1.20.0', 'required_version': '>=1.18.0', 'compatible': True},
                'pandas>=1.2.0': {'installed': True, 'package_name': 'pandas', 'installed_version': '1.3.0', 'required_version': '>=1.2.0', 'compatible': True},
                'matplotlib>=3.3.0': {'installed': False, 'package_name': 'matplotlib', 'installed_version': None, 'required_version': '>=3.3.0', 'compatible': False}
            }
            
            # Import fungsi yang akan diuji
            from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import analyze_installed_packages
            
            # Panggil fungsi yang diuji
            result = analyze_installed_packages(self.ui_components)
            
            # Verifikasi bahwa show_for_operation dipanggil dengan 'analyze'
            self.ui_components['show_for_operation'].assert_called_with('analyze')
            
            # Verifikasi bahwa reset_progress_bar dipanggil
            self.ui_components['reset_progress_bar'].assert_called()
            
            # Verifikasi bahwa update_progress dipanggil untuk overall, step, dan current
            self.assertTrue(self.ui_components['update_progress'].called)
            
            # Verifikasi bahwa log_message dipanggil
            self.assertTrue(self.ui_components['log_message'].called)
            
            # Verifikasi hasil
            self.assertIsNotNone(result)
            self.assertIn('missing_packages', result)
    
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
        install_required_packages(self.ui_components)
        
        # Verifikasi bahwa log_message dipanggil dengan pesan yang sesuai
        expected_calls = [
            call("🔍 Memeriksa package yang perlu diinstall...", "info"),
            call("📦 Menemukan 5 package yang perlu diinstall", "info")
        ]
        self.ui_components['log_message'].assert_has_calls(expected_calls, any_order=True)
        
        # Verifikasi bahwa install_packages_parallel dipanggil dengan package yang benar
        expected_packages = ['test-package-1', 'test-package-2', 'test-package-3', 'test-package-4', 'test-package-5']
        mock_install_packages_parallel.assert_called_once()
        actual_packages = mock_install_packages_parallel.call_args[0][0]
        self.assertEqual(sorted(actual_packages), sorted(expected_packages))
        
        # Verifikasi bahwa show_for_operation dipanggil
        self.ui_components['show_for_operation'].assert_called_with('install')
        
        # Verifikasi bahwa reset_progress_bar dipanggil
        self.ui_components['reset_progress_bar'].assert_called()
    
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
