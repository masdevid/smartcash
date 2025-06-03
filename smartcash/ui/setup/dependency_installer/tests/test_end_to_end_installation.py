"""
File: smartcash/ui/setup/dependency_installer/tests/test_end_to_end_installation.py
Deskripsi: Test end-to-end untuk alur proses instalasi paket dan progress tracking
"""

import unittest
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from IPython.display import display
from typing import Dict, Any, List
import threading
import time

class TestEndToEndInstallation(unittest.TestCase):
    """Test end-to-end untuk alur proses instalasi paket dan progress tracking."""
    
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
            'progress_tracker': MagicMock(),
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
        
        # Simpan progress updates untuk verifikasi
        self.progress_updates = []
        self.ui_components['update_progress'].side_effect = self._track_progress_updates
        
        # Flag untuk menandai apakah progress tracking berjalan
        self.progress_tracking_running = False
    
    def _track_progress_updates(self, progress_type, value, message, color=None):
        """Fungsi untuk melacak progress updates."""
        self.progress_updates.append({
            'type': progress_type,
            'value': value,
            'message': message,
            'color': color
        })
    
    def _simulate_progress_updates(self, packages, ui_components, interval=0.1):
        """Fungsi untuk mensimulasikan progress updates selama instalasi."""
        self.progress_tracking_running = True
        total_packages = len(packages)
        
        for i, package in enumerate(packages):
            if not self.progress_tracking_running:
                break
                
            # Hitung progress
            progress = int(((i + 1) / total_packages) * 100)
            
            # Update progress
            message = f"Menginstall {package}... ({i+1}/{total_packages})"
            ui_components['update_progress']('overall', progress, message)
            
            # Simulasikan waktu instalasi
            time.sleep(interval)
        
        # Update progress selesai
        if self.progress_tracking_running:
            ui_components['update_progress']('step', 100, "Instalasi selesai", "#28a745")
    
    @patch('subprocess.Popen')
    @patch('smartcash.ui.setup.dependency_installer.utils.analyzer_utils.analyze_installed_packages')
    def test_end_to_end_installation_with_progress_tracking(self, mock_analyze_installed_packages, mock_popen):
        """Test end-to-end untuk alur proses instalasi paket dan progress tracking."""
        # Setup mock process
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ('Output', '')
        mock_popen.return_value = mock_process
        
        # Setup mock analyze_installed_packages
        mock_analyze_installed_packages.return_value = {
            'missing': ['test-package-1', 'test-package-2', 'test-package-3'],
            'upgrade': ['test-package-4', 'test-package-5'],
            'installed': ['test-package-6', 'test-package-7']
        }
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.analyzer_utils import analyze_installed_packages
        from smartcash.ui.setup.dependency_installer.utils.package_installer import install_required_packages
        
        # Tambahkan hasil analisis secara manual ke ui_components
        analysis_result = mock_analyze_installed_packages.return_value
        self.ui_components['analysis_result'] = analysis_result
        self.ui_components['analysis_categories'] = {
            'missing': analysis_result['missing'],
            'upgrade': analysis_result['upgrade'],
            'installed': analysis_result['installed']
        }
        
        # Verifikasi bahwa analisis berhasil
        self.assertIn('analysis_result', self.ui_components)
        self.assertIn('analysis_categories', self.ui_components)
        self.assertEqual(len(self.ui_components['analysis_categories']['missing']), 3)
        self.assertEqual(len(self.ui_components['analysis_categories']['upgrade']), 2)
        
        # Simulasikan progress updates dalam thread terpisah
        packages_to_install = self.ui_components['analysis_categories']['missing'] + self.ui_components['analysis_categories']['upgrade']
        progress_thread = threading.Thread(
            target=self._simulate_progress_updates,
            args=(packages_to_install, self.ui_components, 0.05)
        )
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Jalankan instalasi paket
            result = install_required_packages(self.ui_components)
            
            # Tunggu thread progress selesai
            self.progress_tracking_running = False
            progress_thread.join(timeout=1.0)
            
            # Verifikasi bahwa progress tracking berjalan
            self.assertGreater(len(self.progress_updates), 0)
            
            # Verifikasi bahwa progress meningkat dari waktu ke waktu
            if len(self.progress_updates) > 1:
                progress_values = [update['value'] for update in self.progress_updates if update['type'] == 'overall']
                for i in range(1, len(progress_values)):
                    self.assertGreaterEqual(progress_values[i], progress_values[i-1])
            
            # Verifikasi bahwa ada update progress dengan nilai 100 (selesai)
            completion_updates = [update for update in self.progress_updates if update['value'] == 100]
            self.assertGreater(len(completion_updates), 0)
            
            # Verifikasi bahwa log_message dipanggil
            self.assertTrue(self.ui_components['log_message'].called)
            
            # Verifikasi bahwa reset_progress_bar dipanggil
            self.assertTrue(self.ui_components['reset_progress_bar'].called)
            
            # Verifikasi bahwa update_status_panel dipanggil
            self.assertTrue(self.ui_components['update_status_panel'].called)
            
            # Verifikasi hasil
            self.assertTrue(result)
            
        finally:
            # Pastikan thread berhenti
            self.progress_tracking_running = False
    
    @patch('subprocess.Popen')
    def test_installation_with_error_handling(self, mock_popen):
        """Test instalasi dengan error handling."""
        # Setup mock process untuk gagal
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = ('', 'Error installing package')
        mock_popen.return_value = mock_process
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.package_installer import install_package
        
        # Panggil fungsi yang diuji
        result = install_package('test-package-error', self.ui_components)
        
        # Verifikasi bahwa log_message dipanggil dengan pesan error
        self.ui_components['log_message'].assert_called_with(
            '❌ Gagal menginstall test-package-error: Error installing package', 
            'error'
        )
        
        # Verifikasi hasil
        self.assertFalse(result)
    
    @patch('subprocess.Popen')
    def test_installation_with_exception_handling(self, mock_popen):
        """Test instalasi dengan exception handling."""
        # Setup mock process untuk raise exception
        mock_popen.side_effect = Exception("Unexpected error")
        
        # Import fungsi yang akan diuji
        from smartcash.ui.setup.dependency_installer.utils.package_installer import install_package
        
        # Panggil fungsi yang diuji
        result = install_package('test-package-exception', self.ui_components)
        
        # Verifikasi bahwa log_message dipanggil dengan pesan error
        self.ui_components['log_message'].assert_called_with(
            '❌ Error saat menginstall test-package-exception: Unexpected error', 
            'error'
        )
        
        # Verifikasi hasil
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
