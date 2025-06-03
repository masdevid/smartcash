"""
File: smartcash/ui/setup/dependency_installer/tests/test_dependency_installer_integration.py
Deskripsi: Test integrasi untuk dependency installer dengan one-liner style dan DRY approach
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any, List
import time

class TestDependencyInstallerIntegration(unittest.TestCase):
    """Test integrasi untuk dependency installer dengan one-liner style dan DRY approach."""
    
    def setUp(self):
        # Patch config and environment manager
        self.patcher_config = patch('smartcash.common.config.manager.get_config_manager', return_value=MagicMock())
        self.patcher_env = patch('smartcash.common.environment.get_environment_manager', return_value=MagicMock())
        self.mock_config = self.patcher_config.start()
        self.mock_env = self.patcher_env.start()
        
        # Patch subprocess untuk mencegah instalasi package sebenarnya
        self.patcher_subprocess = patch('smartcash.ui.setup.dependency_installer.handlers.package_handler.subprocess.run')
        self.mock_subprocess = self.patcher_subprocess.start()
        
        # Setup mock subprocess return value
        self.mock_subprocess.return_value = MagicMock(returncode=0)
        
        # Create mock observer manager
        self.mock_observer_manager = MagicMock()
        self.mock_observer_manager.notify = MagicMock()
        
        # Patch logger_helper.log_message di package_handler dan progress_helper
        self.patcher_log_message = patch('smartcash.ui.setup.dependency_installer.utils.logger_helper.log_message')
        self.mock_log_message = self.patcher_log_message.start()
        
        # Buat base UI components untuk digunakan di semua tes
        self.base_ui_components = self._create_base_ui_components()
    
    def tearDown(self):
        self.patcher_config.stop()
        self.patcher_env.stop()
        self.patcher_subprocess.stop()
        self.patcher_log_message.stop()
        
    def _create_base_ui_components(self) -> Dict[str, Any]:
        """Membuat base UI components yang digunakan di semua tes untuk mengurangi duplikasi"""
        return {
            'update_progress': MagicMock(),
            'log_message': MagicMock(),
            'update_status_panel': MagicMock(),
            'reset_progress_bar': MagicMock(),
            'show_for_operation': MagicMock(),
            'complete_operation': MagicMock(),
            'error_operation': MagicMock(),
            'observer_manager': self.mock_observer_manager
        }
        
    def _verify_common_calls(self, ui_components: Dict[str, Any], verify_status_panel: bool = True, 
                         verify_progress: bool = True, verify_log: bool = True):
        """Verifikasi common calls yang diharapkan dari semua handler"""
        if verify_progress:
            ui_components['update_progress'].assert_called()
        if verify_log:
            # Karena tidak semua handler sudah direfaktor, kita perlu memeriksa keduanya
            # Baik log_message dari ui_components maupun dari logger_helper
            log_called = ui_components['log_message'].called or self.mock_log_message.called
            self.assertTrue(log_called, "Tidak ada log_message yang dipanggil")
        if verify_status_panel:
            ui_components['update_status_panel'].assert_called()
    
    def test_package_handler_integration(self):
        """Test integrasi package_handler dengan one-liner style dan DRY approach."""
        from smartcash.ui.setup.dependency_installer.handlers.package_handler import get_all_missing_packages, run_batch_installation
        
        # Gunakan base UI components
        ui_components = self.base_ui_components.copy()
        ui_components['dependency_installer_initialized'] = True  # Pastikan is_initialized() mengembalikan True
        
        # Patch subprocess untuk mencegah instalasi package sebenarnya
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = MagicMock(returncode=0)
            
            # Test get_all_missing_packages
            with patch('smartcash.ui.setup.dependency_installer.utils.package_utils.get_installed_packages', return_value=[]), \
                 patch('smartcash.ui.setup.dependency_installer.utils.package_utils.get_package_groups', return_value={}), \
                 patch('smartcash.ui.setup.dependency_installer.utils.package_utils.check_missing_packages', return_value=[]):
                
                # Test get_all_missing_packages
                missing_packages = get_all_missing_packages(ui_components)
                self.assertIsInstance(missing_packages, list)
                
                # Verifikasi bahwa update_progress dipanggil
                ui_components['update_progress'].assert_called()
                ui_components['update_progress'].reset_mock()
            
            # Test run_batch_installation
            result = run_batch_installation(['test_package1', 'test_package2'], ui_components)
            
            # Verifikasi hasil run_batch_installation
            self.assertIsInstance(result, dict)
            expected_keys = ['total', 'success', 'failed', 'duration']
            for key in expected_keys:
                self.assertIn(key, result, f"Key '{key}' tidak ditemukan dalam result")
            
            # Verifikasi update_progress dipanggil
            ui_components['update_progress'].assert_called()
    
    def test_install_handler_integration(self):
        """Test integrasi install_handler dengan one-liner style dan DRY approach."""
        from smartcash.ui.setup.dependency_installer.handlers.install_handler import on_install_click
        
        # Gunakan base UI components dan tambahkan komponen khusus untuk install handler
        ui_components = self.base_ui_components.copy()
        ui_components.update({
            'install_button': widgets.Button(),
            'status': MagicMock(),
            'dependency_installer_initialized': True  # Pastikan is_initialized() mengembalikan True
        })
        
        # Patch package_handler.run_batch_installation dan analyze_installed_packages
        with patch('smartcash.ui.setup.dependency_installer.handlers.package_handler.run_batch_installation') as mock_run_batch, \
             patch('smartcash.ui.setup.dependency_installer.handlers.package_handler.get_all_missing_packages', return_value=['test_package1', 'test_package2']), \
             patch('smartcash.ui.setup.dependency_installer.utils.package_utils.analyze_installed_packages', return_value={'yolov5': True, 'smartcash': True, 'torch': False}):
            
            mock_run_batch.return_value = {'total': 2, 'success': 2, 'failed': 0, 'duration': 1.5, 'errors': []}
            
            # Test on_install_click
            on_install_click(None, ui_components)
            
            # Verifikasi bahwa run_batch_installation dipanggil
            mock_run_batch.assert_called_once()
            
            # Verifikasi bahwa update_status_panel dipanggil
            ui_components['update_status_panel'].assert_called()
            
            # Verifikasi bahwa update_progress dipanggil
            ui_components['update_progress'].assert_called()
    
    def test_analyzer_handler_integration(self):
        """Test integrasi analyzer_handler dengan one-liner style dan DRY approach."""
        from smartcash.ui.setup.dependency_installer.handlers.analyzer_handler import setup_analyzer_handler
        
        # Gunakan base UI components dan tambahkan komponen khusus untuk analyzer handler
        ui_components = self.base_ui_components.copy()
        ui_components.update({
            'analyze_button': widgets.Button(),
            'yolov5_req': MagicMock(),
            'yolov5_req_status': MagicMock(),
            'smartcash_req': MagicMock(),
            'smartcash_req_status': MagicMock(),
            'torch_req': MagicMock(),
            'torch_req_status': MagicMock(),
            'dependency_installer_initialized': True  # Pastikan is_initialized() mengembalikan True
        })
        
        # Mock analyze_installed_packages
        with patch('smartcash.ui.setup.dependency_installer.utils.package_utils.analyze_installed_packages', 
                  return_value={'yolov5': True, 'smartcash': True, 'torch': False}):
            
            # Setup analyzer handler
            setup_analyzer_handler(ui_components)
            
            # Verifikasi bahwa analyze_installed_packages function ditambahkan ke ui_components
            self.assertIn('analyze_installed_packages', ui_components)
            self.assertTrue(callable(ui_components['analyze_installed_packages']))
            
            # Test analyze_installed_packages function
            ui_components['analyze_installed_packages']()
            
            # Verifikasi bahwa update_progress dipanggil
            ui_components['update_progress'].assert_called()
            
            # Verifikasi update_status_panel dipanggil
            ui_components['update_status_panel'].assert_called()

if __name__ == '__main__':
    unittest.main()
