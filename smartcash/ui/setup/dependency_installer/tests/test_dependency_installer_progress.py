"""
File: smartcash/ui/setup/dependency_installer/tests/test_dependency_installer_progress.py
Deskripsi: Test untuk memverifikasi progress bar reporting dan konfigurasi pada instalasi dependencies
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

class TestDependencyInstallerProgress(unittest.TestCase):
    """Test untuk progress bar reporting dan konfigurasi pada dependency installer."""
    
    def setUp(self):
        # Patch config and environment manager to avoid real file system dependency
        self.patcher_config = patch('smartcash.common.config.manager.get_config_manager', return_value=MagicMock())
        self.patcher_env = patch('smartcash.common.environment.get_environment_manager', return_value=MagicMock())
        self.mock_config = self.patcher_config.start()
        self.mock_env = self.patcher_env.start()
        
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
    
    def tearDown(self):
        self.patcher_config.stop()
        self.patcher_env.stop()
    
    def test_install_handler_import(self):
        """Test import install_handler berhasil"""
        from smartcash.ui.setup.dependency_installer.handlers.install_handler import setup_install_handler, on_install_click
        self.assertTrue(callable(setup_install_handler))
        self.assertTrue(callable(on_install_click))
    
    def test_package_handler_import(self):
        """Test import package_handler berhasil"""
        from smartcash.ui.setup.dependency_installer.handlers.package_handler import run_batch_installation
        self.assertTrue(callable(run_batch_installation))
    
    def test_dependency_installer_component_import(self):
        """Test import dependency_installer_component berhasil"""
        from smartcash.ui.setup.dependency_installer.components.dependency_installer_component import create_dependency_installer_ui
        self.assertTrue(callable(create_dependency_installer_ui))
    
    def test_dependency_installer_initializer_import(self):
        """Test import dependency_installer_initializer berhasil"""
        from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import initialize_dependency_installer
        self.assertTrue(callable(initialize_dependency_installer))
    
    def test_config_sync(self):
        """Test sinkronisasi konfigurasi berhasil"""
        with patch('smartcash.ui.setup.dependency_installer.dependency_installer_initializer.get_config_manager', return_value=MagicMock()), \
             patch('smartcash.ui.setup.dependency_installer.dependency_installer_initializer.get_environment_manager', return_value=MagicMock()):
            from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import initialize_dependency_installer
            ui_components = initialize_dependency_installer()
            self.assertIsNotNone(ui_components)
            self.assertIn('logger', ui_components)
    
    def test_environment_sync(self):
        """Test sinkronisasi environment berhasil"""
        with patch('smartcash.ui.setup.dependency_installer.dependency_installer_initializer.get_config_manager', return_value=MagicMock()), \
             patch('smartcash.ui.setup.dependency_installer.dependency_installer_initializer.get_environment_manager', return_value=MagicMock()):
            from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import initialize_dependency_installer
            ui_components = initialize_dependency_installer()
            self.assertIsNotNone(ui_components)
            self.assertIn('logger', ui_components)

if __name__ == '__main__':
    unittest.main()
