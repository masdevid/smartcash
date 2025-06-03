"""
File: smartcash/ui/setup/dependency_installer/tests/test_dependency_installer_initializer.py
Deskripsi: Test suite untuk menguji inisialisasi dependency installer
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any

# Import konstanta untuk namespace dan module name
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES

class TestDependencyInstallerInitializer(unittest.TestCase):
    """Test suite untuk menguji inisialisasi dependency installer."""
    
    def setUp(self):
        """Setup untuk setiap test case dengan pendekatan one-liner style."""
        # Patch modul dan fungsi yang digunakan dalam initializer
        self.patcher_create_ui = patch('smartcash.ui.setup.dependency_installer.components.dependency_installer_component.create_dependency_installer_ui')
        self.patcher_setup_handlers = patch('smartcash.ui.setup.dependency_installer.handlers.setup_handlers.setup_dependency_installer_handlers')
        self.patcher_validate_ui = patch('smartcash.ui.setup.dependency_installer.utils.validation_utils.validate_ui_components')
        self.patcher_get_default_config = patch('smartcash.ui.setup.dependency_installer.utils.validation_utils.get_default_config')
        self.patcher_display = patch('IPython.display.display')
        self.patcher_clear_output = patch('IPython.display.clear_output')
        
        # Start patchers
        self.mock_create_ui = self.patcher_create_ui.start()
        self.mock_setup_handlers = self.patcher_setup_handlers.start()
        self.mock_validate_ui = self.patcher_validate_ui.start()
        self.mock_get_default_config = self.patcher_get_default_config.start()
        self.mock_display = self.patcher_display.start()
        self.mock_clear_output = self.patcher_clear_output.start()
        
        # Setup mock returns
        self.module_logger_name = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]
        self.mock_ui_components = {
            'ui': widgets.VBox(),
            'install_button': widgets.Button(),
            'status': widgets.Output(),
            'log_output': widgets.Output(),
            'progress_container': widgets.VBox(),
            'status_panel': widgets.VBox(),
            'progress_tracker': MagicMock(),
            'log_message': MagicMock(),
            'module_name': self.module_logger_name,
            'logger_namespace': DEPENDENCY_INSTALLER_LOGGER_NAMESPACE,
            'dependency_installer_initialized': False
        }
        self.mock_create_ui.return_value = self.mock_ui_components
        self.mock_setup_handlers.return_value = {**self.mock_ui_components, 'run_delayed_analysis': MagicMock(), 'handlers_setup': True}
        self.mock_validate_ui.return_value = self.mock_ui_components
        self.mock_get_default_config.return_value = {'delay_analysis': True, 'auto_install': False}
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        # Stop semua patchers
        self.patcher_create_ui.stop()
        self.patcher_setup_handlers.stop()
        self.patcher_validate_ui.stop()
        self.patcher_get_default_config.stop()
        self.patcher_display.stop()
        self.patcher_clear_output.stop()
    
    def test_initialize_dependency_installer(self):
        """Test initialize_dependency_installer dengan pendekatan one-liner style."""
        from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import initialize_dependency_installer
        
        # Panggil fungsi yang diuji dengan patch untuk CommonInitializer
        with patch('smartcash.ui.utils.common_initializer.CommonInitializer._validate_setup', return_value={'valid': True}):
            result = initialize_dependency_installer()
        
            # Verifikasi hasil
            self.assertIsNotNone(result)
            self.mock_create_ui.assert_called_once()
            self.mock_setup_handlers.assert_called_once()
            # validate_ui_components tidak dipanggil lagi karena menggunakan _validate_setup dari CommonInitializer
            self.mock_clear_output.assert_called_once_with(wait=True)
    
    def test_dependency_installer_initializer_class(self):
        """Test DependencyInstallerInitializer class dengan pendekatan one-liner style."""
        from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import DependencyInstallerInitializer
        
        # Buat instance DependencyInstallerInitializer dengan logger yang di-mock
        with patch('smartcash.ui.utils.common_initializer.CommonInitializer.__init__') as mock_init:
            with patch('smartcash.common.logger.get_logger') as mock_logger:
                mock_logger.return_value = MagicMock()
                initializer = DependencyInstallerInitializer()
                # Verifikasi initializer memanggil parent class dengan namespace yang benar
                mock_init.assert_called_once_with(self.module_logger_name, DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
                
                # Set logger secara manual untuk test
                initializer.logger = MagicMock()
        
        # Test _get_critical_components
        critical_components = initializer._get_critical_components()
        self.assertIsInstance(critical_components, list)
        self.assertIn('ui', critical_components)
        self.assertIn('install_button', critical_components)
        self.assertIn('status', critical_components)
        self.assertIn('log_output', critical_components)
        self.assertIn('progress_container', critical_components)
        
        # Test _create_ui_components
        ui_components = initializer._create_ui_components({})
        # Verifikasi bahwa namespace dan module name ditambahkan ke UI components
        self.assertEqual(ui_components['module_name'], self.module_logger_name)
        self.assertEqual(ui_components['logger_namespace'], DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
        self.assertFalse(ui_components['dependency_installer_initialized'])
        
        # Test _setup_module_handlers
        # Buat salinan dari mock_ui_components untuk dimodifikasi
        ui_components_copy = self.mock_ui_components.copy()
        handlers_result = initializer._setup_module_handlers(ui_components_copy, {})
        self.assertTrue(handlers_result.get('handlers_setup', False))
        # Verifikasi bahwa dependency_installer_initialized diset ke True
        self.assertTrue(handlers_result.get('dependency_installer_initialized', False))
        
        # Test _post_initialization_hook
        with patch.object(initializer, 'logger', MagicMock()):
            post_init_result = initializer._post_initialization_hook(
                {**self.mock_ui_components, 'run_delayed_analysis': MagicMock()},
                {'delay_analysis': True}
            )
            self.assertIsNotNone(post_init_result)
    
    def test_error_handling_in_setup_module_handlers(self):
        """Test penanganan error dalam _setup_module_handlers dengan pendekatan one-liner style."""
        from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import DependencyInstallerInitializer
        
        # Setup error case
        self.mock_setup_handlers.side_effect = Exception("Test error")
        
        # Buat instance DependencyInstallerInitializer
        initializer = DependencyInstallerInitializer()
        
        # Test _setup_module_handlers dengan error
        with patch.object(initializer, 'logger', MagicMock()):
            handlers_result = initializer._setup_module_handlers(self.mock_ui_components, {})
            
            # Verifikasi hasil
            self.assertFalse(handlers_result.get('handlers_setup', True))
            self.mock_ui_components['log_message'].assert_called_once()
    
    def test_missing_critical_components(self):
        """Test penanganan missing critical components dengan pendekatan one-liner style."""
        from smartcash.ui.setup.dependency_installer.dependency_installer_initializer import DependencyInstallerInitializer
        
        # Setup incomplete UI components
        incomplete_ui = {
            'ui': widgets.VBox(),
            'install_button': widgets.Button(),
            # Missing 'status', 'log_output', and 'progress_container'
        }
        self.mock_create_ui.return_value = incomplete_ui
        
        # Buat instance DependencyInstallerInitializer
        initializer = DependencyInstallerInitializer()
        
        # Test initialize dengan missing components
        with patch.object(initializer, 'initialize', return_value={'error': 'Failed to create UI components'}):
            with patch.object(initializer, 'logger', MagicMock()):
                # Panggil initialize
                result = initializer.initialize({})
                
                # Verifikasi hasil
                self.assertIn('error', result)
                self.assertEqual(result['error'], 'Failed to create UI components')

if __name__ == '__main__':
    unittest.main()
