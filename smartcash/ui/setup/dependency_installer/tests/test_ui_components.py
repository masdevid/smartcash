"""
File: smartcash/ui/setup/dependency_installer/tests/test_ui_components.py
Deskripsi: Test suite untuk menguji UI components dependency installer
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any

# Import konstanta untuk namespace dan module name
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES

class TestDependencyInstallerUIComponents(unittest.TestCase):
    """Test suite untuk menguji UI components dependency installer."""
    
    def setUp(self):
        """Setup untuk setiap test case dengan pendekatan one-liner style."""
        # Patch modul dan fungsi yang digunakan dalam UI components
        self.patcher_create_header = patch('smartcash.ui.utils.header_utils.create_header')
        self.patcher_get_dependencies_info = patch('smartcash.ui.info_boxes.dependencies_info.get_dependencies_info')
        self.patcher_get_package_categories = patch('smartcash.ui.setup.dependency_installer.utils.package_utils.get_package_categories')
        self.patcher_create_progress_tracking = patch('smartcash.ui.components.progress_tracking.create_progress_tracking_container')
        self.patcher_create_log_accordion = patch('smartcash.ui.components.log_accordion.create_log_accordion')
        
        # Start patchers
        self.mock_create_header = self.patcher_create_header.start()
        self.mock_get_dependencies_info = self.patcher_get_dependencies_info.start()
        self.mock_get_package_categories = self.patcher_get_package_categories.start()
        self.mock_create_progress_tracking = self.patcher_create_progress_tracking.start()
        self.mock_create_log_accordion = self.patcher_create_log_accordion.start()
        
        # Setup mock returns
        self.mock_create_header.return_value = widgets.HTML("Mock Header")
        self.mock_get_dependencies_info.return_value = widgets.HTML("Mock Info")
        self.mock_get_package_categories.return_value = [
            {
                'name': 'Test Category',
                'key': 'test_category',
                'description': 'Test Description',
                'icon': '🔧',
                'packages': [
                    {
                        'name': 'Test Package',
                        'key': 'test_package',
                        'description': 'Test Package Description',
                        'icon': '📦',
                        'default': True
                    }
                ]
            }
        ]
        self.mock_create_progress_tracking.return_value = {
            'container': widgets.VBox(),
            'tracker': MagicMock(),
            'update_progress': MagicMock(),
            'show_for_operation': MagicMock()
        }
        self.mock_create_log_accordion.return_value = {
            'log_output': widgets.Output(),
            'log_accordion': widgets.Accordion()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        # Stop semua patchers
        self.patcher_create_header.stop()
        self.patcher_get_dependencies_info.stop()
        self.patcher_get_package_categories.stop()
        self.patcher_create_progress_tracking.stop()
        self.patcher_create_log_accordion.stop()
    
    def test_create_dependency_installer_ui(self):
        """Test create_dependency_installer_ui dengan pendekatan one-liner style."""
        from smartcash.ui.setup.dependency_installer.components.dependency_installer_component import create_dependency_installer_ui
        
        # Panggil fungsi yang diuji
        ui_components = create_dependency_installer_ui()
        
        # Verifikasi hasil
        self.assertIsNotNone(ui_components)
        self.assertIn('ui', ui_components)
        self.assertIn('status', ui_components)
        self.assertIn('log_output', ui_components)
        self.assertIn('status_panel', ui_components)
        self.assertIn('install_button', ui_components)
        self.assertIn('custom_packages', ui_components)
        self.assertIn('progress_tracker', ui_components)
        
        # Verifikasi bahwa module_name dan logger_namespace ditambahkan oleh initializer
        # Catatan: Ini tidak diverifikasi di sini karena ditambahkan oleh DependencyInstallerInitializer,
        # bukan oleh create_dependency_installer_ui
    
    def test_assemble_ui_components(self):
        """Test assemble_ui_components dengan pendekatan one-liner style."""
        from smartcash.ui.setup.dependency_installer.components.ui_components import assemble_ui_components
        
        # Panggil fungsi yang diuji
        ui_components = assemble_ui_components()
        
        # Verifikasi hasil
        self.assertIsNotNone(ui_components)
        self.assertIn('main_container', ui_components)
        self.assertIn('header', ui_components)
        self.assertIn('log_output', ui_components)
        self.assertIn('install_button', ui_components)
        self.assertIn('analyze_button', ui_components)
        self.assertIn('reset_button', ui_components)
        self.assertIn('progress_container', ui_components)
        self.assertIn('progress_bar', ui_components)
        self.assertIn('progress_label', ui_components)
        self.assertIn('status_container', ui_components)
        self.assertIn('status_widget', ui_components)
        self.assertIn('error_widget', ui_components)
    
    def test_create_category_box(self):
        """Test create_category_box dengan pendekatan one-liner style."""
        from smartcash.ui.setup.dependency_installer.components.dependency_installer_component import create_category_box
        
        # Setup test data
        category = {
            'name': 'Test Category',
            'key': 'test_category',
            'description': 'Test Description',
            'icon': '🔧',
            'packages': [
                {
                    'name': 'Test Package',
                    'key': 'test_package',
                    'description': 'Test Package Description',
                    'icon': '📦',
                    'default': True
                }
            ]
        }
        checkboxes = {}
        
        # Panggil fungsi yang diuji
        category_box = create_category_box(category, checkboxes)
        
        # Verifikasi hasil
        self.assertIsInstance(category_box, widgets.VBox)
        self.assertEqual(len(checkboxes), 2)  # checkbox dan status widget
        self.assertIn('test_package', checkboxes)
        self.assertIn('test_package_status', checkboxes)
        self.assertIsInstance(checkboxes['test_package'], widgets.Checkbox)
        self.assertIsInstance(checkboxes['test_package_status'], widgets.HTML)

if __name__ == '__main__':
    unittest.main()
