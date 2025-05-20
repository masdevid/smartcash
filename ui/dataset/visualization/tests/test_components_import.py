"""
File: smartcash/ui/dataset/visualization/tests/test_components_import.py
Deskripsi: Tes untuk memverifikasi bahwa komponen dapat diimpor dengan benar
"""

import unittest

class TestComponentsImport(unittest.TestCase):
    """Test untuk memverifikasi bahwa komponen dapat diimpor dengan benar"""
    
    def test_header_import(self):
        """Test impor komponen header"""
        try:
            from smartcash.ui.components.header import create_header
            self.assertTrue(callable(create_header))
        except ImportError as e:
            self.fail(f"Gagal mengimpor create_header: {e}")
    
    def test_tabs_import(self):
        """Test impor komponen tabs"""
        try:
            from smartcash.ui.components.tabs import create_tabs
            self.assertTrue(callable(create_tabs))
        except ImportError as e:
            self.fail(f"Gagal mengimpor create_tabs: {e}")
    
    def test_header_utils_import(self):
        """Test impor header_utils"""
        try:
            from smartcash.ui.utils.header_utils import create_header
            self.assertTrue(callable(create_header))
        except ImportError as e:
            self.fail(f"Gagal mengimpor create_header dari header_utils: {e}")
    
    def test_tab_factory_import(self):
        """Test impor tab_factory"""
        try:
            from smartcash.ui.components.tab_factory import create_tab_widget
            self.assertTrue(callable(create_tab_widget))
        except ImportError as e:
            self.fail(f"Gagal mengimpor create_tab_widget: {e}")

if __name__ == '__main__':
    unittest.main() 