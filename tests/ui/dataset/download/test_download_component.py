import unittest
from unittest.mock import MagicMock
from smartcash.ui.dataset.download.components import create_download_main_ui

class TestDownloadComponent(unittest.TestCase):
    """
    Unit test untuk komponen UI download
    """
    
    def test_ui_creation(self):
        """
        Test pembuatan UI komponen download
        """
        ui_components = create_download_main_ui()
        
        # Verifikasi komponen kritis ada
        self.assertIn('project_input', ui_components)
        self.assertIn('version_input', ui_components)
        self.assertIn('api_key_input', ui_components)
        self.assertIn('download_button', ui_components)
        self.assertIn('log_output', ui_components)
        self.assertIn('status_panel', ui_components)

if __name__ == '__main__':
    unittest.main()
