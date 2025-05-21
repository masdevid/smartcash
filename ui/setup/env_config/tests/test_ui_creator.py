"""
File: smartcash/ui/setup/env_config/tests/test_ui_creator.py
Deskripsi: Unit test untuk ui_creator.py
"""

import unittest
from unittest.mock import patch, MagicMock

class TestUiCreator(unittest.TestCase):
    """
    Test untuk ui_creator.py
    """
    
    @patch('smartcash.ui.setup.env_config.components.ui_factory.UIFactory.create_ui_components')
    def test_create_env_config_ui(self, mock_create_ui_components):
        """
        Test create_env_config_ui
        """
        # Setup mock
        expected_result = {
            'header': MagicMock(),
            'setup_button': MagicMock(),
            'status_panel': MagicMock()
        }
        mock_create_ui_components.return_value = expected_result
        
        # Import di sini untuk menghindari circular import
        from smartcash.ui.setup.env_config.components.ui_creator import create_env_config_ui
        
        # Call function to test
        result = create_env_config_ui()
        
        # Verify UIFactory.create_ui_components dipanggil
        mock_create_ui_components.assert_called_once()
        
        # Verify hasil yang dikembalikan
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main() 