"""
File: smartcash/ui/dataset/preprocessing/tests/test_preprocessing_ui.py
Deskripsi: Unit test untuk komponen UI preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

class TestPreprocessingUI(unittest.TestCase):
    """Test untuk komponen UI preprocessing dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk create_preprocessing_ui
        self.create_ui_patch = patch('smartcash.ui.dataset.preprocessing.components.preprocessing_component.create_preprocessing_ui')
        self.mock_create_ui = self.create_ui_patch.start()
        
        # Mock UI components
        self.mock_ui_components = {
            'main_container': widgets.VBox(),
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value,
            'preprocess_button': widgets.Button(description='Preprocess'),
            'stop_button': widgets.Button(description='Stop'),
            'reset_button': widgets.Button(description='Reset'),
            'cleanup_button': widgets.Button(description='Cleanup'),
            'save_button': widgets.Button(description='Save'),
            'split_selector': widgets.Dropdown(
                options=['All Splits', 'Train Only', 'Validation Only', 'Test Only'],
                value='All Splits'
            ),
            'preprocess_options': widgets.VBox([
                widgets.IntSlider(value=640, min=32, max=1280, description='Size:'),
                widgets.Checkbox(value=True, description='Normalize'),
                widgets.Checkbox(value=True, description='Preserve Aspect Ratio'),
                widgets.Checkbox(value=True, description='Cache'),
                widgets.IntSlider(value=4, min=1, max=16, description='Workers:')
            ]),
            'validation_options': widgets.VBox([
                widgets.Checkbox(value=True, description='Validate'),
                widgets.Checkbox(value=True, description='Fix Issues'),
                widgets.Checkbox(value=True, description='Move Invalid'),
                widgets.Text(value='data/invalid', description='Invalid Dir:')
            ])
        }
        
        # Setup return value untuk create_preprocessing_ui
        self.mock_create_ui.return_value = self.mock_ui_components
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.create_ui_patch.stop()
    
    def test_create_preprocessing_ui(self):
        """Test untuk fungsi create_preprocessing_ui."""
        from smartcash.ui.dataset.preprocessing.components.preprocessing_component import create_preprocessing_ui
        
        # Panggil fungsi yang akan ditest
        result = create_preprocessing_ui()
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        self.mock_create_ui.assert_called_once()
    
    @patch('smartcash.ui.utils.base_initializer.initialize_module_ui')
    def test_initialize_preprocessing_ui(self, mock_initialize_module_ui):
        """Test untuk fungsi initialize_preprocessing_ui."""
        from smartcash.ui.dataset.preprocessing.preprocessing_initializer import initialize_preprocessing_ui
        
        # Setup return value untuk initialize_module_ui
        mock_initialize_module_ui.return_value = self.mock_ui_components
        
        # Panggil fungsi yang akan ditest
        result = initialize_preprocessing_ui()
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        mock_initialize_module_ui.assert_called_once()
        
        # Verifikasi parameter yang diberikan ke initialize_module_ui
        args, kwargs = mock_initialize_module_ui.call_args
        self.assertEqual(kwargs['module_name'], 'preprocessing')
        self.assertEqual(kwargs['button_keys'], ['preprocess_button', 'stop_button', 'reset_button', 'cleanup_button', 'save_button'])
        self.assertTrue('multi_progress_config' in kwargs)
        self.assertEqual(kwargs['observer_group'], 'preprocessing_observers')

if __name__ == '__main__':
    unittest.main()
