"""
File: smartcash/ui/dataset/preprocessing/tests/test_button_handlers.py
Deskripsi: Unit test untuk button handler preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
from IPython.display import display

class TestButtonHandlers(unittest.TestCase):
    """Test untuk button handler preprocessing dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk display
        self.display_patch = patch('IPython.display.display')
        self.mock_display = self.display_patch.start()
        
        # Mock untuk create_status_indicator
        self.status_indicator_patch = patch('smartcash.ui.utils.alert_utils.create_status_indicator')
        self.mock_status_indicator = self.status_indicator_patch.start()
        
        # Mock untuk update_status_panel
        self.update_status_patch = patch('smartcash.ui.dataset.preprocessing.handlers.status_handler.update_status_panel')
        self.mock_update_status = self.update_status_patch.start()
        
        # Mock UI components
        self.mock_ui_components = {
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
            ]),
            'preprocessing_running': False,
            'update_progress': MagicMock(),
            'reset_progress_bar': MagicMock()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.display_patch.stop()
        self.status_indicator_patch.stop()
        self.update_status_patch.stop()
    
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_start')
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.disable_ui_during_processing')
    @patch('smartcash.ui.dataset.preprocessing.handlers.button_handler.execute_preprocessing')
    def test_on_primary_click(self, mock_execute, mock_disable, mock_notify):
        """Test untuk handler on_primary_click."""
        from smartcash.ui.dataset.preprocessing.handlers.button_handler import setup_preprocessing_button_handlers
        
        # Setup mock
        mock_button = MagicMock()
        
        # Setup button handlers
        ui_components = setup_preprocessing_button_handlers(self.mock_ui_components)
        
        # Panggil handler on_primary_click
        ui_components['on_primary_click'](mock_button)
        
        # Verifikasi hasil
        self.assertTrue(ui_components['preprocessing_running'])
        mock_disable.assert_called_once_with(ui_components, True)
        self.mock_update_status.assert_called_once()
        mock_notify.assert_called_once()
        mock_execute.assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_stop')
    def test_on_stop_click(self, mock_notify):
        """Test untuk handler on_stop_click."""
        from smartcash.ui.dataset.preprocessing.handlers.button_handler import setup_preprocessing_button_handlers
        
        # Setup mock
        mock_button = MagicMock()
        
        # Setup button handlers
        ui_components = setup_preprocessing_button_handlers(self.mock_ui_components)
        ui_components['preprocessing_running'] = True
        
        # Panggil handler on_stop_click
        ui_components['on_stop_click'](mock_button)
        
        # Verifikasi hasil
        self.assertFalse(ui_components['preprocessing_running'])
        self.mock_display.assert_called_once()
        self.mock_update_status.assert_called_once()
        mock_notify.assert_called_once()
    
    def test_on_reset_click(self):
        """Test untuk handler on_reset_click."""
        from smartcash.ui.dataset.preprocessing.handlers.button_handler import setup_preprocessing_button_handlers
        
        # Setup mock
        mock_button = MagicMock()
        
        # Setup button handlers
        ui_components = setup_preprocessing_button_handlers(self.mock_ui_components)
        
        # Panggil handler on_reset_click
        ui_components['on_reset_click'](mock_button)
        
        # Verifikasi hasil
        self.mock_display.assert_called_once()
        self.mock_update_status.assert_called_once()
        self.mock_ui_components['reset_progress_bar'].assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_complete')
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_error')
    def test_execute_preprocessing_success(self, mock_notify_error, mock_notify_complete):
        """Test untuk fungsi execute_preprocessing dengan hasil sukses."""
        from smartcash.ui.dataset.preprocessing.handlers.button_handler import execute_preprocessing
        
        # Setup mock
        self.mock_ui_components['dataset_manager'] = MagicMock()
        self.mock_ui_components['dataset_manager'].preprocess.return_value = {'status': 'success'}
        
        # Panggil fungsi yang akan ditest
        execute_preprocessing(self.mock_ui_components, 'train', 'Train Split')
        
        # Verifikasi hasil
        self.mock_ui_components['dataset_manager'].preprocess.assert_called_once()
        mock_notify_complete.assert_called_once()
        mock_notify_error.assert_not_called()
    
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_complete')
    @patch('smartcash.ui.dataset.preprocessing.utils.ui_observers.notify_process_error')
    def test_execute_preprocessing_error(self, mock_notify_error, mock_notify_complete):
        """Test untuk fungsi execute_preprocessing dengan hasil error."""
        from smartcash.ui.dataset.preprocessing.handlers.button_handler import execute_preprocessing
        
        # Setup mock
        self.mock_ui_components['dataset_manager'] = MagicMock()
        self.mock_ui_components['dataset_manager'].preprocess.side_effect = Exception("Test error")
        
        # Panggil fungsi yang akan ditest
        execute_preprocessing(self.mock_ui_components, 'train', 'Train Split')
        
        # Verifikasi hasil
        self.mock_ui_components['dataset_manager'].preprocess.assert_called_once()
        mock_notify_error.assert_called_once()
        mock_notify_complete.assert_not_called()

if __name__ == '__main__':
    unittest.main()
