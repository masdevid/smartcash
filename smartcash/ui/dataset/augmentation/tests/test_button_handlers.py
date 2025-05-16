"""
File: smartcash/ui/dataset/augmentation/tests/test_button_handlers.py
Deskripsi: Test untuk handler tombol augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from IPython.display import display

from smartcash.ui.dataset.augmentation.handlers.button_handlers import (
    on_augment_click,
    on_stop_click,
    on_reset_click,
    on_save_click,
    on_cleanup_click,
    register_button_handlers
)

class TestButtonHandlers(unittest.TestCase):
    """Test case untuk button handlers augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock UI components
        self.ui_components = {
            'status': MagicMock(),
            'log_accordion': MagicMock(),
            'progress_bar': MagicMock(layout=MagicMock(visibility='hidden')),
            'current_progress': MagicMock(layout=MagicMock(visibility='hidden')),
            'overall_label': MagicMock(layout=MagicMock(visibility='hidden')),
            'step_label': MagicMock(layout=MagicMock(visibility='hidden')),
            'augment_button': MagicMock(layout=MagicMock(display='block')),
            'stop_button': MagicMock(layout=MagicMock(display='none')),
            'cleanup_button': MagicMock(layout=MagicMock(display='none')),
            'visualization_buttons': MagicMock(layout=MagicMock(display='none')),
            'summary_container': MagicMock(layout=MagicMock(display='none')),
            'visualization_container': MagicMock(layout=MagicMock(display='none')),
            'update_config_from_ui': MagicMock(return_value={}),
            'update_ui_from_config': MagicMock(),
            'logger': MagicMock(),
            'augmentation_running': False,
            'action_buttons': MagicMock(
                children=[
                    MagicMock(),  # run_button
                    MagicMock(),  # reset_button
                    MagicMock(),  # clean_button
                    MagicMock()   # visualize_button
                ]
            )
        }
        
        # Mock button
        self.button = MagicMock()
    
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.display')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.clear_output')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.create_status_indicator')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.run_augmentation')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.save_augmentation_config')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.sync_config_with_drive')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.notify_process_start')
    def test_on_augment_click(self, mock_notify, mock_sync, mock_save, mock_run, mock_status, mock_clear, mock_display):
        """Test fungsi on_augment_click."""
        # Setup mocks
        mock_save.return_value = True
        mock_sync.return_value = True
        
        # Call function
        on_augment_click(self.button, self.ui_components)
        
        # Assertions
        self.assertEqual(self.ui_components['augmentation_running'], True)
        self.ui_components['update_config_from_ui'].assert_called_once_with(self.ui_components)
        mock_save.assert_called_once()
        mock_sync.assert_called_once_with(self.ui_components)
        mock_notify.assert_called_once_with(self.ui_components)
        
        # Verify UI updates
        self.assertEqual(self.ui_components['progress_bar'].layout.visibility, 'visible')
        self.assertEqual(self.ui_components['augment_button'].layout.display, 'none')
        self.assertEqual(self.ui_components['stop_button'].layout.display, 'block')
    
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.display')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.clear_output')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.create_status_indicator')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.notify_process_stop')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.cleanup_ui')
    def test_on_stop_click(self, mock_cleanup, mock_notify, mock_status, mock_clear, mock_display):
        """Test fungsi on_stop_click."""
        # Call function
        on_stop_click(self.button, self.ui_components)
        
        # Assertions
        self.assertEqual(self.ui_components['augmentation_running'], False)
        mock_notify.assert_called_once_with(self.ui_components)
        mock_cleanup.assert_called_once_with(self.ui_components)
    
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.display')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.clear_output')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.create_status_indicator')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.reset_config_to_default')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.ensure_ui_persistence')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.update_status_panel')
    def test_on_reset_click(self, mock_update_status, mock_ensure, mock_reset, mock_status, mock_clear, mock_display):
        """Test fungsi on_reset_click."""
        # Setup mocks
        mock_reset.return_value = True
        
        # Call function
        on_reset_click(self.button, self.ui_components)
        
        # Assertions
        mock_reset.assert_called_once_with(self.ui_components)
        self.ui_components['update_ui_from_config'].assert_called_once_with(self.ui_components)
        mock_ensure.assert_called_once_with(self.ui_components)
        mock_update_status.assert_called_once()
    
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.display')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.clear_output')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.create_status_indicator')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.save_augmentation_config')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.sync_config_with_drive')
    def test_on_save_click(self, mock_sync, mock_save, mock_status, mock_clear, mock_display):
        """Test fungsi on_save_click."""
        # Setup mocks
        mock_save.return_value = True
        mock_sync.return_value = True
        
        # Call function
        on_save_click(self.button, self.ui_components)
        
        # Assertions
        self.ui_components['update_config_from_ui'].assert_called_once_with(self.ui_components)
        mock_save.assert_called_once()
        mock_sync.assert_called_once_with(self.ui_components)
    
    def test_register_button_handlers(self):
        """Test fungsi register_button_handlers."""
        # Call function
        register_button_handlers(self.ui_components)
        
        # Assertions
        run_button = self.ui_components['action_buttons'].children[0]
        reset_button = self.ui_components['action_buttons'].children[1]
        clean_button = self.ui_components['action_buttons'].children[2]
        visualize_button = self.ui_components['action_buttons'].children[3]
        
        # Verify on_click was registered
        self.assertEqual(run_button.on_click.call_count, 1)
        self.assertEqual(reset_button.on_click.call_count, 1)
        self.assertEqual(clean_button.on_click.call_count, 1)
        self.assertEqual(visualize_button.on_click.call_count, 1)

if __name__ == '__main__':
    unittest.main()
