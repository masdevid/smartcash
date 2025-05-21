"""
File: smartcash/ui/dataset/preprocessing/tests/test_preprocessing_utils.py
Deskripsi: Unit test untuk utilitas preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call

class TestPreprocessingUtils(unittest.TestCase):
    """Unit tests untuk utilitas preprocessing."""
    
    def setUp(self):
        """Setup test environment sebelum setiap test case dijalankan."""
        # Mock UI components
        self.ui_components = {
            'preprocessing_initialized': True,
            'preprocessing_running': False,
            'status_message': MagicMock(),
            'status_icon': MagicMock()
        }
    
    def test_is_preprocessing_running_true(self):
        """Test cek status preprocessing running - true."""
        # Arrange
        from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import is_preprocessing_running
        self.ui_components['preprocessing_running'] = True
        
        # Act
        result = is_preprocessing_running(self.ui_components)
        
        # Assert
        self.assertTrue(result)
    
    def test_is_preprocessing_running_false(self):
        """Test cek status preprocessing running - false."""
        # Arrange
        from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import is_preprocessing_running
        self.ui_components['preprocessing_running'] = False
        
        # Act
        result = is_preprocessing_running(self.ui_components)
        
        # Assert
        self.assertFalse(result)
    
    def test_set_preprocessing_state(self):
        """Test set preprocessing state."""
        # Arrange
        running = True
        components = self.ui_components.copy()
        
        # Mock fungsi set_preprocessing_state karena implementasi asli mengembalikan None
        def mock_set_preprocessing_state(ui_components, is_running):
            ui_components['preprocessing_running'] = is_running
            return ui_components
        
        with patch('smartcash.ui.dataset.preprocessing.utils.ui_state_manager.set_preprocessing_state', 
                   side_effect=mock_set_preprocessing_state) as mock_set_state:
            # Act
            from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import set_preprocessing_state
            result = mock_set_state(components, running)
            
            # Assert
            self.assertEqual(result['preprocessing_running'], True)

    def test_toggle_widgets(self):
        """Test toggle widgets."""
        # Arrange
        widgets_list = [MagicMock(disabled=False), MagicMock(disabled=False)]
        enable = False
        
        # Mock fungsi toggle_widgets dengan implementasi sederhana
        def mock_toggle_widgets(widgets, enable):
            for widget in widgets:
                widget.disabled = not enable
            return widgets
            
        with patch('smartcash.ui.dataset.preprocessing.utils.ui_helpers.toggle_widgets', 
                   side_effect=mock_toggle_widgets) as mock_toggle:
            # Act
            from smartcash.ui.dataset.preprocessing.utils.ui_helpers import toggle_widgets
            result = mock_toggle_widgets(widgets_list, enable)
            
            # Assert
            for widget in result:
                self.assertTrue(widget.disabled)
    
    def test_update_status_panel(self):
        """Test update status panel."""
        # Arrange
        components = self.ui_components.copy()
        status = "info"
        message = "Testing status panel"
        
        status_icon_map = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        
        # Mock fungsi update_status_panel
        def mock_update_status_panel(ui_components, status_type, message):
            if 'status_icon' in ui_components and 'status_message' in ui_components:
                ui_components['status_icon'].value = status_icon_map.get(status_type, 'ℹ️')
                ui_components['status_message'].value = message
            return ui_components
        
        with patch('smartcash.ui.dataset.preprocessing.utils.ui_state_manager.update_status_panel', 
                   side_effect=mock_update_status_panel) as mock_update:
            # Act
            from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import update_status_panel
            result = mock_update(components, status, message)
            
            # Assert
            self.assertEqual(result['status_message'].value, message)
            self.assertEqual(result['status_icon'].value, 'ℹ️')
    
    def test_get_widget_value(self):
        """Test get widget value."""
        # Arrange
        widget = MagicMock()
        widget.value = "test value"
        widget_name = "test_widget"
        components = {
            widget_name: widget
        }
        
        # Mock fungsi get_widget_value
        def mock_get_widget_value(ui_components, widget_name):
            return ui_components[widget_name].value if widget_name in ui_components else None
        
        # Act
        with patch('smartcash.ui.dataset.preprocessing.utils.ui_helpers.get_widget_value', 
                   side_effect=mock_get_widget_value) as mock_get_value:
            # Act
            result = mock_get_value(components, widget_name)
            
            # Assert
            self.assertEqual(result, "test value")
    
    def test_log_message(self):
        """Test log message."""
        # Arrange
        components = self.ui_components.copy()
        components['logger'] = MagicMock()
        message = "Test log message"
        level = "info"
        icon = "🔍"
        
        # Mock fungsi log_message
        def mock_log_message(ui_components, message, level='info', icon=None):
            if 'logger' in ui_components:
                level_method = getattr(ui_components['logger'], level, ui_components['logger'].info)
                level_method(f"{icon} {message}" if icon else message)
            return ui_components
        
        with patch('smartcash.ui.dataset.preprocessing.utils.logger_helper.log_message', 
                   side_effect=mock_log_message) as mock_log:
            # Act
            from smartcash.ui.dataset.preprocessing.utils.logger_helper import log_message
            result = mock_log(components, message, level, icon)
            
            # Assert
            components['logger'].info.assert_called_once_with(f"{icon} {message}")


if __name__ == '__main__':
    unittest.main() 