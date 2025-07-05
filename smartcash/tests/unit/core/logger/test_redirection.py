"""Tests for logging redirection to log accordion in operation container."""
import unittest
from unittest.mock import MagicMock, patch, ANY
import ipywidgets as widgets

class TestLoggingRedirection(unittest.TestCase):
    """Test logging redirection to log accordion in operation container."""
    
    @patch('smartcash.ui.components.operation_container.OperationContainer')
    @patch('smartcash.ui.components.log_accordion.LogAccordion')
    def test_logging_redirection(self, mock_log_accordion, mock_op_container):
        """Test that logs are properly redirected to the log accordion."""
        # Setup mock log accordion
        mock_accordion = MagicMock()
        mock_log_accordion.return_value = mock_accordion
        
        # Setup mock operation container
        mock_container = MagicMock()
        mock_container.log_accordion = mock_accordion
        mock_op_container.return_value = mock_container
        
        # Import after patching to ensure mocks are in place
        from smartcash.ui.components.operation_container import create_operation_container
        
        # Create container
        container = create_operation_container(
            show_progress=True,
            show_dialog=True,
            show_logs=True,
            log_module_name="TestLogger"
        )
        
        # Test logging at different levels
        test_messages = [
            ("info", "Test info message"),
            ("warning", "Test warning message"),
            ("error", "Test error message"),
            ("debug", "Test debug message"),
            ("critical", "Test critical message")
        ]
        
        for level, message in test_messages:
            # Call the appropriate log method
            getattr(container, level)(message)
            
            # Check that log_accordion.add_log was called with the right arguments
            mock_accordion.add_log.assert_any_call(
                message=message,
                level=level.upper(),
                timestamp=ANY,  # We can't predict the exact timestamp
                module_name="TestLogger"
            )

if __name__ == '__main__':
    unittest.main()
