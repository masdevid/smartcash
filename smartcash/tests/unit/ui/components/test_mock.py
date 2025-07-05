"""
Test file to verify mock functionality.
"""
import unittest
from unittest.mock import MagicMock

class TestMock(unittest.TestCase):
    """Test case with basic mock functionality."""
    
    def test_mock_basic(self):
        """Test basic mock functionality."""
        # Create a mock object
        mock_obj = MagicMock()
        
        # Set a return value for a method
        mock_obj.some_method.return_value = 42
        
        # Call the method and verify the return value
        result = mock_obj.some_method()
        self.assertEqual(result, 42)
        
        # Verify the method was called
        mock_obj.some_method.assert_called_once()

if __name__ == '__main__':
    unittest.main()
