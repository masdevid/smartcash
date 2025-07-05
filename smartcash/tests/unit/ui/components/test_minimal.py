"""
Minimal test file to isolate pytest plugin issues.
"""
import unittest
from unittest.mock import MagicMock, patch

class TestMinimal(unittest.TestCase):
    """Minimal test case to verify test execution."""
    
    def test_minimal(self):
        """Test that the test framework is working."""
        self.assertTrue(True)
        
    @patch('unittest.mock.MagicMock')
    def test_with_mock(self, mock_magic):
        """Test with a simple mock."""
        mock_instance = MagicMock()
        mock_instance.some_method.return_value = 42
        self.assertEqual(mock_instance.some_method(), 42)

if __name__ == '__main__':
    unittest.main()
