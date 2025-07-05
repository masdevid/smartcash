"""
Simple test file to verify basic test execution without mocks.
"""
import unittest

class TestSimple(unittest.TestCase):
    """Simple test case without any mocks."""
    
    def test_addition(self):
        """Test basic addition."""
        self.assertEqual(1 + 1, 2)
        
    def test_string_concatenation(self):
        """Test string concatenation."""
        result = "Hello, " + "World!"
        self.assertEqual(result, "Hello, World!")

if __name__ == '__main__':
    unittest.main()
