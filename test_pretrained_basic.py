"""
File: test_pretrained_basic.py
Deskripsi: Basic tests for PretrainedInitializer with minimal mocking
"""

import pytest
from unittest.mock import patch, MagicMock

def test_pretrained_initializer_creation():
    """Test that PretrainedInitializer can be created"""
    # Mock the parent class __init__ to do nothing
    with patch('smartcash.ui.initializers.common_initializer.CommonInitializer.__init__', return_value=None):
        from smartcash.ui.pretrained.pretrained_init import PretrainedInitializer
        
        # Create instance - this should work without errors
        initializer = PretrainedInitializer()
        
        # Basic verification that we got an instance
        assert initializer is not None

def test_initialize_pretrained_ui():
    """Test the factory function"""
    with patch('smartcash.ui.pretrained.pretrained_init._pretrained_initializer') as mock_initializer:
        # Setup mock return value
        expected_result = {'ui': 'test'}
        mock_initializer.initialize.return_value = expected_result
        
        from smartcash.ui.pretrained.pretrained_init import initialize_pretrained_ui
        
        # Call the factory function
        result = initialize_pretrained_ui(env='test_env', config={'test': 'config'})
        
        # Verify the result
        assert result == expected_result
        mock_initializer.initialize.assert_called_once_with(
            env='test_env',
            config={'test': 'config'}
        )

if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
