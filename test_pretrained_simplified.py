"""
File: test_pretrained_simplified.py
Deskripsi: Simplified unit tests for PretrainedInitializer
"""

import pytest
from unittest.mock import MagicMock, patch, ANY

# Mock the parent class and config handler before importing
with patch('smartcash.ui.initializers.common_initializer.CommonInitializer') as mock_common:
    # Import after setting up the mock
    from smartcash.ui.pretrained.pretrained_init import PretrainedInitializer
    
    # Configure the mock
    mock_common.return_value.module_name = 'pretrained_models'
    mock_common.return_value.config_handler = MagicMock()
    mock_common.return_value._create_ui_components.return_value = {}
    mock_common.return_value.initialize.return_value = {}

# Test the basic functionality
def test_initializer_creation():
    """Test basic initialization"""
    initializer = PretrainedInitializer()
    assert initializer is not None
    assert hasattr(initializer, 'module_name')
    assert hasattr(initializer, 'config_handler')
    assert initializer.module_name == 'pretrained_models'

def test_initialize_method():
    """Test the initialize method"""
    initializer = PretrainedInitializer()
    result = initializer.initialize()
    assert result == {}

if __name__ == "__main__":
    pytest.main([__file__, '-v', '-s'])
