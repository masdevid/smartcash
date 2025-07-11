"""
Basic test suite for Colab UIModule implementation.
Simple tests to verify core functionality works.
"""

import pytest
from unittest.mock import Mock, patch

from smartcash.ui.setup.colab.colab_uimodule import (
    ColabUIModule,
    create_colab_uimodule,
    register_colab_template
)
from smartcash.ui.core.ui_module import ModuleStatus
from smartcash.ui.core.ui_module_factory import UIModuleFactory
from smartcash.ui.core.ui_module import SharedMethodRegistry


class TestColabUIModuleBasic:
    """Basic test for ColabUIModule functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        UIModuleFactory.reset_factory()
        SharedMethodRegistry._shared_methods.clear()
        SharedMethodRegistry._method_metadata.clear()
    
    def test_basic_initialization(self):
        """Test basic ColabUIModule initialization."""
        module = ColabUIModule()
        
        assert module.module_name == "colab"
        assert module.parent_module == "setup"
        assert module.full_module_name == "setup.colab"
        assert module.get_status() == ModuleStatus.PENDING
        assert not module.is_ready()
    
    def test_environment_detection_local(self):
        """Test environment detection in local environment."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.configs.colab_config_handler.ColabConfigHandler'), \
             patch('smartcash.ui.setup.colab.operations.operation_manager.ColabOperationManager') as mock_ops:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops.return_value.get_operations.return_value = {}
            
            # No Google Colab available (normal case)
            module = ColabUIModule()
            module.initialize()
            
            assert not module.is_colab_environment()
            assert module.get_config("is_colab") is False
            assert module.get_config("environment_type") == "local"
    
    def test_create_colab_uimodule(self):
        """Test create_colab_uimodule function."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.ColabUIModule') as mock_module_class:
            mock_module = Mock()
            mock_module.initialize.return_value = mock_module
            mock_module_class.return_value = mock_module
            
            # Create module
            result = create_colab_uimodule({"test_config": "value"}, auto_initialize=True)
            
            assert result is mock_module
            mock_module.initialize.assert_called_once()
    
    def test_template_registration(self):
        """Test Colab template registration."""
        register_colab_template()
        
        template = UIModuleFactory.get_template("colab", "setup")
        
        assert template is not None
        assert template.module_name == "colab"
        assert template.parent_module == "setup"
        assert template.auto_initialize is False  # Colab-specific
        assert len(template.required_components) > 0
        assert len(template.required_operations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])