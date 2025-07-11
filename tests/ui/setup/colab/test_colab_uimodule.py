"""
Comprehensive test suite for Colab UIModule implementation.
Tests the new UIModule pattern integration with existing Colab functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from smartcash.ui.setup.colab.colab_uimodule import (
    ColabUIModule,
    create_colab_uimodule,
    get_colab_uimodule,
    reset_colab_uimodule,
    register_colab_template,
    register_colab_shared_methods,
    initialize_colab_ui,
    get_colab_components,
    display_colab_ui
)
from smartcash.ui.core.ui_module import ModuleStatus, SharedMethodRegistry
from smartcash.ui.core.ui_module_factory import UIModuleFactory


class TestColabUIModule:
    """Test ColabUIModule class functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        # Reset global state
        reset_colab_uimodule()
        SharedMethodRegistry._shared_methods.clear()
        SharedMethodRegistry._method_metadata.clear()
        UIModuleFactory.reset_factory()
    
    def test_basic_initialization(self):
        """Test basic ColabUIModule initialization."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui:
            mock_ui.return_value = {
                'main_container': Mock(),
                'form_container': Mock(),
                'action_container': Mock()
            }
            
            module = ColabUIModule()
            
            assert module.module_name == "colab"
            assert module.parent_module == "setup"
            assert module.full_module_name == "setup.colab"
            assert module.get_status() == ModuleStatus.PENDING
            assert not module.is_ready()
    
    def test_environment_detection_colab(self):
        """Test environment detection in Google Colab."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops.return_value.get_operations.return_value = {'test_op': Mock()}
            
            # Mock Google Colab import
            with patch.dict('sys.modules', {'google.colab': Mock()}):
                module = ColabUIModule()
                module.initialize()
                
                assert module.is_colab_environment()
                assert module.get_config("is_colab") is True
                assert module.get_config("environment_type") == "colab"
    
    def test_environment_detection_local(self):
        """Test environment detection in local environment."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops.return_value.get_operations.return_value = {'test_op': Mock()}
            
            # No Google Colab available
            module = ColabUIModule()
            module.initialize()
            
            assert not module.is_colab_environment()
            assert module.get_config("is_colab") is False
            assert module.get_config("environment_type") == "local"
    
    def test_ui_component_creation(self):
        """Test UI component creation and registration."""
        mock_components = {
            'main_container': Mock(),
            'header_container': Mock(),
            'form_container': Mock(),
            'action_container': Mock(),
            'operation_container': Mock()
        }
        
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = mock_components
            mock_ops.return_value.get_operations.return_value = {}
            
            module = ColabUIModule()
            module.initialize()
            
            # Check components are registered
            assert len(module.list_components()) == len(mock_components)
            for component_type in mock_components:
                assert module.get_component(component_type) is mock_components[component_type]
    
    def test_operation_registration(self):
        """Test operation registration."""
        mock_operations = {
            'full_setup': Mock(),
            'init': Mock(),
            'drive_mount': Mock(),
            'verify': Mock()
        }
        
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops.return_value.get_operations.return_value = mock_operations
            
            module = ColabUIModule()
            module.initialize()
            
            # Check operations are registered (including convenience methods)
            operations = module.list_operations()
            assert len(operations) >= len(mock_operations)
            for op_name in mock_operations:
                assert op_name in operations
            
            # Check convenience methods
            assert 'status' in operations
            assert 'reset' in operations
    
    def test_get_environment_status(self):
        """Test environment status retrieval."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops.return_value.get_operations.return_value = {'test_op': Mock()}
            
            module = ColabUIModule()
            module.initialize()
            
            status = module.get_environment_status()
            
            assert isinstance(status, dict)
            assert status["module"] == "setup.colab"
            assert "environment_type" in status
            assert "is_colab" in status
            assert "module_status" in status
            assert "ready" in status
            assert "timestamp" in status
    
    def test_reset_environment(self):
        """Test environment reset functionality."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops_instance = Mock()
            mock_ops.return_value = mock_ops_instance
            mock_ops_instance.get_operations.return_value = {'test_op': Mock()}
            
            module = ColabUIModule()
            module.initialize()
            
            # Reset environment
            result = module.reset_environment()
            
            assert result["success"] is True
            assert "message" in result
            assert "environment_type" in result
            mock_ops_instance.cleanup.assert_called_once()
    
    def test_execute_full_setup(self):
        """Test full setup execution."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops_instance = Mock()
            mock_ops.return_value = mock_ops_instance
            mock_ops_instance.get_operations.return_value = {'full_setup': Mock()}
            
            # Mock operation result
            from smartcash.ui.core.handlers.operation_handler import OperationResult, OperationStatus
            mock_result = OperationResult(
                status=OperationStatus.COMPLETED,
                message="Setup completed successfully",
                duration=5.0,
                data={"steps_completed": 7}
            )
            mock_ops_instance.execute_named_operation.return_value = mock_result
            
            module = ColabUIModule()
            module.initialize()
            
            # Execute full setup
            result = module.execute_full_setup(project_name="test_project")
            
            assert result["success"] is True
            assert result["status"] == "completed"
            assert result["message"] == "Setup completed successfully"
            assert result["duration"] == 5.0
            mock_ops_instance.execute_named_operation.assert_called_with("full_setup", project_name="test_project")


class TestColabUIModuleFactory:
    """Test Colab UIModule factory functions."""
    
    def setup_method(self):
        """Setup for each test."""
        reset_colab_uimodule()
        SharedMethodRegistry._shared_methods.clear()
        SharedMethodRegistry._method_metadata.clear()
        UIModuleFactory.reset_factory()
    
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
    
    def test_create_colab_uimodule_singleton(self):
        """Test singleton behavior of create_colab_uimodule."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.ColabUIModule') as mock_module_class:
            mock_module = Mock()
            mock_module_class.return_value = mock_module
            
            # Create first instance
            result1 = create_colab_uimodule()
            
            # Create second instance (should be same)
            result2 = create_colab_uimodule()
            
            assert result1 is result2
            mock_module_class.assert_called_once()  # Only called once due to singleton
    
    def test_create_colab_uimodule_force_new(self):
        """Test force_new parameter."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.ColabUIModule') as mock_module_class:
            mock_module1 = Mock()
            mock_module2 = Mock()
            mock_module_class.side_effect = [mock_module1, mock_module2]
            
            # Create first instance
            result1 = create_colab_uimodule()
            
            # Force create new instance
            result2 = create_colab_uimodule(force_new=True)
            
            assert result1 is not result2
            assert mock_module_class.call_count == 2
    
    def test_get_colab_uimodule(self):
        """Test get_colab_uimodule function."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.ColabUIModule') as mock_module_class:
            mock_module = Mock()
            mock_module_class.return_value = mock_module
            
            # Get when none exists (should create)
            result1 = get_colab_uimodule(create_if_missing=True)
            assert result1 is mock_module
            
            # Get existing
            result2 = get_colab_uimodule()
            assert result2 is result1
    
    def test_get_colab_uimodule_no_create(self):
        """Test get_colab_uimodule with create_if_missing=False."""
        result = get_colab_uimodule(create_if_missing=False)
        assert result is None
    
    def test_reset_colab_uimodule(self):
        """Test reset_colab_uimodule function."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.ColabUIModule') as mock_module_class:
            mock_module = Mock()
            mock_module_class.return_value = mock_module
            
            # Create instance
            create_colab_uimodule()
            
            # Reset
            reset_colab_uimodule()
            
            # Verify cleanup was called
            mock_module.cleanup.assert_called_once()
            
            # Verify new instance is created on next call
            create_colab_uimodule()
            assert mock_module_class.call_count == 2


class TestColabSharedMethods:
    """Test Colab shared methods registration and functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        SharedMethodRegistry._shared_methods.clear()
        SharedMethodRegistry._method_metadata.clear()
    
    def test_register_colab_shared_methods(self):
        """Test registration of Colab shared methods."""
        register_colab_shared_methods()
        
        methods = SharedMethodRegistry.list_methods(category="operations")
        
        assert "mount_google_drive" in methods
        assert "detect_colab_environment" in methods
        assert "setup_project_structure" in methods
    
    def test_mount_google_drive_method(self):
        """Test mount_google_drive shared method."""
        register_colab_shared_methods()
        mount_drive = SharedMethodRegistry.get_method("mount_google_drive")
        
        # Test in non-Colab environment
        result = mount_drive("/test/drive")
        assert result["success"] is False
        assert "Not running in Google Colab" in result["error"]
    
    def test_detect_colab_environment_method(self):
        """Test detect_colab_environment shared method."""
        register_colab_shared_methods()
        detect_env = SharedMethodRegistry.get_method("detect_colab_environment")
        
        # Test in non-Colab environment
        result = detect_env()
        assert result["is_colab"] is False
        assert result["runtime_type"] == "local"
        assert result["gpu_available"] is False
    
    def test_setup_project_structure_method(self):
        """Test setup_project_structure shared method."""
        register_colab_shared_methods()
        setup_project = SharedMethodRegistry.get_method("setup_project_structure")
        
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.__truediv__') as mock_div:
            
            mock_path = Mock()
            mock_div.return_value = mock_path
            
            result = setup_project("test_project", "/tmp")
            
            assert result["success"] is True
            assert "test_project" in result["project_path"]
            assert "created_dirs" in result


class TestBackwardCompatibility:
    """Test backward compatibility layer."""
    
    def setup_method(self):
        """Setup for each test."""
        reset_colab_uimodule()
        SharedMethodRegistry._shared_methods.clear()
        SharedMethodRegistry._method_metadata.clear()
        UIModuleFactory.reset_factory()
    
    def test_initialize_colab_ui(self):
        """Test initialize_colab_ui backward compatibility function."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_uimodule') as mock_create, \
             patch('IPython.display.display') as mock_display:
            
            mock_module = Mock()
            mock_container = Mock()
            mock_module.get_component.return_value = mock_container
            mock_create.return_value = mock_module
            
            # Call legacy function
            initialize_colab_ui({"test": "config"})
            
            # Verify calls
            mock_create.assert_called_once_with({"test": "config"}, auto_initialize=True)
            mock_module.get_component.assert_called_once_with('main_container')
            mock_display.assert_called_once_with(mock_container)
    
    def test_get_colab_components(self):
        """Test get_colab_components backward compatibility function."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_uimodule') as mock_create:
            mock_module = Mock()
            mock_module.list_components.return_value = ['comp1', 'comp2']
            mock_module.get_component.side_effect = lambda x: f"component_{x}"
            mock_create.return_value = mock_module
            
            # Call legacy function
            result = get_colab_components({"test": "config"})
            
            # Verify result
            assert result == {
                'comp1': 'component_comp1',
                'comp2': 'component_comp2'
            }
            mock_create.assert_called_once_with({"test": "config"}, auto_initialize=True)
    
    def test_display_colab_ui(self):
        """Test display_colab_ui backward compatibility function."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.initialize_colab_ui') as mock_init:
            
            # Call legacy function
            display_colab_ui({"test": "config"})
            
            # Verify it calls initialize_colab_ui
            mock_init.assert_called_once_with({"test": "config"})


class TestTemplateRegistration:
    """Test Colab template registration."""
    
    def setup_method(self):
        """Setup for each test."""
        UIModuleFactory.reset_factory()
    
    def test_register_colab_template(self):
        """Test Colab template registration."""
        register_colab_template()
        
        template = UIModuleFactory.get_template("colab", "setup")
        
        assert template is not None
        assert template.module_name == "colab"
        assert template.parent_module == "setup"
        assert template.auto_initialize is False  # Colab-specific
        assert len(template.required_components) > 0
        assert len(template.required_operations) > 0
        assert "Google Colab" in template.description


class TestErrorHandling:
    """Test error handling in Colab UIModule."""
    
    def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui:
            mock_ui.side_effect = Exception("UI creation failed")
            
            module = ColabUIModule()
            
            with pytest.raises(Exception, match="UI creation failed"):
                module.initialize()
            
            assert module.get_status() == ModuleStatus.ERROR
    
    def test_operation_error_handling(self):
        """Test error handling in operations."""
        with patch('smartcash.ui.setup.colab.colab_uimodule.create_colab_ui') as mock_ui, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabOperationManager') as mock_ops, \
             patch('smartcash.ui.setup.colab.colab_uimodule.ColabConfigHandler') as mock_config:
            
            mock_ui.return_value = {'main_container': Mock()}
            mock_ops_instance = Mock()
            mock_ops.return_value = mock_ops_instance
            mock_ops_instance.get_operations.return_value = {}
            mock_ops_instance.execute_named_operation.side_effect = Exception("Operation failed")
            
            module = ColabUIModule()
            module.initialize()
            
            # Execute operation that fails
            result = module.execute_full_setup()
            
            assert result["success"] is False
            assert "Operation failed" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])