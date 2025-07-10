"""
Test module for colab initializer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from smartcash.ui.setup.colab.colab_initializer import ColabInitializer, initialize_colab_ui


class TestColabInitializer:
    """Test cases for ColabInitializer."""
    
    @pytest.fixture
    def initializer(self):
        """Create a ColabInitializer instance for testing."""
        return ColabInitializer()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'environment': {
                'type': 'colab',
                'auto_mount_drive': True,
                'project_name': 'SmartCash',
                'gpu_enabled': True
            },
            'setup': {
                'stages': ['environment_detection', 'drive_mount'],
                'auto_start': False
            }
        }
    
    def test_init(self, initializer):
        """Test ColabInitializer initialization."""
        assert initializer is not None
        assert initializer.module_name == 'colab'
        assert initializer.parent_module == 'setup'
        assert hasattr(initializer, '_ui_components')
        assert hasattr(initializer, '_operation_handlers')
        assert hasattr(initializer, '_environment_manager')
    
    def test_get_default_config(self, initializer):
        """Test getting default configuration."""
        config = initializer.get_default_config()
        
        assert isinstance(config, dict)
        assert 'environment' in config
        assert 'setup' in config
        assert config['environment']['type'] == 'colab'
    
    @patch('smartcash.ui.setup.colab.colab_initializer.create_colab_ui')
    def test_initialize_impl_success(self, mock_create_components, initializer, sample_config):
        """Test successful initialization implementation."""
        # Mock UI components
        mock_ui_components = {
            'main_container': Mock(),
            'ui': Mock(),
            'environment_type_dropdown': Mock(),
            'save_button': Mock()
        }
        mock_create_components.return_value = mock_ui_components
        
        # Mock handler creation
        with patch.object(initializer, 'create_module_handler') as mock_create_handler:
            mock_handler = Mock()
            mock_create_handler.return_value = mock_handler
            
            result = initializer._initialize_impl(sample_config)
        
        assert result['success'] is True
        assert result['ui_components'] == mock_ui_components
        assert result['config'] == sample_config
        mock_create_components.assert_called_once_with(sample_config)
    
    @patch('smartcash.ui.setup.colab.colab_initializer.create_colab_ui')
    def test_initialize_impl_ui_components_failure(self, mock_create_components, initializer):
        """Test initialization failure when UI components creation fails."""
        mock_create_components.return_value = None
        
        result = initializer._initialize_impl()
        
        assert result['success'] is False
        assert 'Failed to create UI components' in result['error']
    
    @patch('smartcash.ui.setup.colab.colab_initializer.create_colab_ui')
    def test_initialize_impl_exception_handling(self, mock_create_components, initializer):
        """Test initialization exception handling."""
        mock_create_components.side_effect = Exception("Test error")
        
        result = initializer._initialize_impl()
        
        assert result['success'] is False
        assert 'Test error' in result['error']
    
    def test_pre_initialize_checks(self, initializer):
        """Test pre-initialization checks."""
        # This should not raise an exception if all imports are available
        try:
            initializer.pre_initialize_checks()
        except RuntimeError as e:
            # If we get an import error, that's expected in test environment
            assert "Missing required components" in str(e)
    
    def test_post_initialize_cleanup_success(self, initializer):
        """Test post-initialization cleanup with valid components."""
        initializer._ui_components = {
            'main_container': Mock(),
            'ui': Mock()
        }
        
        # Should not raise an exception
        initializer.post_initialize_cleanup()
    
    def test_post_initialize_cleanup_failure(self, initializer):
        """Test post-initialization cleanup with missing components."""
        initializer._ui_components = None
        
        with pytest.raises(RuntimeError, match="No UI components were created"):
            initializer.post_initialize_cleanup()
    
    def test_setup_environment_manager_success(self, initializer):
        """Test successful environment manager setup."""
        with patch('smartcash.common.environment.get_environment_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            initializer.setup_environment_manager()
            
            assert initializer._environment_manager == mock_manager
    
    def test_setup_environment_manager_failure(self, initializer):
        """Test environment manager setup failure."""
        # This should handle import errors gracefully
        initializer.setup_environment_manager()
        
        # Should not crash, manager should be None
        assert initializer._environment_manager is None
    
    @patch('smartcash.ui.setup.colab.colab_initializer.create_colab_ui')
    def test_setup_handlers(self, mock_create_components, initializer):
        """Test handlers setup."""
        mock_ui_components = {
            'environment_type_dropdown': Mock(),
            'save_button': Mock()
        }
        
        # Mock handler creation
        with patch.object(initializer, 'create_module_handler') as mock_create_handler:
            mock_handler = Mock()
            mock_create_handler.return_value = mock_handler
            
            initializer.setup_handlers(mock_ui_components)
        
        assert hasattr(initializer, '_module_handler')
        assert hasattr(initializer, '_handlers')
        assert 'module' in initializer._handlers
    
    def test_setup_handlers_no_components(self, initializer):
        """Test handlers setup with no components."""
        with pytest.raises(ValueError, match="No UI components provided"):
            initializer.setup_handlers({})
    
    def test_setup_operation_handlers_with_factory(self, initializer):
        """Test operation handlers setup with factory available."""
        mock_ui_components = {'test': 'component'}
        initializer._ui_components = mock_ui_components
        
        with patch('smartcash.ui.setup.colab.operations.factory.OperationHandlerFactory') as mock_factory:
            mock_factory.create_handler.return_value = Mock()
            
            initializer.setup_operation_handlers()
            
            # Should have created operation handlers
            assert isinstance(initializer._operation_handlers, dict)
    
    def test_setup_operation_handlers_without_factory(self, initializer):
        """Test operation handlers setup without factory available."""
        # This should handle import errors gracefully
        initializer.setup_operation_handlers()
        
        # Should create empty handlers dict
        assert isinstance(initializer._operation_handlers, dict)


class TestInitializeColabUI:
    """Test cases for initialize_colab_ui function."""
    
    @patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer.initialize_module_ui')
    def test_initialize_colab_ui(self, mock_initialize):
        """Test colab UI initialization function."""
        mock_ui = Mock()
        mock_initialize.return_value = mock_ui
        
        config = {'test': 'config'}
        result = initialize_colab_ui(config)
        
        mock_initialize.assert_called_once_with(
            module_name='colab',
            parent_module='setup',
            config=config,
            initializer_class=ColabInitializer
        )
        assert result == mock_ui
    
    @patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer.initialize_module_ui')
    def test_initialize_colab_ui_no_config(self, mock_initialize):
        """Test colab UI initialization without config."""
        mock_ui = Mock()
        mock_initialize.return_value = mock_ui
        
        result = initialize_colab_ui()
        
        mock_initialize.assert_called_once_with(
            module_name='colab',
            parent_module='setup',
            config=None,
            initializer_class=ColabInitializer
        )