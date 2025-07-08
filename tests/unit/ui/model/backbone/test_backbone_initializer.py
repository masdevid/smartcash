"""
Test module for backbone initializer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from smartcash.ui.model.backbone.backbone_init import BackboneInitializer, initialize_backbone_ui


class TestBackboneInitializer:
    """Test cases for BackboneInitializer."""
    
    @pytest.fixture
    def initializer(self):
        """Create a BackboneInitializer instance for testing."""
        return BackboneInitializer()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'model': {
                'backbone': 'efficientnet_b4',
                'detection_layers': ['banknote'],
                'layer_mode': 'single',
                'feature_optimization': {
                    'enabled': True,
                    'use_attention': True
                },
                'mixed_precision': True
            }
        }
    
    def test_init(self, initializer):
        """Test BackboneInitializer initialization."""
        assert initializer is not None
        assert initializer.module_name == 'backbone'
        assert initializer.parent_module == 'model'
        assert hasattr(initializer, '_ui_components')
        assert hasattr(initializer, '_backbone_factory')
    
    def test_get_default_config(self, initializer):
        """Test getting default configuration."""
        config = initializer.get_default_config()
        
        assert isinstance(config, dict)
        assert 'model' in config
        assert 'backbone' in config['model']
        assert config['model']['backbone'] == 'efficientnet_b4'
    
    @patch('smartcash.ui.model.backbone.backbone_init.create_backbone_ui_components')
    def test_initialize_success(self, mock_create_components, initializer, sample_config):
        """Test successful initialization."""
        # Mock UI components
        mock_ui_components = {
            'main_container': Mock(),
            'ui': Mock(),
            'backbone_dropdown': Mock(),
            'save_button': Mock()
        }
        mock_create_components.return_value = mock_ui_components
        
        # Mock handler creation
        with patch.object(initializer, 'create_module_handler') as mock_create_handler:
            mock_handler = Mock()
            mock_create_handler.return_value = mock_handler
            
            result = initializer.initialize(sample_config)
        
        assert result['success'] is True
        assert result['ui_components'] == mock_ui_components
        assert result['config'] == sample_config
        mock_create_components.assert_called_once_with(sample_config)
    
    @patch('smartcash.ui.model.backbone.backbone_init.create_backbone_ui_components')
    def test_initialize_ui_components_failure(self, mock_create_components, initializer):
        """Test initialization failure when UI components creation fails."""
        mock_create_components.return_value = None
        
        result = initializer.initialize()
        
        assert result['success'] is False
        assert 'Failed to create UI components' in result['error']
    
    @patch('smartcash.ui.model.backbone.backbone_init.create_backbone_ui_components')
    def test_initialize_exception_handling(self, mock_create_components, initializer):
        """Test initialization exception handling."""
        mock_create_components.side_effect = Exception("Test error")
        
        result = initializer.initialize()
        
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
    
    @patch('model.utils.backbone_factory.BackboneFactory')
    def test_setup_backbone_factory_success(self, mock_factory_class, initializer):
        """Test successful backbone factory setup."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        initializer.setup_backbone_factory()
        
        assert initializer._backbone_factory == mock_factory
    
    def test_setup_backbone_factory_failure(self, initializer):
        """Test backbone factory setup failure."""
        # This should handle import errors gracefully
        initializer.setup_backbone_factory()
        
        # Should not crash, factory should be None
        assert initializer._backbone_factory is None
    
    @patch('smartcash.ui.model.backbone.backbone_init.create_backbone_ui_components')
    def test_setup_handlers(self, mock_create_components, initializer):
        """Test handlers setup."""
        mock_ui_components = {
            'backbone_dropdown': Mock(),
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


class TestInitializeBackboneUI:
    """Test cases for initialize_backbone_ui function."""
    
    @patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer.initialize_module_ui')
    def test_initialize_backbone_ui(self, mock_initialize):
        """Test backbone UI initialization function."""
        mock_ui = Mock()
        mock_initialize.return_value = mock_ui
        
        config = {'test': 'config'}
        result = initialize_backbone_ui(config)
        
        mock_initialize.assert_called_once_with(
            module_name='backbone',
            parent_module='model',
            config=config,
            initializer_class=BackboneInitializer
        )
        assert result == mock_ui
    
    @patch('smartcash.ui.core.initializers.module_initializer.ModuleInitializer.initialize_module_ui')
    def test_initialize_backbone_ui_no_config(self, mock_initialize):
        """Test backbone UI initialization without config."""
        mock_ui = Mock()
        mock_initialize.return_value = mock_ui
        
        result = initialize_backbone_ui()
        
        mock_initialize.assert_called_once_with(
            module_name='backbone',
            parent_module='model',
            config=None,
            initializer_class=BackboneInitializer
        )