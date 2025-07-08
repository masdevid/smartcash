"""
Test module for backbone UI handler.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from smartcash.ui.model.backbone.handlers.backbone_ui_handler import BackboneUIHandler


class TestBackboneUIHandler:
    """Test cases for BackboneUIHandler."""
    
    @pytest.fixture
    def ui_handler(self):
        """Create a BackboneUIHandler instance for testing."""
        return BackboneUIHandler()
    
    @pytest.fixture
    def mock_ui_components(self):
        """Mock UI components for testing."""
        return {
            'backbone_dropdown': Mock(value='efficientnet_b4'),
            'detection_layers_select': Mock(value=['banknote']),
            'layer_mode_dropdown': Mock(value='single'),
            'feature_optimization_checkbox': Mock(value=True),
            'mixed_precision_checkbox': Mock(value=True),
            'config_summary': Mock(),
            'build_btn': Mock(),
            'validate_btn': Mock(),
            'info_btn': Mock(),
            'save_button': Mock(),
            'reset_button': Mock()
        }
    
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
    
    def test_init(self, ui_handler):
        """Test BackboneUIHandler initialization."""
        assert ui_handler is not None
        assert ui_handler.module_name == 'backbone'
        assert ui_handler.parent_module == 'model'
        assert hasattr(ui_handler, 'config_handler')
        assert hasattr(ui_handler, '_backbone_factory')
    
    def test_extract_config_from_ui(self, ui_handler, mock_ui_components):
        """Test extracting configuration from UI components."""
        ui_handler._ui_components = mock_ui_components
        
        config = ui_handler.extract_config_from_ui()
        
        assert isinstance(config, dict)
        assert 'model' in config
        assert config['model']['backbone'] == 'efficientnet_b4'
        assert config['model']['detection_layers'] == ['banknote']
        assert config['model']['layer_mode'] == 'single'
        assert config['model']['feature_optimization']['enabled'] is True
        assert config['model']['mixed_precision'] is True
    
    def test_extract_config_from_ui_missing_components(self, ui_handler):
        """Test extracting configuration with missing UI components."""
        ui_handler._ui_components = {}
        
        # Should return default config without error
        config = ui_handler.extract_config_from_ui()
        assert isinstance(config, dict)
    
    def test_update_ui_from_config(self, ui_handler, mock_ui_components, sample_config):
        """Test updating UI components from configuration."""
        ui_handler._ui_components = mock_ui_components
        
        ui_handler.update_ui_from_config(sample_config)
        
        # Check that UI components were updated
        assert mock_ui_components['backbone_dropdown'].value == 'efficientnet_b4'
        assert mock_ui_components['detection_layers_select'].value == ('banknote',)
        assert mock_ui_components['layer_mode_dropdown'].value == 'single'
        assert mock_ui_components['feature_optimization_checkbox'].value is True
        assert mock_ui_components['mixed_precision_checkbox'].value is True
    
    def test_setup(self, ui_handler, mock_ui_components):
        """Test UI handler setup."""
        with patch.object(ui_handler, '_setup_event_handlers') as mock_setup_events:
            with patch.object(ui_handler, 'sync_ui_with_config') as mock_sync:
                ui_handler.setup(mock_ui_components)
        
        assert ui_handler._ui_components == mock_ui_components
        mock_setup_events.assert_called_once()
        mock_sync.assert_called_once()
    
    def test_setup_event_handlers(self, ui_handler, mock_ui_components):
        """Test event handlers setup."""
        ui_handler._ui_components = mock_ui_components
        
        # Mock observe and on_click methods
        for component in mock_ui_components.values():
            component.observe = Mock()
            component.on_click = Mock()
        
        ui_handler._setup_event_handlers()
        
        # Check that observe was called for form widgets
        mock_ui_components['backbone_dropdown'].observe.assert_called()
        mock_ui_components['detection_layers_select'].observe.assert_called()
        
        # Check that on_click was called for buttons
        mock_ui_components['build_btn'].on_click.assert_called()
        mock_ui_components['validate_btn'].on_click.assert_called()
    
    def test_on_form_change(self, ui_handler, mock_ui_components):
        """Test form change handling."""
        ui_handler._ui_components = mock_ui_components
        
        with patch.object(ui_handler, 'extract_config_from_ui') as mock_extract:
            with patch.object(ui_handler, '_validate_form_state') as mock_validate:
                mock_extract.return_value = {'model': {'backbone': 'efficientnet_b4'}}
                
                change_mock = Mock()
                ui_handler._on_form_change('backbone_dropdown', change_mock)
        
        mock_extract.assert_called_once()
        mock_validate.assert_called_once()
    
    def test_validate_form_state(self, ui_handler):
        """Test form state validation."""
        config = {
            'model': {
                'layer_mode': 'single',
                'detection_layers': ['banknote', 'nominal'],  # Multiple layers with single mode
                'backbone': 'efficientnet_b4'
            }
        }
        
        with patch.object(ui_handler, 'track_status') as mock_track:
            ui_handler._validate_form_state(config)
        
        # Should track a warning about layer mode compatibility
        mock_track.assert_called()
        call_args = mock_track.call_args[0]
        assert 'Single layer mode with multiple detection layers' in call_args[0]
        assert call_args[1] == 'warning'
    
    def test_sync_config_with_ui(self, ui_handler, mock_ui_components):
        """Test syncing configuration with UI state."""
        ui_handler._ui_components = mock_ui_components
        
        with patch.object(ui_handler, 'extract_config_from_ui') as mock_extract:
            mock_config = {'test': 'config'}
            mock_extract.return_value = mock_config
            
            ui_handler.sync_config_with_ui()
        
        mock_extract.assert_called_once()
    
    def test_sync_ui_with_config(self, ui_handler):
        """Test syncing UI with configuration."""
        with patch.object(ui_handler, 'update_ui_from_config') as mock_update:
            ui_handler.sync_ui_with_config()
        
        mock_update.assert_called_once()
    
    @patch('model.utils.backbone_factory.BackboneFactory')
    def test_get_backbone_factory_success(self, mock_factory_class, ui_handler):
        """Test successful backbone factory retrieval."""
        mock_factory = Mock()
        mock_factory_class.return_value = mock_factory
        
        factory = ui_handler.get_backbone_factory()
        
        assert factory == mock_factory
        assert ui_handler._backbone_factory == mock_factory
    
    def test_get_backbone_factory_failure(self, ui_handler):
        """Test backbone factory retrieval failure."""
        # This should handle import errors gracefully
        factory = ui_handler.get_backbone_factory()
        
        assert factory is None
        assert ui_handler._backbone_factory is None
    
    def test_handle_build_model_success(self, ui_handler, mock_ui_components):
        """Test successful model building."""
        ui_handler._ui_components = mock_ui_components
        
        # Mock backbone factory
        mock_factory = Mock()
        mock_backbone = Mock()
        mock_factory.create_backbone.return_value = mock_backbone
        mock_factory.validate_backbone_compatibility.return_value = True
        mock_backbone.get_info.return_value = {'num_parameters': 1000000}
        
        with patch.object(ui_handler, 'get_backbone_factory') as mock_get_factory:
            with patch.object(ui_handler, 'track_status') as mock_track:
                with patch.object(ui_handler, 'extract_config_from_ui') as mock_extract:
                    mock_get_factory.return_value = mock_factory
                    mock_extract.return_value = {
                        'model': {
                            'backbone': 'efficientnet_b4',
                            'pretrained': True,
                            'feature_optimization': {}
                        }
                    }
                    
                    ui_handler._handle_build_model()
        
        # Check that success status was tracked
        success_calls = [call for call in mock_track.call_args_list 
                        if call[0][1] == 'success']
        assert len(success_calls) > 0
    
    def test_handle_build_model_no_factory(self, ui_handler, mock_ui_components):
        """Test model building when no factory is available."""
        ui_handler._ui_components = mock_ui_components
        
        with patch.object(ui_handler, 'get_backbone_factory') as mock_get_factory:
            with patch.object(ui_handler, 'track_status') as mock_track:
                mock_get_factory.return_value = None
                
                ui_handler._handle_build_model()
        
        # Check that error status was tracked
        error_calls = [call for call in mock_track.call_args_list 
                      if call[0][1] == 'error']
        assert len(error_calls) > 0
    
    def test_handle_validate_config(self, ui_handler, mock_ui_components):
        """Test configuration validation handling."""
        ui_handler._ui_components = mock_ui_components
        
        with patch.object(ui_handler, 'extract_config_from_ui') as mock_extract:
            with patch.object(ui_handler, 'track_status') as mock_track:
                mock_extract.return_value = {'model': {'backbone': 'efficientnet_b4'}}
                
                ui_handler._handle_validate_config()
        
        mock_track.assert_called()
    
    def test_handle_reset_config(self, ui_handler):
        """Test configuration reset handling."""
        with patch.object(ui_handler, 'update_ui_from_config') as mock_update:
            with patch.object(ui_handler, 'track_status') as mock_track:
                ui_handler._handle_reset_config()
        
        mock_update.assert_called_once()
        mock_track.assert_called_once()
    
    def test_get_selected_backbone(self, ui_handler, mock_ui_components):
        """Test getting selected backbone."""
        ui_handler._ui_components = mock_ui_components
        
        backbone = ui_handler.get_selected_backbone()
        assert backbone == 'efficientnet_b4'
    
    def test_get_detection_layers(self, ui_handler, mock_ui_components):
        """Test getting detection layers."""
        ui_handler._ui_components = mock_ui_components
        
        layers = ui_handler.get_detection_layers()
        assert layers == ['banknote']
    
    def test_get_layer_mode(self, ui_handler, mock_ui_components):
        """Test getting layer mode."""
        ui_handler._ui_components = mock_ui_components
        
        mode = ui_handler.get_layer_mode()
        assert mode == 'single'
    
    def test_set_backbone(self, ui_handler):
        """Test setting backbone."""
        with patch.object(ui_handler.config_handler, 'set_backbone') as mock_set:
            with patch.object(ui_handler, 'sync_ui_with_config') as mock_sync:
                mock_set.return_value = True
                
                result = ui_handler.set_backbone('cspdarknet')
        
        assert result is True
        mock_set.assert_called_once_with('cspdarknet')
        mock_sync.assert_called_once()
    
    def test_initialize(self, ui_handler):
        """Test UI handler initialization."""
        # Should not raise an exception
        ui_handler.initialize()