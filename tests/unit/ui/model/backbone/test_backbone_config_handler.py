"""
Test module for backbone configuration handler.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from smartcash.ui.model.backbone.configs.backbone_config_handler import BackboneConfigHandler
from smartcash.ui.model.backbone.configs.backbone_defaults import get_default_backbone_config


class TestBackboneConfigHandler:
    """Test cases for BackboneConfigHandler."""
    
    @pytest.fixture
    def config_handler(self):
        """Create a BackboneConfigHandler instance for testing."""
        return BackboneConfigHandler()
    
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
    
    def test_init(self, config_handler):
        """Test BackboneConfigHandler initialization."""
        assert config_handler is not None
        assert hasattr(config_handler, '_config')
        assert hasattr(config_handler, '_available_backbones')
        assert hasattr(config_handler, '_detection_layers')
        assert hasattr(config_handler, '_layer_modes')
    
    def test_get_config(self, config_handler):
        """Test getting configuration."""
        config = config_handler.get_config()
        
        assert isinstance(config, dict)
        assert 'model' in config
        assert 'backbone' in config['model']
        
        # Should return a copy, not the original
        config['model']['backbone'] = 'modified'
        original_config = config_handler.get_config()
        assert original_config['model']['backbone'] != 'modified'
    
    def test_update_config_valid(self, config_handler, sample_config):
        """Test updating configuration with valid data."""
        result = config_handler.update_config(sample_config)
        
        assert result is True
        updated_config = config_handler.get_config()
        assert updated_config['model']['backbone'] == 'efficientnet_b4'
        assert updated_config['model']['detection_layers'] == ['banknote']
    
    def test_update_config_invalid_backbone(self, config_handler):
        """Test updating configuration with invalid backbone."""
        invalid_config = {
            'model': {
                'backbone': 'invalid_backbone',
                'detection_layers': ['banknote']
            }
        }
        
        result = config_handler.update_config(invalid_config)
        assert result is False
    
    def test_validate_config_valid(self, config_handler, sample_config):
        """Test configuration validation with valid config."""
        result = config_handler.validate_config(sample_config)
        assert result is True
    
    def test_validate_config_invalid_backbone(self, config_handler):
        """Test configuration validation with invalid backbone."""
        invalid_config = {
            'model': {
                'backbone': 'nonexistent_backbone'
            }
        }
        
        result = config_handler.validate_config(invalid_config)
        assert result is False
    
    def test_validate_config_invalid_layer_mode(self, config_handler):
        """Test configuration validation with incompatible layer mode."""
        invalid_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'detection_layers': ['banknote', 'nominal'],
                'layer_mode': 'single'
            }
        }
        
        result = config_handler.validate_config(invalid_config)
        assert result is False
    
    def test_reset_to_defaults(self, config_handler):
        """Test resetting configuration to defaults."""
        # First modify the config
        config_handler.set_backbone('cspdarknet')
        
        # Reset to defaults
        config_handler.reset_to_defaults()
        
        config = config_handler.get_config()
        default_config = get_default_backbone_config()
        assert config['model']['backbone'] == default_config['model']['backbone']
    
    def test_set_backbone_valid(self, config_handler):
        """Test setting valid backbone."""
        result = config_handler.set_backbone('cspdarknet')
        assert result is True
        
        config = config_handler.get_config()
        assert config['model']['backbone'] == 'cspdarknet'
    
    def test_set_backbone_invalid(self, config_handler):
        """Test setting invalid backbone."""
        result = config_handler.set_backbone('invalid_backbone')
        assert result is False
    
    def test_set_detection_layers_valid(self, config_handler):
        """Test setting valid detection layers."""
        layers = ['banknote', 'nominal']
        result = config_handler.set_detection_layers(layers)
        assert result is True
        
        config = config_handler.get_config()
        assert 'banknote' in config['model']['detection_layers']
        assert 'nominal' in config['model']['detection_layers']
    
    def test_set_detection_layers_auto_add_banknote(self, config_handler):
        """Test that banknote layer is automatically added."""
        layers = ['nominal']
        result = config_handler.set_detection_layers(layers)
        assert result is True
        
        config = config_handler.get_config()
        assert 'banknote' in config['model']['detection_layers']
        assert 'nominal' in config['model']['detection_layers']
    
    def test_set_layer_mode_valid(self, config_handler):
        """Test setting valid layer mode."""
        result = config_handler.set_layer_mode('multilayer')
        assert result is True
        
        config = config_handler.get_config()
        assert config['model']['layer_mode'] == 'multilayer'
    
    def test_set_layer_mode_single_auto_adjust(self, config_handler):
        """Test that single mode auto-adjusts detection layers."""
        # First set multiple layers
        config_handler.set_detection_layers(['banknote', 'nominal'])
        
        # Then set single mode
        result = config_handler.set_layer_mode('single')
        assert result is True
        
        config = config_handler.get_config()
        assert config['model']['layer_mode'] == 'single'
        assert config['model']['detection_layers'] == ['banknote']
    
    def test_set_feature_optimization(self, config_handler):
        """Test setting feature optimization."""
        config_handler.set_feature_optimization(True, True)
        
        config = config_handler.get_config()
        feature_opt = config['model']['feature_optimization']
        assert feature_opt['enabled'] is True
        assert feature_opt['use_attention'] is True
        assert feature_opt['testing_mode'] is False
    
    def test_set_mixed_precision(self, config_handler):
        """Test setting mixed precision."""
        config_handler.set_mixed_precision(False)
        
        config = config_handler.get_config()
        assert config['model']['mixed_precision'] is False
    
    def test_get_backbone_info(self, config_handler):
        """Test getting backbone information."""
        info = config_handler.get_backbone_info('efficientnet_b4')
        
        assert isinstance(info, dict)
        assert 'display_name' in info
        assert 'description' in info
        
        # Test current backbone info
        current_info = config_handler.get_backbone_info()
        assert isinstance(current_info, dict)
    
    def test_get_available_backbones(self, config_handler):
        """Test getting available backbones."""
        backbones = config_handler.get_available_backbones()
        
        assert isinstance(backbones, dict)
        assert 'efficientnet_b4' in backbones
        assert 'cspdarknet' in backbones
    
    def test_get_detection_layers_config(self, config_handler):
        """Test getting detection layers configuration."""
        layers_config = config_handler.get_detection_layers_config()
        
        assert isinstance(layers_config, dict)
        assert 'banknote' in layers_config
        assert 'nominal' in layers_config
        assert 'security' in layers_config
    
    def test_get_layer_modes_config(self, config_handler):
        """Test getting layer modes configuration."""
        modes_config = config_handler.get_layer_modes_config()
        
        assert isinstance(modes_config, dict)
        assert 'single' in modes_config
        assert 'multilayer' in modes_config