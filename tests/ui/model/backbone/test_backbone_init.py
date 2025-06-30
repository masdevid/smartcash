"""
Test script for initializing and running the Backbone module.
"""
import os
import sys
import pytest
import yaml
import ipywidgets as widgets
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, ANY, call

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Sample configuration for testing
SAMPLE_CONFIG = {
    'model': {
        'backbone': 'efficientnet_b4',
        'pretrained': True,
        'freeze_backbone': False,
        'detection_layers': ['reduction_4', 'reduction_5'],
        'layer_mode': 'all',
        'feature_optimization': {
            'enabled': True,
            'method': 'fpn',
            'channels': 256
        },
        'input_size': 640,
        'num_classes': 80
    }
}

# Mock for the ConfigCellInitializer
class MockConfigCellInitializer:
    def __init__(self, config=None, component_id=None, parent_id=None, title=None, description=None, icon=None):
        self.config = config or {}
        self.component_id = component_id or 'backbone'
        self.parent_id = parent_id or 'model'
        self.title = title or 'Model Configuration'
        self.description = description or 'Konfigurasi backbone model YOLOv5 dengan EfficientNet-B4'
        self.icon = icon or '🤖'
        self.ui_components = {}
        self.logger_bridge = MagicMock()
        self.parent_component = MagicMock()
        self._handler = None
        self._logger = MagicMock()
        self._shared_config_manager = None
        self._unsubscribe_func = None
    
    def create_child_components(self):
        return {}
    
    def create_handler(self):
        return MagicMock()
    
    def get_container(self):
        return MagicMock()
    
    def cleanup(self):
        pass

@pytest.fixture
def mock_backbone_initializer():
    """Fixture to create a BackboneInitializer with proper mocks."""
    with patch('smartcash.ui.model.backbone.backbone_init.ConfigCellInitializer', new=MockConfigCellInitializer), \
         patch('smartcash.ui.model.backbone.backbone_init.BackboneConfigHandler') as mock_config_handler, \
         patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_child_components') as mock_create_components, \
         patch('smartcash.ui.model.backbone.components.ui_components.get_layout_sections') as mock_get_layout_sections, \
         patch('smartcash.ui.info_boxes.model_info.get_model_info_content') as mock_get_info_content:
        
        # Setup mocks
        mock_config_handler.get_default_config.return_value = SAMPLE_CONFIG
        mock_config_handler.return_value = MagicMock()
        mock_config_handler.return_value.get_default_config.return_value = SAMPLE_CONFIG
        mock_create_components.return_value = {'form': MagicMock(), 'sections': {}}
        mock_get_layout_sections.return_value = {}
        mock_get_info_content.return_value = '<div>Model Info</div>'
        
        # Mock the _load_existing_config method
        with patch.object(BackboneInitializer, '_load_existing_config', return_value=None):
            # Import here to ensure mocks are in place
            from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
            
            # Create instance with test config
            backbone = BackboneInitializer(config=SAMPLE_CONFIG)
            
            # Add necessary attributes that would be set by parent class
            backbone.ui_components = {}
            backbone.logger_bridge = MagicMock()
            
            yield backbone, mock_config_handler, mock_create_components, mock_get_info_content

@patch('smartcash.ui.model.backbone.handlers.config_handler.BackboneConfigHandler')
def test_initialization(mock_config_handler):
    """Test initialization of the BackboneInitializer."""
    print("\n=== Starting test_initialization ===")
    
    # Setup mocks
    # Create a complete default config structure
    default_config = {
        'model': {
            'backbone': 'default_backbone',
            'pretrained': False,
            'freeze_backbone': True,
            'detection_layers': ['default_layer'],
            'layer_mode': 'default',
            'feature_optimization': {
                'enabled': False,
                'method': 'default',
                'channels': 128
            },
            'input_size': 512,
            'num_classes': 10
        }
    }
    print(f"Default config: {default_config}")
    
    mock_config_handler.get_default_config.return_value = default_config
    
    # Import here to ensure mocks are in place
    from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
    
    # Mock the _load_existing_config method
    with patch.object(BackboneInitializer, '_load_existing_config', return_value=None) as mock_load:
        print("Creating BackboneInitializer instance...")
        # Create instance with test config
        backbone = BackboneInitializer(config=SAMPLE_CONFIG)
        print(f"BackboneInitializer created. Config: {backbone.config}")
        
        # Test basic attributes
        print("Testing basic attributes...")
        assert backbone.component_id == 'backbone', f"Expected 'backbone', got {backbone.component_id}"
        assert backbone.parent_id == 'model', f"Expected 'model', got {backbone.parent_id}"
        assert backbone.title == 'Model Configuration', f"Expected 'Model Configuration', got {backbone.title}"
        assert 'EfficientNet' in backbone.description, f"Expected 'EfficientNet' in description, got {backbone.description}"
        assert backbone.icon == '🤖', f"Expected '🤖', got {backbone.icon}"
        
        # Test that the config was properly merged
        assert 'model' in backbone.config, "No 'model' key in config"
        model_config = backbone.config['model']
        print(f"Model config: {model_config}")
        
        # Check that values from SAMPLE_CONFIG overwrote the defaults
        print("Validating config values...")
        assert model_config['backbone'] == 'efficientnet_b4', f"Expected 'efficientnet_b4', got {model_config['backbone']}"
        assert model_config['pretrained'] is True, f"Expected pretrained=True, got {model_config['pretrained']}"
        assert model_config['freeze_backbone'] is False, f"Expected freeze_backbone=False, got {model_config['freeze_backbone']}"
        assert model_config['detection_layers'] == ['reduction_4', 'reduction_5'], f"Expected ['reduction_4', 'reduction_5'], got {model_config['detection_layers']}"
        assert model_config['layer_mode'] == 'all', f"Expected 'all', got {model_config['layer_mode']}"
        assert model_config['feature_optimization']['enabled'] is True, f"Expected feature_optimization.enabled=True, got {model_config['feature_optimization']['enabled']}"
        assert model_config['feature_optimization']['method'] == 'fpn', f"Expected 'fpn', got {model_config['feature_optimization']['method']}"
        assert model_config['feature_optimization']['channels'] == 256, f"Expected 256, got {model_config['feature_optimization']['channels']}"
        assert model_config['input_size'] == 640, f"Expected 640, got {model_config['input_size']}"
        assert model_config['num_classes'] == 80, f"Expected 80, got {model_config['num_classes']}"
        
        print("All assertions passed!")

def test_create_child_components():
    """Test creation of child components."""
    with patch('smartcash.ui.model.backbone.components.ui_components.create_backbone_child_components') as mock_create_components, \
         patch('smartcash.ui.model.backbone.components.ui_components.get_layout_sections') as mock_get_layout_sections:
        
        # Setup mocks
        mock_create_components.return_value = {'form': MagicMock(), 'sections': {}}
        mock_get_layout_sections.return_value = {}
        
        # Import here to ensure mocks are in place
        from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
        
        # Create instance
        backbone = BackboneInitializer(config=SAMPLE_CONFIG)
        
        # Call the method
        components = backbone.create_child_components()
        
        # Verify the function was called
        mock_create_components.assert_called_once()
        mock_get_layout_sections.assert_called_once()
        
        # Verify the returned components
        assert 'form' in components
        assert 'sections' in components
        assert '_layout_sections' in components

def test_create_handler():
    """Test creation of the model handler."""
    with patch('smartcash.ui.model.backbone.backbone_init.BackboneModelHandler') as mock_handler_class:
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        # Import here to ensure mocks are in place
        from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
        
        # Create instance
        backbone = BackboneInitializer(config=SAMPLE_CONFIG)
        backbone.ui_components = {}
        backbone.logger_bridge = MagicMock()
        
        # Call the method
        handler = backbone.create_handler()
        
        # Verify the handler was created with the correct arguments
        mock_handler_class.assert_called_once()
        
        # Verify the logger_bridge was added to ui_components
        assert 'logger_bridge' in backbone.ui_components
        
        # Verify the handler was returned
        assert handler == mock_handler

def test_get_info_content():
    """Test getting info content."""
    with patch('smartcash.ui.info_boxes.model_info.get_model_info_content') as mock_get_info_content:
        # Setup mocks
        mock_get_info_content.return_value = '<div>Model Info</div>'
        
        # Import here to ensure mocks are in place
        from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
        
        # Create instance
        backbone = BackboneInitializer(config=SAMPLE_CONFIG)
        
        # Call the method
        content = backbone.get_info_content()
        
        # Verify the function was called
        mock_get_info_content.assert_called_once()
        
        # Verify the content was returned
        assert content == '<div>Model Info</div>'

def test_get_container_layout():
    """Test getting container layout."""
    # Import here to ensure mocks are in place
    from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
    
    # Create instance
    backbone = BackboneInitializer(config=SAMPLE_CONFIG)
    
    # Call the method
    layout = backbone.get_container_layout()
    
    # Verify the layout properties
    assert isinstance(layout, widgets.Layout)
    assert layout.width == '100%'
    assert layout.max_width == '1280px'

@patch('os.path.exists')
@patch('builtins.open', new_callable=mock_open, read_data='model:\n  backbone: test_backbone')
@patch('yaml.safe_load')
@patch('smartcash.ui.model.backbone.handlers.config_handler.BackboneConfigHandler')
def test_load_existing_config(mock_config_handler, mock_safe_load, mock_file, mock_exists):
    """Test loading existing configuration from file."""
    # Setup mocks
    mock_exists.return_value = True
    mock_safe_load.return_value = {'model': {'backbone': 'test_backbone'}}
    
    # Mock the default config
    default_config = {
        'model': {
            'backbone': 'default_backbone',
            'pretrained': False,
            'freeze_backbone': True,
            'detection_layers': ['default_layer'],
            'layer_mode': 'default',
            'feature_optimization': {
                'enabled': False,
                'method': 'default',
                'channels': 128
            },
            'input_size': 512,
            'num_classes': 10
        }
    }
    mock_config_handler.get_default_config.return_value = default_config
    
    # Import here to ensure mocks are in place
    from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
    
    # Create instance (should load from file)
    backbone = BackboneInitializer()
    
    # Verify the config was loaded from file
    assert backbone.config['model']['backbone'] == 'test_backbone', \
        f"Expected 'test_backbone', got {backbone.config['model']['backbone']}"
    
    # Verify the rest of the config is from defaults
    model_config = backbone.config['model']
    assert model_config['pretrained'] is False
    assert model_config['freeze_backbone'] is True

@patch('os.path.exists')
@patch('smartcash.ui.model.backbone.handlers.config_handler.BackboneConfigHandler')
def test_load_default_config(mock_config_handler, mock_exists):
    """Test loading default configuration when no config file exists."""
    # Setup mocks
    mock_exists.return_value = False
    
    # Mock the default config
    default_config = {
        'model': {
            'backbone': 'efficientnet_b4',
            'pretrained': True,
            'freeze_backbone': True,
            'detection_layers': ['layer1', 'layer2'],
            'layer_mode': 'default',
            'feature_optimization': {
                'enabled': True,
                'method': 'fpn',
                'channels': 256
            },
            'input_size': 640,
            'num_classes': 80
        }
    }
    mock_config_handler.get_default_config.return_value = default_config
    
    # Import here to ensure mocks are in place
    from smartcash.ui.model.backbone.backbone_init import BackboneInitializer
    
    # Create instance (should use default config)
    backbone = BackboneInitializer()
    
    # Verify the default config was used
    assert backbone.config['model']['backbone'] == 'efficientnet_b4', \
        f"Expected 'efficientnet_b4', got {backbone.config['model']['backbone']}"
    
    # Verify other default values
    model_config = backbone.config['model']
    assert model_config['pretrained'] is True
    assert model_config['freeze_backbone'] is True
    assert model_config['detection_layers'] == ['layer1', 'layer2']
    assert model_config['feature_optimization']['enabled'] is True
