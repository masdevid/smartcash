"""
Integration tests for BackboneInitializer.
"""
import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock
from smartcash.ui.model.backbone.backbone_init import BackboneInitializer

class TestBackboneInitializer:
    """Integration tests for BackboneInitializer."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        os.makedirs('config', exist_ok=True)
        
        # Create a minimal config file
        with open('config/model_config.yaml', 'w') as f:
            f.write("""
            model:
              backbone: test_backbone
              model_name: test_model
            """)
        
        yield temp_dir
        
        # Cleanup
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)
    
    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        initializer = BackboneInitializer()
        assert initializer.config['model']['backbone'] == 'efficientnet_b4'
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'model': {
                'backbone': 'custom_backbone',
                'model_name': 'custom_model'
            }
        }
        initializer = BackboneInitializer(config=custom_config)
        assert initializer.config['model']['backbone'] == 'custom_backbone'
    
    def test_initialization_loads_existing_config(self, temp_config_dir):
        """Test that initialization loads existing config file."""
        initializer = BackboneInitializer()
        assert initializer.config['model']['backbone'] == 'test_backbone'
    
    def test_create_child_components(self):
        """Test creation of child components."""
        initializer = BackboneInitializer()
        components = initializer.create_child_components()
        
        # Verify essential components are created
        assert 'model_form' in components
        assert 'config_summary' in components
        assert '_layout_sections' in components
    
    def test_get_info_content(self):
        """Test getting info content."""
        initializer = BackboneInitializer()
        content = initializer.get_info_content()
        
        # Verify content contains expected sections
        assert 'Model Configuration Guide' in content
        assert 'Backbone Options' in content
        assert 'Detection Layers' in content
