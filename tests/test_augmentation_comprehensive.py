"""
Comprehensive test suite for augmentation module.
Tests UI initialization, live preview functionality, and operations.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import the augmentation module components
from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule
from smartcash.ui.dataset.augmentation.components.augmentation_ui import create_augment_ui
from smartcash.ui.dataset.augmentation.components.live_preview_widget import create_live_preview_widget
from smartcash.ui.dataset.augmentation.operations.augment_preview_operation import AugmentPreviewOperation


class TestAugmentationModule:
    """Test suite for augmentation module functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.test_config = {
            'target_split': 'train',
            'output_path': '/data/augmented',
            'preview_path': '/data/aug_preview.jpg'
        }
    
    def test_ui_module_initialization(self):
        """Test that the augmentation UI module initializes correctly."""
        # Create module instance
        module = AugmentationUIModule()
        
        # Test initialization
        result = module.initialize()
        assert result == True, "Module initialization should succeed"
        
        # Test that UI components are created
        assert hasattr(module, '_ui_components'), "Module should have UI components"
        assert module._ui_components is not None, "UI components should not be None"
        assert 'main_container' in module._ui_components, "Should have main_container"
    
    def test_ui_components_creation(self):
        """Test that UI components are created correctly."""
        # Create UI components
        ui_components = create_augment_ui(self.test_config)
        
        # Test basic structure
        assert isinstance(ui_components, dict), "Should return dictionary"
        assert 'main_container' in ui_components, "Should have main_container"
        assert 'header_container' in ui_components, "Should have header_container"
        assert 'form_container' in ui_components, "Should have form_container"
        assert 'action_container' in ui_components, "Should have action_container"
        assert 'operation_container' in ui_components, "Should have operation_container"
        
        # Test button availability
        assert 'augment' in ui_components, "Should have augment button"
        assert 'cleanup' in ui_components, "Should have cleanup button"
        assert 'generate' in ui_components, "Should have generate button"
    
    def test_live_preview_widget_creation(self):
        """Test that live preview widget is created correctly."""
        # Create live preview widget
        preview_widget = create_live_preview_widget()
        
        # Test structure
        assert isinstance(preview_widget, dict), "Should return dictionary"
        assert 'container' in preview_widget, "Should have container"
        assert 'widgets' in preview_widget, "Should have widgets"
        
        # Test widgets
        widgets = preview_widget['widgets']
        assert 'preview_image' in widgets, "Should have preview_image widget"
        assert 'generate' in widgets, "Should have generate button"
        assert 'preview_status' in widgets, "Should have preview_status widget"
        
        # Test configuration
        assert 'preview_config' in preview_widget, "Should have preview config"
        preview_config = preview_widget['preview_config']
        assert preview_config['output_path'] == '/data/aug_preview.jpg', "Should have correct output path"
        assert preview_config['image_size'] == (200, 200), "Should have correct image size"
    
    def test_live_preview_operation_initialization(self):
        """Test that preview operation initializes correctly."""
        # Mock UI module
        ui_module = MagicMock()
        ui_module._ui_components = {
            'form_container': {
                'widgets': {
                    'preview_widget': {
                        'widgets': {
                            'preview_image': MagicMock(),
                            'preview_status': MagicMock()
                        }
                    }
                }
            }
        }
        
        # Create preview operation
        operation = AugmentPreviewOperation(
            ui_module=ui_module,
            config=self.test_config,
            callbacks={}
        )
        
        # Test initialization
        assert operation._ui_module == ui_module, "Should store UI module reference"
        assert operation._config == self.test_config, "Should store config"
        assert operation._preview_path is None, "Should initialize with no preview path"
    
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_preview_loading(self, mock_open, mock_exists):
        """Test preview image loading functionality."""
        # Mock file system
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = b'fake_image_data'
        
        # Mock UI components
        mock_image_widget = MagicMock()
        mock_status_widget = MagicMock()
        
        ui_module = MagicMock()
        ui_module._ui_components = {
            'form_container': {
                'widgets': {
                    'preview_widget': {
                        'widgets': {
                            'preview_image': mock_image_widget,
                            'preview_status': mock_status_widget
                        }
                    }
                }
            }
        }
        
        # Create operation and test loading
        operation = AugmentPreviewOperation(
            ui_module=ui_module,
            config=self.test_config,
            callbacks={}
        )
        
        # Test loading existing preview
        result = operation.load_existing_preview()
        
        # Verify the loading process
        assert mock_exists.called, "Should check if files exist"
        assert mock_open.called, "Should open file to read image data"
        
        # Since we mocked the widgets, we can't assert the result directly
        # but we can verify the method completed without error
        assert isinstance(result, bool), "Should return boolean result"
    
    def test_preview_operation_execution(self):
        """Test preview operation execution."""
        # Mock backend service
        mock_service = MagicMock()
        mock_service.return_value = {
            'preview_path': '/data/aug_preview.jpg',
            'success': True
        }
        
        # Mock UI module
        ui_module = MagicMock()
        ui_module._ui_components = {
            'form_container': {
                'widgets': {
                    'preview_widget': {
                        'widgets': {
                            'preview_image': MagicMock(),
                            'preview_status': MagicMock()
                        }
                    }
                }
            }
        }
        
        # Create operation
        operation = AugmentPreviewOperation(
            ui_module=ui_module,
            config=self.test_config,
            callbacks={}
        )
        
        # Mock the backend API
        operation.get_backend_api = MagicMock(return_value=mock_service)
        operation._load_preview_to_widget = MagicMock(return_value=True)
        operation.log_operation_start = MagicMock()
        operation.update_operation_status = MagicMock()
        operation._handle_error = MagicMock()
        
        # Execute operation
        result = operation.execute()
        
        # Test result
        assert isinstance(result, dict), "Should return dictionary"
        assert result.get('success') == True, "Should succeed"
        assert 'preview_path' in result, "Should include preview path"
        assert result['message'] == 'Preview created successfully', "Should have success message"
    
    def test_button_handler_integration(self):
        """Test that button handlers are properly integrated."""
        # Create module
        module = AugmentationUIModule()
        module.initialize()
        
        # Get button handlers
        handlers = module._get_module_button_handlers()
        
        # Test that all expected handlers are present
        expected_handlers = ['augment', 'status', 'cleanup', 'generate', 'save', 'reset']
        for handler_name in expected_handlers:
            assert handler_name in handlers, f"Should have {handler_name} handler"
            assert callable(handlers[handler_name]), f"{handler_name} handler should be callable"
    
    def test_module_config_handling(self):
        """Test configuration handling in the module."""
        # Create module
        module = AugmentationUIModule()
        
        # Test default config
        default_config = module.get_default_config()
        assert isinstance(default_config, dict), "Should return default config dict"
        
        # Test config handler creation
        config_handler = module.create_config_handler(default_config)
        assert config_handler is not None, "Should create config handler"
    
    def test_ui_component_access_patterns(self):
        """Test various UI component access patterns used by operations."""
        # Create UI components
        ui_components = create_augment_ui(self.test_config)
        
        # Test that the preview widget can be accessed through various paths
        # (This simulates how operations find UI widgets)
        form_container = ui_components.get('form_container')
        assert form_container is not None, "Should have form container"
        
        if isinstance(form_container, dict) and 'widgets' in form_container:
            form_widgets = form_container['widgets']
            preview_widget = form_widgets.get('preview_widget')
            
            if preview_widget and isinstance(preview_widget, dict):
                preview_widgets = preview_widget.get('widgets', {})
                assert 'preview_image' in preview_widgets or 'generate' in preview_widgets, \
                    "Should have preview-related widgets"
    
    def test_error_handling_in_ui_creation(self):
        """Test error handling during UI creation."""
        # Test with invalid config
        invalid_config = {'invalid': 'config'}
        
        # Should not raise exception, should return valid UI structure
        ui_components = create_augment_ui(invalid_config)
        assert isinstance(ui_components, dict), "Should handle invalid config gracefully"
        assert 'main_container' in ui_components, "Should still create main container"
    
    def test_module_cleanup_and_resources(self):
        """Test module cleanup and resource management."""
        # Create module
        module = AugmentationUIModule()
        module.initialize()
        
        # Test that UI components are properly referenced
        ui_components = module.get_ui_components()
        assert isinstance(ui_components, dict), "Should return UI components"
        
        # Test that main container is accessible
        main_container = module.get_main_widget()
        assert main_container is not None, "Should have accessible main widget"


class TestAugmentationLivePreview:
    """Specific tests for live preview functionality."""
    
    def setup_method(self):
        """Setup for preview tests."""
        self.test_preview_path = "/tmp/test_aug_preview.jpg"
        self.test_image_data = b"fake_jpeg_data"
    
    def test_preview_widget_structure(self):
        """Test the structure of the preview widget."""
        widget_data = create_live_preview_widget()
        
        # Check main structure
        assert 'container' in widget_data
        assert 'widgets' in widget_data
        assert 'preview_config' in widget_data
        
        # Check widgets
        widgets = widget_data['widgets']
        assert 'preview_image' in widgets
        assert 'generate' in widgets
        assert 'preview_status' in widgets
        
        # Check preview config
        config = widget_data['preview_config']
        assert config['output_path'] == '/data/aug_preview.jpg'
        assert config['image_size'] == (200, 200)
        assert config['format'] == 'jpg'
        assert config['responsive'] == True
    
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open')
    def test_preview_image_loading_success(self, mock_open, mock_getsize, mock_exists):
        """Test successful preview image loading."""
        # Setup mocks
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # 1KB file
        mock_open.return_value.__enter__.return_value.read.return_value = self.test_image_data
        
        # Create mock UI module with widgets
        mock_image_widget = MagicMock()
        mock_image_widget.value = None
        
        mock_status_widget = MagicMock()
        mock_status_widget.value = ""
        
        ui_module = MagicMock()
        ui_module._ui_components = {
            'form_container': {
                'widgets': {
                    'preview_widget': {
                        'widgets': {
                            'preview_image': mock_image_widget,
                            'preview_status': mock_status_widget
                        }
                    }
                }
            }
        }
        
        # Create operation and test
        operation = AugmentPreviewOperation(
            ui_module=ui_module,
            config={'test': 'config'},
            callbacks={}
        )
        
        # Test loading
        result = operation._load_preview_to_widget(self.test_preview_path)
        
        # Verify calls
        mock_exists.assert_called_with(self.test_preview_path)
        mock_getsize.assert_called_with(self.test_preview_path)
        mock_open.assert_called_with(self.test_preview_path, 'rb')
        
        # Verify widget updates
        assert mock_image_widget.value == self.test_image_data
        assert "Preview loaded" in mock_status_widget.value
    
    def test_preview_image_loading_file_not_found(self):
        """Test preview loading when file doesn't exist."""
        ui_module = MagicMock()
        
        operation = AugmentPreviewOperation(
            ui_module=ui_module,
            config={},
            callbacks={}
        )
        
        # Test with non-existent file
        result = operation._load_preview_to_widget("/nonexistent/path.jpg")
        assert result == False, "Should return False for non-existent file"
    
    def test_widget_access_methods(self):
        """Test widget access helper methods."""
        # Create realistic UI component structure
        mock_image_widget = MagicMock()
        mock_status_widget = MagicMock()
        
        ui_module = MagicMock()
        ui_module._ui_components = {
            'form_container': {
                'widgets': {
                    'preview_widget': {
                        'widgets': {
                            'preview_image': mock_image_widget,
                            'preview_status': mock_status_widget
                        }
                    }
                }
            }
        }
        
        operation = AugmentPreviewOperation(
            ui_module=ui_module,
            config={},
            callbacks={}
        )
        
        # Test widget access
        image_widget = operation._get_preview_image_widget()
        status_widget = operation._get_preview_status_widget()
        
        assert image_widget == mock_image_widget, "Should find image widget"
        assert status_widget == mock_status_widget, "Should find status widget"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])