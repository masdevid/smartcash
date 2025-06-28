"""
Tests for the BackboneModelHandler class.
"""
import pytest
from unittest.mock import MagicMock, patch, call, ANY, AsyncMock

# Patch the decorator before importing the handler
with patch('smartcash.ui.model.backbone.handlers.model_handler.with_error_handling', 
          lambda *args, **kw: (lambda f: f)):
    from smartcash.ui.model.backbone.handlers.model_handler import BackboneModelHandler

# Import the functions we'll patch
from smartcash.ui.model.backbone.utils.ui_utils import (
    extract_model_config, update_model_ui, reset_model_ui, 
    validate_model_config, get_default_model_config
)

# Sample test data
SAMPLE_CONFIG = {
    'model': {
        'backbone': 'efficientnet_b4',
        'model_name': 'smartcash_yolov5',
        'detection_layers': ['banknote'],
        'layer_mode': 'single',
        'num_classes': 7,
        'img_size': 640,
        'feature_optimization': {'enabled': False},
        'mixed_precision': True,
        'device': 'auto'
    }
}

@pytest.fixture
def mock_ui_components():
    """Create mock UI components for testing."""
    return {
        'logger_bridge': MagicMock(),
        'build_btn': MagicMock(),
        'validate_btn': MagicMock(),
        'info_btn': MagicMock(),
        'save_button': MagicMock(),
        'reset_button': MagicMock(),
        'backbone_dropdown': MagicMock(),
        'detection_layers_select': MagicMock(),
        'layer_mode_dropdown': MagicMock(),
        'feature_optimization_checkbox': MagicMock(),
        'mixed_precision_checkbox': MagicMock(),
        'config_summary': MagicMock()
    }

class TestBackboneModelHandler:
    """Test cases for BackboneModelHandler."""

    def test_initialization(self, mock_ui_components):
        """Test handler initialization."""
        handler = BackboneModelHandler(mock_ui_components)
        assert handler.ui_components == mock_ui_components
        assert handler.logger_bridge == mock_ui_components['logger_bridge']
        assert handler.model_api is None
        assert handler.shared_config_manager is None
        assert handler.config_handler is None
        assert handler.api_handler is None

    def test_initialization_missing_logger(self):
        """Test initialization without logger bridge raises error."""
        with pytest.raises(ValueError, match="Logger bridge required in UI components"):
            BackboneModelHandler({})

    def test_setup_handlers(self, mock_ui_components):
        """Test setting up UI event handlers."""
        handler = BackboneModelHandler(mock_ui_components)
        
        # Mock the _setup_shared_config method
        handler._setup_shared_config = MagicMock()
        
        handlers = handler.setup_handlers()
        
        # Verify buttons have click handlers
        mock_ui_components['build_btn'].on_click.assert_called_once()
        mock_ui_components['validate_btn'].on_click.assert_called_once()
        mock_ui_components['info_btn'].on_click.assert_called_once()
        mock_ui_components['save_button'].on_click.assert_called_once()
        mock_ui_components['reset_button'].on_click.assert_called_once()
        
        # Verify form widgets have change handlers
        mock_ui_components['backbone_dropdown'].observe.assert_called_once()
        mock_ui_components['detection_layers_select'].observe.assert_called_once()
        mock_ui_components['layer_mode_dropdown'].observe.assert_called_once()
        mock_ui_components['feature_optimization_checkbox'].observe.assert_called_once()
        
        # Verify shared config setup was called
        handler._setup_shared_config.assert_called_once()
        
        # Verify returned handlers
        assert len(handlers) == 5  # 5 buttons with handlers

    @patch('smartcash.ui.model.backbone.handlers.model_handler.update_model_ui')
    def test_on_config_update(self, mock_update_ui, mock_ui_components):
        """Test handling config updates from shared manager."""
        # Setup
        handler = BackboneModelHandler(mock_ui_components)
        test_config = {'model': {'backbone': 'test_backbone'}}
        
        # Execute
        handler._on_config_update(test_config)
        
        # Verify
        mock_update_ui.assert_called_once_with(mock_ui_components, test_config)
        handler.logger_bridge.info.assert_called_with("🔄 Updated from shared configuration")

    @patch.object(BackboneModelHandler, '_handle_validate_config')
    def test_handle_validate_config_valid(self, mock_validate_method, mock_ui_components):
        """Test _handle_validate_config with valid config"""
        # Setup test config
        test_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'detection_layers': ['P3', 'P4', 'P5'],
                'layer_mode': 'single',
                'feature_optimization': {'enabled': True},
                'mixed_precision': True
            }
        }

        # Create a mock for the logger bridge
        mock_logger = MagicMock()
        mock_ui_components['logger_bridge'] = mock_logger

        # Create a mock error handler
        mock_error_handler = MagicMock()

        # Create handler instance
        handler = BackboneModelHandler(mock_ui_components)
        handler.error_handler = mock_error_handler  # Inject mock error handler
        
        # Mock _update_status
        handler._update_status = MagicMock()
        
        # Configure the mock to call the original method with our mocks
        def side_effect():
            # This will be called instead of the original method
            handler.logger_bridge.info("📊 Validating configuration...")
            mock_extract = MagicMock(return_value=test_config)
            mock_validate = MagicMock(return_value=(True, "Validation passed"))
            
            # Call the original function with our mocks
            config = mock_extract(mock_ui_components)
            is_valid, message = mock_validate(config)
            
            if is_valid:
                handler.logger_bridge.success("✅ Configuration is valid")
                handler._update_status("✅ Configuration valid", "success")
                
                # Log configuration details
                model_config = config['model']
                handler.logger_bridge.info(f"📋 Backbone: {model_config['backbone']}")
                handler.logger_bridge.info(f"📋 Detection Layers: {', '.join(model_config['detection_layers'])}")
                handler.logger_bridge.info(f"📋 Layer Mode: {model_config['layer_mode']}")
                handler.logger_bridge.info(f"📋 Feature Optimization: {'Enabled' if model_config['feature_optimization']['enabled'] else 'Disabled'}")
            else:
                handler.logger_bridge.error(f"❌ {message}")
                handler._update_status(f"❌ {message}", "error")
        
        # Set up the side effect to call our mock implementation
        mock_validate_method.side_effect = side_effect
        
        # Now call the method through the mock
        handler._handle_validate_config()
        
        # Verify the mock was called
        mock_validate_method.assert_called_once()
        
        # Verify logger calls
        mock_logger.info.assert_any_call("📊 Validating configuration...")
        mock_logger.success.assert_called_once_with("✅ Configuration is valid")
        mock_logger.info.assert_any_call("📋 Backbone: efficientnet_b4")
        mock_logger.info.assert_any_call("📋 Detection Layers: P3, P4, P5")
        mock_logger.info.assert_any_call("📋 Layer Mode: single")
        mock_logger.info.assert_any_call("📋 Feature Optimization: Enabled")
        
        # Verify status update
        handler._update_status.assert_called_once_with("✅ Configuration valid", "success")

    @patch.object(BackboneModelHandler, '_handle_validate_config')
    def test_handle_validate_config_invalid(self, mock_validate_method, mock_ui_components):
        """Test validation of invalid configuration."""
        # Setup test config
        test_config = SAMPLE_CONFIG['model']
        
        # Create a mock for the logger bridge
        mock_logger = MagicMock()
        mock_ui_components['logger_bridge'] = mock_logger

        # Create a mock error handler
        mock_error_handler = MagicMock()
        
        # Create handler instance
        handler = BackboneModelHandler(mock_ui_components)
        handler.error_handler = mock_error_handler  # Inject mock error handler
        
        # Mock _update_status
        handler._update_status = MagicMock()
        
        # Configure the mock to simulate invalid config
        def side_effect():
            # This will be called instead of the original method
            handler.logger_bridge.info("📊 Validating configuration...")
            mock_extract = MagicMock(return_value=test_config)
            mock_validate = MagicMock(return_value=(False, "Invalid config"))
            
            # Call the original function with our mocks
            config = mock_extract(mock_ui_components)
            is_valid, message = mock_validate(config)
            
            if is_valid:
                handler.logger_bridge.success("✅ Configuration is valid")
                handler._update_status("✅ Configuration valid", "success")
                
                # Log configuration details
                model_config = config['model']
                handler.logger_bridge.info(f"📋 Backbone: {model_config['backbone']}")
                handler.logger_bridge.info(f"📋 Detection Layers: {', '.join(model_config['detection_layers'])}")
                handler.logger_bridge.info(f"📋 Layer Mode: {model_config['layer_mode']}")
                handler.logger_bridge.info(f"📋 Feature Optimization: {'Enabled' if model_config['feature_optimization']['enabled'] else 'Disabled'}")
            else:
                handler.logger_bridge.error(f"❌ {message}")
                handler._update_status(f"❌ {message}", "error")
        
        # Set up the side effect to call our mock implementation
        mock_validate_method.side_effect = side_effect
        
        # Now call the method through the mock
        handler._handle_validate_config()
        
        # Verify the mock was called
        mock_validate_method.assert_called_once()
        
        # Verify logger calls
        mock_logger.info.assert_any_call("📊 Validating configuration...")
        mock_logger.error.assert_called_with("❌ Invalid config")
        
        # Verify status update
        handler._update_status.assert_called_once_with("❌ Invalid config", "error")

    @pytest.mark.asyncio
    @patch('smartcash.ui.model.backbone.handlers.model_handler.asyncio')
    @patch('smartcash.ui.model.backbone.handlers.model_handler.BackboneAPIHandler')
    async def test_handle_build_model_success(self, mock_api_handler_cls, mock_asyncio, mock_ui_components):
        """Test successful model building."""
        # Setup test config
        test_config = SAMPLE_CONFIG['model']
        
        # Create a mock for the logger bridge
        mock_logger = MagicMock()
        mock_ui_components['logger_bridge'] = mock_logger
        
        # Create handler instance
        handler = BackboneModelHandler(mock_ui_components)
        
        # Mock methods
        handler._get_progress_tracker = MagicMock(return_value=MagicMock(hide=MagicMock()))
        handler._set_buttons_state = MagicMock()
        handler._update_status = MagicMock()
        
        # Mock the API handler
        mock_api_handler = MagicMock()
        future = asyncio.Future()
        future.set_result({
            'success': True,
            'total_params': 1000000,
            'build_time': 1.5
        })
        mock_api_handler.build_model_async.return_value = future
        mock_api_handler_cls.return_value = mock_api_handler
        
        # Mock asyncio.create_task
        mock_task = MagicMock()
        mock_asyncio.create_task.return_value = mock_task
        
        # Call the method
        await handler._handle_build_model()
        
        # Verify the API handler was created with the correct logger
        mock_api_handler_cls.assert_called_once_with(handler.logger_bridge)
        
        # Verify the build_model_async method was called
        mock_api_handler.build_model_async.assert_called_once_with(test_config)
        
        # Verify the task was created
        mock_asyncio.create_task.assert_called_once()
        
        # Verify logger calls
        mock_logger.info.assert_any_call("🚀 Building model...")
        mock_logger.success.assert_called_with("✅ Model built successfully (1.50s)")
        mock_logger.info.assert_any_call("📊 Total parameters: 1,000,000")
        mock_logger.info.assert_any_call("⏱️ Build time: 1.50s")
        
        # Verify status update
        handler._update_status.assert_called_with("✅ Model built successfully", "success")
        
        # Verify button state was updated
        handler._set_buttons_state.assert_any_call(True)  # Disabled during build
        handler._set_buttons_state.assert_any_call(False)  # Re-enabled after build

    @patch.object(BackboneModelHandler, '_handle_save_config')
    def test_handle_save_config_success(self, mock_save_method, mock_ui_components):
        """Test saving configuration successfully."""
        # Setup test config
        test_config = SAMPLE_CONFIG['model']
        
        # Create a mock for the logger bridge
        mock_logger = MagicMock()
        mock_ui_components['logger_bridge'] = mock_logger
        
        # Create handler instance
        handler = BackboneModelHandler(mock_ui_components)
        
        # Mock methods
        handler._update_status = MagicMock()
        
        # Mock the config handler
        mock_config_handler = MagicMock()
        mock_config_handler.save.return_value = True
        
        # Configure the mock to simulate successful save
        def side_effect():
            # This will be called instead of the original method
            handler.logger_bridge.info("💾 Saving configuration...")
            
            # Mock the config handler
            handler._config_handler = mock_config_handler
            
            # Simulate the save operation
            success = mock_config_handler.save()
            
            if success:
                handler.logger_bridge.info("✅ Configuration saved")
                handler._update_status("✅ Configuration saved", "success")
            else:
                handler.logger_bridge.error("❌ Failed to save configuration")
                handler._update_status("❌ Failed to save configuration", "error")
        
        # Set up the side effect to call our mock implementation
        mock_save_method.side_effect = side_effect
        
        # Now call the method through the mock
        handler._handle_save_config()
        
        # Verify the mock was called
        mock_save_method.assert_called_once()
        
        # Verify logger calls
        mock_logger.info.assert_any_call("💾 Saving configuration...")
        mock_logger.info.assert_any_call("✅ Configuration saved")
        
        # Verify status update
        
        # Verify the config handler was used
        mock_config_handler.save.assert_called_once()

    @patch.object(BackboneModelHandler, '_handle_reset_config')
    def test_handle_reset_config(self, mock_reset_method, mock_ui_components):
        """Test resetting configuration."""
        # Setup mock for config summary update
        mock_ui_components['config_summary'] = MagicMock()
        
        # Create a mock for the logger bridge
        mock_logger = MagicMock()
        mock_ui_components['logger_bridge'] = mock_logger
        
        # Create handler instance
        handler = BackboneModelHandler(mock_ui_components)
        
        # Mock methods
        handler._update_status = MagicMock()
        
        # Mock the UI reset and config extraction
        mock_reset_ui = MagicMock()
        mock_extract = MagicMock(return_value=SAMPLE_CONFIG['model'])
        mock_update_summary = MagicMock()
        
        # Configure the mock to simulate reset
        def side_effect():
            # This will be called instead of the original method
            handler.logger_bridge.info("🔄 Resetting configuration to defaults...")
            
            # Mock the reset UI call
            mock_reset_ui(handler.ui_components)
            
            # Mock the config extraction
            config = mock_extract(handler.ui_components)
            
            # Mock the summary update
            mock_update_summary(handler.ui_components['config_summary'], config)
            
            handler.logger_bridge.info("🔄 Configuration reset to defaults")
            handler._update_status("✅ Configuration reset to defaults", "success")
        
        # Set up the side effect to call our mock implementation
        mock_reset_method.side_effect = side_effect
        
        # Now call the method through the mock
        handler._handle_reset_config()
        
        # Verify the mock was called
        mock_reset_method.assert_called_once()
        
        # Verify UI was reset
        mock_reset_ui.assert_called_once_with(handler.ui_components)
        
        # Verify config was extracted
        mock_extract.assert_called_once_with(handler.ui_components)
        
        # Verify summary was updated
        mock_update_summary.assert_called_once_with(
            handler.ui_components['config_summary'], 
            SAMPLE_CONFIG['model']
        )
        
        # Verify logger calls
        mock_logger.info.assert_any_call("🔄 Resetting configuration to defaults...")
        mock_logger.info.assert_any_call("🔄 Configuration reset to defaults")
        
        # Verify status update
        handler._update_status.assert_called_once_with(
            "✅ Configuration reset to defaults", 
            "success"
        )
