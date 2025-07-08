"""
File: tests/unit/ui/model/pretrained/test_pretrained_integration.py
Integration tests for the complete pretrained module.
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.ui.model.pretrained.pretrained_initializer import PretrainedInitializer, _pretrained_initialize_legacy
from smartcash.ui.model.pretrained.constants import DEFAULT_CONFIG


class TestPretrainedIntegration:
    """Integration tests for pretrained module."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create test configuration."""
        return {
            'models_dir': temp_dir,
            'model_urls': {
                'yolov5s': 'https://test.url/yolov5s.pt',
                'efficientnet_b4': ''  # Use timm
            }
        }
    
    def test_initializer_creation(self):
        """Test initializer can be created."""
        initializer = PretrainedInitializer()
        assert initializer is not None
        assert initializer.module_name == "pretrained"
        assert initializer.parent_module == "model"
        assert hasattr(initializer, 'config_handler')
        assert hasattr(initializer, 'service')
    
    @patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui')
    def test_initialize_impl_success(self, mock_create_ui, test_config):
        """Test successful module initialization."""
        # Mock UI components creation
        mock_ui_components = {
            'ui': Mock(),
            'download_button': Mock(),
            'progress_tracker': Mock(),
            'log_output': Mock(),
            'input_options': {
                'model_dir_input': Mock(),
                'yolo_url_input': Mock(),
                'efficientnet_url_input': Mock()
            }
        }
        mock_create_ui.return_value = mock_ui_components
        
        initializer = PretrainedInitializer()
        
        # Mock UI handler creation
        with patch('smartcash.ui.model.pretrained.handlers.pretrained_ui_handler.PretrainedUIHandler') as mock_handler_class:
            mock_handler = Mock()
            mock_handler_class.return_value = mock_handler
            
            result = initializer._initialize_impl(test_config)
        
        # Verify successful initialization
        assert 'ui' in result
        assert 'ui_handler' in result
        assert result['ui_handler'] == mock_handler
        
        # Verify UI creation was called
        mock_create_ui.assert_called_once()
        
        # Verify handler was created with UI components
        mock_handler_class.assert_called_once()
        call_args = mock_handler_class.call_args[0]
        assert 'ui' in call_args[0]
        assert 'download_button' in call_args[0]
    
    def test_initialize_impl_error_handling(self, test_config):
        """Test error handling in initialization."""
        initializer = PretrainedInitializer()
        
        # Mock UI creation to raise exception
        with patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui', side_effect=Exception("UI creation failed")):
            result = initializer._initialize_impl(test_config)
        
        # Verify error response
        assert 'error' in result
        assert result['error'] is True
        assert 'ui' in result  # Should have error UI component
        assert 'error_message' in result
        assert "Failed to initialize pretrained module" in result['error_message']
    
    @patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui')
    @patch('smartcash.ui.model.pretrained.handlers.pretrained_ui_handler.PretrainedUIHandler')
    @pytest.mark.asyncio
    async def test_post_init_check_all_models_present(self, mock_handler_class, mock_create_ui, test_config):
        """Test post-init check when all models are present."""
        # Mock UI components
        log_output = Mock()
        log_output.log = Mock()
        
        mock_ui_components = {
            'ui': Mock(),
            'log_output': log_output,
            'progress_tracker': Mock(),
            'input_options': {}
        }
        mock_create_ui.return_value = mock_ui_components
        
        # Mock UI handler
        mock_handler = Mock()
        mock_handler.check_models_status = AsyncMock(return_value={
            'total_found': 2,
            'all_present': True,
            'models_found': [
                {'name': 'YOLOv5s', 'file_size_mb': 14.4},
                {'name': 'EfficientNet-B4', 'file_size_mb': 75.0}
            ],
            'models_missing': []
        })
        mock_handler_class.return_value = mock_handler
        
        initializer = PretrainedInitializer()
        
        # Wait a bit to let the async post-init check complete
        result = initializer._initialize_impl(test_config)
        await asyncio.sleep(0.1)  # Allow async task to complete
        
        # Verify handler method was called
        mock_handler.check_models_status.assert_called_once()
        
        # Verify success message was logged
        log_output.log.assert_called()
        log_calls = [call[0][0] for call in log_output.log.call_args_list]
        success_logged = any("✅ All pretrained models are available" in call for call in log_calls)
        assert success_logged
    
    @patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui')
    @patch('smartcash.ui.model.pretrained.handlers.pretrained_ui_handler.PretrainedUIHandler')
    @pytest.mark.asyncio
    async def test_post_init_check_no_models(self, mock_handler_class, mock_create_ui, test_config):
        """Test post-init check when no models are present."""
        # Mock UI components
        log_output = Mock()
        log_output.log = Mock()
        
        mock_ui_components = {
            'ui': Mock(),
            'log_output': log_output,
            'progress_tracker': Mock(),
            'input_options': {}
        }
        mock_create_ui.return_value = mock_ui_components
        
        # Mock UI handler
        mock_handler = Mock()
        mock_handler.check_models_status = AsyncMock(return_value={
            'total_found': 0,
            'all_present': False,
            'models_found': [],
            'models_missing': [
                {'name': 'YOLOv5s'},
                {'name': 'EfficientNet-B4'}
            ]
        })
        mock_handler_class.return_value = mock_handler
        
        initializer = PretrainedInitializer()
        
        # Wait for async post-init check
        result = initializer._initialize_impl(test_config)
        await asyncio.sleep(0.1)
        
        # Verify appropriate message was logged
        log_output.log.assert_called()
        log_calls = [call[0][0] for call in log_output.log.call_args_list]
        no_models_logged = any("📋 No pretrained models found" in call for call in log_calls)
        assert no_models_logged
    
    @patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui')
    @patch('smartcash.ui.model.pretrained.handlers.pretrained_ui_handler.PretrainedUIHandler')
    @pytest.mark.asyncio
    async def test_post_init_check_partial_models(self, mock_handler_class, mock_create_ui, test_config):
        """Test post-init check with partial models."""
        # Mock UI components
        log_output = Mock()
        log_output.log = Mock()
        
        mock_ui_components = {
            'ui': Mock(),
            'log_output': log_output,
            'progress_tracker': Mock(),
            'input_options': {}
        }
        mock_create_ui.return_value = mock_ui_components
        
        # Mock UI handler
        mock_handler = Mock()
        mock_handler.check_models_status = AsyncMock(return_value={
            'total_found': 1,
            'all_present': False,
            'models_found': [
                {'name': 'YOLOv5s', 'file_size_mb': 14.4}
            ],
            'models_missing': [
                {'name': 'EfficientNet-B4'}
            ]
        })
        mock_handler_class.return_value = mock_handler
        
        initializer = PretrainedInitializer()
        
        # Wait for async post-init check
        result = initializer._initialize_impl(test_config)
        await asyncio.sleep(0.1)
        
        # Verify partial status message was logged
        log_output.log.assert_called()
        log_calls = [call[0][0] for call in log_output.log.call_args_list]
        partial_logged = any("📋 Found 1/2 pretrained models" in call for call in log_calls)
        assert partial_logged
    
    @patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui')
    @patch('smartcash.ui.model.pretrained.handlers.pretrained_ui_handler.PretrainedUIHandler')
    @pytest.mark.asyncio
    async def test_post_init_check_error_handling(self, mock_handler_class, mock_create_ui, test_config):
        """Test post-init check error handling."""
        # Mock UI components
        log_output = Mock()
        log_output.log = Mock()
        
        mock_ui_components = {
            'ui': Mock(),
            'log_output': log_output,
            'progress_tracker': Mock(),
            'input_options': {}
        }
        mock_create_ui.return_value = mock_ui_components
        
        # Mock UI handler to raise exception
        mock_handler = Mock()
        mock_handler.check_models_status = AsyncMock(side_effect=Exception("Check failed"))
        mock_handler_class.return_value = mock_handler
        
        initializer = PretrainedInitializer()
        
        # Wait for async post-init check
        result = initializer._initialize_impl(test_config)
        await asyncio.sleep(0.1)
        
        # Verify warning was logged
        log_output.log.assert_called()
        log_calls = [call[0][0] for call in log_output.log.call_args_list]
        warning_logged = any("⚠️ Warning: Could not check models status" in call for call in log_calls)
        assert warning_logged
    
    def test_legacy_function_wrapper(self, test_config):
        """Test legacy function wrapper."""
        with patch.object(PretrainedInitializer, '_initialize_impl') as mock_init:
            mock_init.return_value = {'ui': Mock(), 'success': True}
            
            result = _pretrained_initialize_legacy(test_config, custom_param=True)
            
            # Verify initializer was called correctly
            mock_init.assert_called_once_with(test_config, custom_param=True)
            assert 'ui' in result
            assert result['success'] is True
    
    def test_create_error_response_with_error_component(self):
        """Test error response creation with error component."""
        initializer = PretrainedInitializer()
        
        with patch('smartcash.ui.components.error.error_component.create_error_component') as mock_error_component:
            mock_error_ui = Mock()
            mock_error_component.return_value = mock_error_ui
            
            result = initializer._create_error_response("Test error", "Detailed error info")
            
            assert result['ui'] == mock_error_ui
            assert result['error'] is True
            assert result['error_message'] == "Test error"
            
            # Verify error component was created correctly
            mock_error_component.assert_called_once_with(
                "Test error",
                "Detailed error info", 
                "Pretrained Models Error"
            )
    
    def test_create_error_response_fallback(self):
        """Test error response creation with fallback."""
        initializer = PretrainedInitializer()
        
        # Mock error component import to fail
        with patch('smartcash.ui.components.error.error_component.create_error_component', side_effect=ImportError()):
            result = initializer._create_error_response("Test error", "Detailed error info")
            
            assert 'ui' in result
            assert result['error'] is True
            assert result['error_message'] == "Test error"
            
            # Verify fallback HTML widget was created
            assert hasattr(result['ui'], 'value')  # HTML widget has value attribute
    
    def test_config_integration(self, test_config):
        """Test configuration integration throughout initialization."""
        initializer = PretrainedInitializer()
        
        with patch('smartcash.ui.model.pretrained.components.pretrained_ui.create_pretrained_ui') as mock_create_ui, \
             patch('smartcash.ui.model.pretrained.handlers.pretrained_ui_handler.PretrainedUIHandler') as mock_handler_class:
            
            mock_ui_components = {
                'ui': Mock(),
                'input_options': {
                    'model_dir_input': Mock(),
                    'yolo_url_input': Mock(),
                    'efficientnet_url_input': Mock()
                }
            }
            mock_create_ui.return_value = mock_ui_components
            mock_handler_class.return_value = Mock()
            
            result = initializer._initialize_impl(test_config)
            
            # Verify config was passed to UI creation
            create_ui_call_args = mock_create_ui.call_args[0][0]
            assert create_ui_call_args['models_dir'] == test_config['models_dir']
            assert 'yolov5s' in create_ui_call_args['model_urls']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])