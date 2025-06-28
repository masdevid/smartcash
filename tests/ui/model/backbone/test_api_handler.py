"""
Tests for BackboneAPIHandler class.
"""
import os
import pytest
from unittest.mock import MagicMock, patch, ANY
from smartcash.ui.model.backbone.handlers.api_handler import BackboneAPIHandler

class TestBackboneAPIHandler:
    """Test cases for BackboneAPIHandler."""
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()
    
    @pytest.fixture
    def mock_model_api(self):
        """Create a mock model API."""
        mock_api = MagicMock()
        mock_api.get_model_info.return_value = {'model_loaded': True}
        mock_api.list_checkpoints.return_value = [
            {'name': 'checkpoint1.pt', 'path': '/path/to/checkpoint1.pt'},
            {'name': 'checkpoint2.pt', 'path': '/path/to/checkpoint2.pt'}
        ]
        return mock_api
    
    def test_initialize_api_success(self, mock_logger):
        """Test successful API initialization."""
        with patch('smartcash.model.create_model_api') as mock_create_api:
            mock_create_api.return_value = MagicMock()
            
            handler = BackboneAPIHandler(mock_logger)
            result = handler.initialize_api()
            
            assert result is True
            assert handler.model_api is not None
            mock_logger.info.assert_called_with("🔧 Initializing Model API...")
            mock_logger.success.assert_called_with("✅ Model API initialized")
    
    def test_initialize_api_failure(self, mock_logger):
        """Test API initialization failure."""
        with patch('smartcash.model.create_model_api', side_effect=Exception("API Error")):
            handler = BackboneAPIHandler(mock_logger)
            result = handler.initialize_api()
            
            assert result is False
            assert handler.model_api is None
            mock_logger.error.assert_called_with("❌ Failed to initialize API: API Error")
    
    def test_build_model_async_success(self, mock_logger, mock_model_api):
        """Test successful async model build."""
        with patch('smartcash.model.create_model_api', return_value=mock_model_api):
            handler = BackboneAPIHandler(mock_logger)
            
            # Mock progress callback
            progress_callback = MagicMock()
            
            # Test config
            config = {
                'model': {
                    'backbone': 'efficientnet_b4',
                    'detection_layers': ['banknote'],
                    'layer_mode': 'single',
                    'num_classes': 7,
                    'img_size': 640,
                    'feature_optimization': {'enabled': False}
                }
            }
            
            # Mock build result
            mock_model_api.build_model.return_value = {'success': True, 'model_info': {}}
            
            result = handler.build_model_async(config, progress_callback)
            
            assert result['success'] is True
            mock_model_api.build_model.assert_called_once_with(
                backbone='efficientnet_b4',
                detection_layers=['banknote'],
                layer_mode='single',
                num_classes=7,
                img_size=640,
                feature_optimization={'enabled': False},
                progress_callback=progress_callback
            )
    
    def test_get_model_info_success(self, mock_logger, mock_model_api):
        """Test getting model info."""
        handler = BackboneAPIHandler(mock_logger)
        handler.model_api = mock_model_api
        
        result = handler.get_model_info()
        
        assert result == {'model_loaded': True}
        mock_model_api.get_model_info.assert_called_once()
    
    def test_validate_model_state_valid(self, mock_logger, mock_model_api):
        """Test model state validation with valid model."""
        handler = BackboneAPIHandler(mock_logger)
        handler.model_api = mock_model_api
        
        is_valid, message = handler.validate_model_state()
        
        assert is_valid is True
        assert message == "Model is ready"
    
    def test_list_available_checkpoints(self, mock_logger, mock_model_api):
        """Test listing available checkpoints."""
        handler = BackboneAPIHandler(mock_logger)
        handler.model_api = mock_model_api
        
        result = handler.list_available_checkpoints()
        
        assert len(result) == 2
        assert result[0]['name'] == 'checkpoint1.pt'
        assert result[1]['name'] == 'checkpoint2.pt'
    
    def test_load_checkpoint_success(self, mock_logger, mock_model_api):
        """Test loading a checkpoint successfully."""
        with patch('smartcash.model.create_model_api', return_value=mock_model_api):
            handler = BackboneAPIHandler(mock_logger)
            
            # Mock progress callback
            progress_callback = MagicMock()
            
            # Mock load result
            mock_model_api.load_checkpoint.return_value = {
                'success': True,
                'message': 'Checkpoint loaded successfully'
            }
            
            result = handler.load_checkpoint(
                '/path/to/checkpoint.pt',
                progress_callback=progress_callback
            )
            
            assert result['success'] is True
            mock_model_api.load_checkpoint.assert_called_once_with(
                '/path/to/checkpoint.pt',
                progress_callback=progress_callback
            )
    
    def test_cleanup(self, mock_logger):
        """Test cleanup of resources."""
        handler = BackboneAPIHandler(mock_logger)
        handler._is_building = True
        
        # Mock the executor's shutdown method
        handler.executor.shutdown = MagicMock()
        
        handler.cleanup()
        
        assert handler._is_building is False
        handler.executor.shutdown.assert_called_once_with(wait=False)
