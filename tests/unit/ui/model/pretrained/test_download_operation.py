"""
File: tests/unit/ui/model/pretrained/test_download_operation.py
Tests for DownloadOperation handler.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.ui.model.pretrained.operations.download_operation import DownloadOperation
from smartcash.ui.model.pretrained.constants import PretrainedOperation


class TestDownloadOperation:
    """Test suite for DownloadOperation class."""
    
    @pytest.fixture
    def operation(self):
        """Create DownloadOperation instance for testing."""
        return DownloadOperation()
    
    @pytest.fixture
    def mock_ui_components(self):
        """Create mock UI components."""
        progress_tracker = Mock()
        progress_tracker.update_progress = Mock()
        
        log_output = Mock()
        log_output.log = Mock()
        
        return {
            'progress_tracker': progress_tracker,
            'log_output': log_output
        }
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            'models_dir': '/test/pretrained',
            'model_urls': {
                'yolov5s': 'https://test.url/yolov5s.pt',
                'efficientnet_b4': 'https://test.url/efficientnet.pth'
            }
        }
    
    def test_operation_initialization(self, operation):
        """Test operation initialization."""
        assert operation is not None
        assert hasattr(operation, 'service')
        assert operation.operation_type == PretrainedOperation.DOWNLOAD.value
        assert hasattr(operation, 'progress_steps')
        assert len(operation.progress_steps) > 0
    
    def test_initialize_method(self, operation):
        """Test the initialize method (required by base class)."""
        # Should not raise any exception
        operation.initialize()
    
    @pytest.mark.asyncio
    async def test_execute_operation_success(self, operation, mock_ui_components, sample_config):
        """Test successful operation execution."""
        # Mock service methods
        mock_existing_check = {
            'models_dir': '/test/pretrained',
            'models_found': [],
            'models_missing': [
                {'model_type': 'yolov5s', 'name': 'YOLOv5s'},
                {'model_type': 'efficientnet_b4', 'name': 'EfficientNet-B4'}
            ],
            'total_found': 0,
            'all_present': False
        }
        
        mock_download_results = {
            'all_successful': True,
            'success_count': 2,
            'total_count': 2,
            'downloads': [
                {'model': 'YOLOv5s', 'success': True},
                {'model': 'EfficientNet-B4', 'success': True}
            ]
        }
        
        mock_summary = {
            'models_count': 2,
            'total_size_mb': 89.4,
            'available_models': [
                {'name': 'YOLOv5s', 'type': 'yolov5s', 'size_mb': 14.4},
                {'name': 'EfficientNet-B4', 'type': 'efficientnet_b4', 'size_mb': 75.0}
            ]
        }
        
        with patch.object(operation.service, 'check_existing_models', return_value=mock_existing_check) as mock_check, \
             patch.object(operation.service, 'download_all_models', return_value=mock_download_results) as mock_download, \
             patch.object(operation.service, 'get_models_summary', return_value=mock_summary) as mock_summary_call:
            
            result = await operation.execute_operation(
                sample_config, 
                mock_ui_components
            )
        
        # Verify service methods were called
        mock_check.assert_called_once()
        mock_download.assert_called_once()
        mock_summary_call.assert_called_once()
        
        # Verify result structure
        assert result['operation'] == PretrainedOperation.DOWNLOAD.value
        assert result['success'] is True
        assert result['models_dir'] == '/test/pretrained'
        assert 'existing_models' in result
        assert 'download_results' in result
        assert 'summary' in result
        
        # Verify UI callbacks were called
        mock_ui_components['progress_tracker'].update_progress.assert_called()
        mock_ui_components['log_output'].log.assert_called()
    
    @pytest.mark.asyncio
    async def test_execute_operation_partial_success(self, operation, mock_ui_components, sample_config):
        """Test operation execution with partial success."""
        mock_existing_check = {
            'models_dir': '/test/pretrained',
            'models_found': [
                {'model_type': 'yolov5s', 'name': 'YOLOv5s', 'file_size_mb': 14.4}
            ],
            'models_missing': [
                {'model_type': 'efficientnet_b4', 'name': 'EfficientNet-B4'}
            ],
            'total_found': 1,
            'all_present': False
        }
        
        mock_download_results = {
            'all_successful': False,
            'success_count': 1,
            'total_count': 2,
            'downloads': [
                {'model': 'YOLOv5s', 'success': True},
                {'model': 'EfficientNet-B4', 'success': False}
            ]
        }
        
        mock_summary = {
            'models_count': 1,
            'total_size_mb': 14.4,
            'available_models': [
                {'name': 'YOLOv5s', 'type': 'yolov5s', 'size_mb': 14.4}
            ]
        }
        
        with patch.object(operation.service, 'check_existing_models', return_value=mock_existing_check), \
             patch.object(operation.service, 'download_all_models', return_value=mock_download_results), \
             patch.object(operation.service, 'get_models_summary', return_value=mock_summary):
            
            result = await operation.execute_operation(
                sample_config,
                mock_ui_components
            )
        
        # Verify partial success handling
        assert result['success'] is False  # Overall success should be False for partial success
        assert result['download_results']['success_count'] == 1
        assert result['download_results']['total_count'] == 2
        
        # Verify warning message was logged
        mock_ui_components['log_output'].log.assert_called()
        log_calls = [call[0][0] for call in mock_ui_components['log_output'].log.call_args_list]
        warning_logged = any("⚠️" in call for call in log_calls)
        assert warning_logged
    
    @pytest.mark.asyncio
    async def test_execute_operation_error(self, operation, mock_ui_components, sample_config):
        """Test operation execution with error."""
        # Mock service to raise exception
        with patch.object(operation.service, 'check_existing_models', side_effect=Exception("Test error")):
            result = await operation.execute_operation(
                sample_config,
                mock_ui_components
            )
        
        # Verify error handling
        assert result['operation'] == PretrainedOperation.DOWNLOAD.value
        assert result['success'] is False
        assert 'error' in result
        assert "Test error" in result['error']
        assert result['models_dir'] == '/test/pretrained'
        
        # Verify error was logged
        mock_ui_components['log_output'].log.assert_called()
        log_calls = [call[0][0] for call in mock_ui_components['log_output'].log.call_args_list]
        error_logged = any("❌" in call for call in log_calls)
        assert error_logged
    
    def test_get_progress_callback_valid(self, operation, mock_ui_components):
        """Test getting valid progress callback."""
        callback = operation._get_progress_callback(mock_ui_components)
        assert callback is not None
        assert callable(callback)
        assert callback == mock_ui_components['progress_tracker'].update_progress
    
    def test_get_progress_callback_missing(self, operation):
        """Test getting progress callback when component is missing."""
        ui_components = {}
        callback = operation._get_progress_callback(ui_components)
        assert callback is None
    
    def test_get_progress_callback_no_method(self, operation):
        """Test getting progress callback when method is missing."""
        ui_components = {'progress_tracker': Mock(spec=[])}  # Mock without update_progress
        callback = operation._get_progress_callback(ui_components)
        assert callback is None
    
    def test_get_log_callback_valid(self, operation, mock_ui_components):
        """Test getting valid log callback."""
        callback = operation._get_log_callback(mock_ui_components)
        assert callback is not None
        assert callable(callback)
        assert callback == mock_ui_components['log_output'].log
    
    def test_get_log_callback_missing(self, operation):
        """Test getting log callback when component is missing."""
        ui_components = {}
        callback = operation._get_log_callback(ui_components)
        assert callback is None
    
    def test_get_log_callback_no_method(self, operation):
        """Test getting log callback when method is missing."""
        ui_components = {'log_output': Mock(spec=[])}  # Mock without log method
        callback = operation._get_log_callback(ui_components)
        assert callback is None
    
    @pytest.mark.asyncio
    async def test_execute_operation_no_callbacks(self, operation, sample_config):
        """Test operation execution without UI callbacks."""
        # UI components without progress tracker or log output
        ui_components = {}
        
        mock_existing_check = {
            'models_dir': '/test/pretrained',
            'models_found': [],
            'models_missing': [],
            'total_found': 0,
            'all_present': True
        }
        
        mock_download_results = {
            'all_successful': True,
            'success_count': 2,
            'total_count': 2
        }
        
        mock_summary = {'models_count': 2}
        
        with patch.object(operation.service, 'check_existing_models', return_value=mock_existing_check), \
             patch.object(operation.service, 'download_all_models', return_value=mock_download_results), \
             patch.object(operation.service, 'get_models_summary', return_value=mock_summary):
            
            result = await operation.execute_operation(sample_config, ui_components)
        
        # Should complete successfully even without callbacks
        assert result['success'] is True
        assert result['operation'] == PretrainedOperation.DOWNLOAD.value
    
    @pytest.mark.asyncio
    async def test_execute_operation_service_error_in_download(self, operation, mock_ui_components, sample_config):
        """Test operation when service download fails."""
        mock_existing_check = {
            'models_dir': '/test/pretrained',
            'models_found': [],
            'models_missing': [],
            'total_found': 0,
            'all_present': False
        }
        
        with patch.object(operation.service, 'check_existing_models', return_value=mock_existing_check), \
             patch.object(operation.service, 'download_all_models', side_effect=Exception("Download service error")), \
             patch.object(operation.service, 'get_models_summary', return_value={}):
            
            result = await operation.execute_operation(sample_config, mock_ui_components)
        
        assert result['success'] is False
        assert "Download service error" in result['error']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])