"""
File: tests/unit/ui/model/pretrained/test_pretrained_service.py
Comprehensive tests for PretrainedService class.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

import sys
sys.path.append('/Users/masdevid/Projects/smartcash')

from smartcash.ui.model.pretrained.services.pretrained_service import PretrainedService
from smartcash.ui.model.pretrained.constants import PretrainedModelType, DEFAULT_MODELS_DIR


class TestPretrainedService:
    """Test suite for PretrainedService class."""
    
    @pytest.fixture
    def service(self):
        """Create PretrainedService instance for testing."""
        return PretrainedService()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_callbacks(self):
        """Create mock callbacks for testing."""
        return {
            'progress_callback': Mock(),
            'log_callback': Mock()
        }
    
    def test_service_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert hasattr(service, 'executor')
        assert hasattr(service, 'download_sessions')
        assert service.download_sessions == {}
    
    @pytest.mark.asyncio
    async def test_check_existing_models_empty_directory(self, service, temp_dir, mock_callbacks):
        """Test checking models in empty directory."""
        result = await service.check_existing_models(
            temp_dir, 
            mock_callbacks['progress_callback'],
            mock_callbacks['log_callback']
        )
        
        assert result['models_dir'] == temp_dir
        assert result['total_found'] == 0
        assert len(result['models_found']) == 0
        assert len(result['models_missing']) == 2  # YOLOv5s + EfficientNet-B4
        assert result['all_present'] is False
        
        # Verify callbacks were called
        mock_callbacks['progress_callback'].assert_called()
        mock_callbacks['log_callback'].assert_called()
    
    @pytest.mark.asyncio
    async def test_check_existing_models_with_files(self, service, temp_dir, mock_callbacks):
        """Test checking models with existing files."""
        # Create mock model files
        yolo_file = Path(temp_dir) / "yolov5s.pt"
        efficientnet_file = Path(temp_dir) / "efficientnet_b4.pth"
        
        yolo_file.write_bytes(b"fake yolo model data" * 1000)  # ~20KB
        efficientnet_file.write_bytes(b"fake efficientnet data" * 1000)  # ~24KB
        
        result = await service.check_existing_models(
            temp_dir,
            mock_callbacks['progress_callback'], 
            mock_callbacks['log_callback']
        )
        
        assert result['total_found'] == 2
        assert len(result['models_found']) == 2
        assert len(result['models_missing']) == 0
        assert result['all_present'] is True
        
        # Check found models details
        found_models = {model['model_type']: model for model in result['models_found']}
        assert 'yolov5s' in found_models
        assert 'efficientnet_b4' in found_models
        assert found_models['yolov5s']['file_size'] > 0
        assert found_models['efficientnet_b4']['file_size'] > 0
    
    @pytest.mark.asyncio
    async def test_check_existing_models_partial(self, service, temp_dir, mock_callbacks):
        """Test checking models with partial files."""
        # Create only YOLOv5s file
        yolo_file = Path(temp_dir) / "yolov5s.pt"
        yolo_file.write_bytes(b"fake yolo model data" * 1000)
        
        result = await service.check_existing_models(
            temp_dir,
            mock_callbacks['progress_callback'],
            mock_callbacks['log_callback']
        )
        
        assert result['total_found'] == 1
        assert len(result['models_found']) == 1
        assert len(result['models_missing']) == 1
        assert result['all_present'] is False
        
        # Verify found model
        assert result['models_found'][0]['model_type'] == 'yolov5s'
        
        # Verify missing model
        assert result['models_missing'][0]['model_type'] == 'efficientnet_b4'
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_download_yolov5s_success(self, mock_get, service, temp_dir, mock_callbacks):
        """Test successful YOLOv5s download."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-length': '14400000'}  # 14.4MB
        mock_response.iter_content.return_value = [b'fake_model_data' * 100 for _ in range(100)]
        mock_get.return_value = mock_response
        
        # Mock file validation
        with patch.object(service, '_validate_model_file', return_value=True):
            result = await service.download_yolov5s(
                temp_dir,
                progress_callback=mock_callbacks['progress_callback'],
                log_callback=mock_callbacks['log_callback']
            )
        
        assert result is True
        mock_callbacks['progress_callback'].assert_called()
        mock_callbacks['log_callback'].assert_called()
        
        # Verify file was created
        expected_file = Path(temp_dir) / "yolov5s.pt"
        assert expected_file.exists()
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_download_yolov5s_network_error(self, mock_get, service, temp_dir, mock_callbacks):
        """Test YOLOv5s download with network error."""
        # Mock network error
        mock_get.side_effect = Exception("Network error")
        
        result = await service.download_yolov5s(
            temp_dir,
            progress_callback=mock_callbacks['progress_callback'],
            log_callback=mock_callbacks['log_callback']
        )
        
        assert result is False
        mock_callbacks['log_callback'].assert_called()
    
    @pytest.mark.asyncio
    @patch('timm.create_model')
    @patch('torch.save')
    async def test_download_efficientnet_via_timm_success(self, mock_torch_save, mock_create_model, service, temp_dir, mock_callbacks):
        """Test successful EfficientNet-B4 download via timm."""
        # Mock timm model creation
        mock_model = Mock()
        mock_model.state_dict.return_value = {'fake': 'state_dict'}
        mock_create_model.return_value = mock_model
        
        result = await service.download_efficientnet_b4(
            temp_dir,
            progress_callback=mock_callbacks['progress_callback'],
            log_callback=mock_callbacks['log_callback']
        )
        
        assert result is True
        mock_create_model.assert_called_once_with('efficientnet_b4', pretrained=True)
        mock_torch_save.assert_called_once()
        mock_callbacks['progress_callback'].assert_called()
        mock_callbacks['log_callback'].assert_called()
    
    @pytest.mark.asyncio
    @patch('timm.create_model')
    async def test_download_efficientnet_via_timm_error(self, mock_create_model, service, temp_dir, mock_callbacks):
        """Test EfficientNet-B4 download via timm with error."""
        # Mock timm error
        mock_create_model.side_effect = Exception("Timm error")
        
        result = await service.download_efficientnet_b4(
            temp_dir,
            progress_callback=mock_callbacks['progress_callback'],
            log_callback=mock_callbacks['log_callback']
        )
        
        assert result is False
        mock_callbacks['log_callback'].assert_called()
    
    @pytest.mark.asyncio
    @patch('requests.get')
    async def test_download_efficientnet_custom_url_success(self, mock_get, service, temp_dir, mock_callbacks):
        """Test EfficientNet-B4 download with custom URL."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers = {'content-length': '75000000'}  # 75MB
        mock_response.iter_content.return_value = [b'fake_model_data' * 100 for _ in range(100)]
        mock_get.return_value = mock_response
        
        # Mock file validation
        with patch.object(service, '_validate_model_file', return_value=True):
            result = await service.download_efficientnet_b4(
                temp_dir,
                custom_url="https://custom.url/efficientnet.pth",
                progress_callback=mock_callbacks['progress_callback'],
                log_callback=mock_callbacks['log_callback']
            )
        
        assert result is True
        mock_callbacks['progress_callback'].assert_called()
        mock_callbacks['log_callback'].assert_called()
    
    @pytest.mark.asyncio
    async def test_download_all_models_success(self, service, temp_dir, mock_callbacks):
        """Test downloading all models successfully."""
        config = {
            "models_dir": temp_dir,
            "model_urls": {}
        }
        
        # Mock individual download methods
        with patch.object(service, 'download_yolov5s', return_value=True) as mock_yolo, \
             patch.object(service, 'download_efficientnet_b4', return_value=True) as mock_efficient:
            
            result = await service.download_all_models(
                config,
                mock_callbacks['progress_callback'],
                mock_callbacks['log_callback']
            )
        
        assert result['all_successful'] is True
        assert result['success_count'] == 2
        assert result['total_count'] == 2
        assert len(result['downloads']) == 2
        
        mock_yolo.assert_called_once()
        mock_efficient.assert_called_once()
        mock_callbacks['progress_callback'].assert_called()
        mock_callbacks['log_callback'].assert_called()
    
    @pytest.mark.asyncio
    async def test_download_all_models_partial_failure(self, service, temp_dir, mock_callbacks):
        """Test downloading all models with partial failure."""
        config = {
            "models_dir": temp_dir,
            "model_urls": {}
        }
        
        # Mock partial success
        with patch.object(service, 'download_yolov5s', return_value=True) as mock_yolo, \
             patch.object(service, 'download_efficientnet_b4', return_value=False) as mock_efficient:
            
            result = await service.download_all_models(
                config,
                mock_callbacks['progress_callback'],
                mock_callbacks['log_callback']
            )
        
        assert result['all_successful'] is False
        assert result['success_count'] == 1
        assert result['total_count'] == 2
        
        # Check individual results
        downloads = {d['model']: d for d in result['downloads']}
        assert downloads['YOLOv5s']['success'] is True
        assert downloads['EfficientNet-B4']['success'] is False
    
    def test_validate_model_file_valid_yolo(self, service, temp_dir):
        """Test model file validation for valid YOLOv5s file."""
        # Create valid-sized file
        file_path = Path(temp_dir) / "yolov5s.pt"
        file_path.write_bytes(b"fake_model_data" * 900000)  # ~14.4MB
        
        result = service._validate_model_file(str(file_path), 'yolov5s')
        assert result is True
    
    def test_validate_model_file_too_small(self, service, temp_dir):
        """Test model file validation for file that's too small."""
        file_path = Path(temp_dir) / "yolov5s.pt"
        file_path.write_bytes(b"small")  # Very small file
        
        result = service._validate_model_file(str(file_path), 'yolov5s')
        assert result is False
    
    def test_validate_model_file_wrong_extension(self, service, temp_dir):
        """Test model file validation for wrong file extension."""
        file_path = Path(temp_dir) / "yolov5s.txt"  # Wrong extension
        file_path.write_bytes(b"fake_model_data" * 900000)
        
        result = service._validate_model_file(str(file_path), 'yolov5s')
        assert result is False
    
    def test_validate_model_file_nonexistent(self, service):
        """Test model file validation for nonexistent file."""
        result = service._validate_model_file("/nonexistent/file.pt", 'yolov5s')
        assert result is False
    
    def test_get_models_summary_empty_dir(self, service, temp_dir):
        """Test getting models summary for empty directory."""
        summary = service.get_models_summary(temp_dir)
        
        assert summary['models_dir'] == temp_dir
        assert summary['models_count'] == 0
        assert summary['total_size_mb'] == 0
        assert len(summary['available_models']) == 0
    
    def test_get_models_summary_with_models(self, service, temp_dir):
        """Test getting models summary with existing models."""
        # Create mock model files
        yolo_file = Path(temp_dir) / "yolov5s.pt"
        efficientnet_file = Path(temp_dir) / "efficientnet_b4.pth"
        
        yolo_file.write_bytes(b"fake yolo data" * 1000000)  # ~15MB
        efficientnet_file.write_bytes(b"fake efficient data" * 5000000)  # ~75MB
        
        summary = service.get_models_summary(temp_dir)
        
        assert summary['models_count'] == 2
        assert summary['total_size_mb'] > 0
        assert len(summary['available_models']) == 2
        
        # Check model details
        models = {model['type']: model for model in summary['available_models']}
        assert 'yolov5s' in models
        assert 'efficientnet_b4' in models
        assert models['yolov5s']['name'] == 'YOLOv5s'
        assert models['efficientnet_b4']['name'] == 'EfficientNet-B4'
    
    def test_get_models_summary_nonexistent_dir(self, service):
        """Test getting models summary for nonexistent directory."""
        summary = service.get_models_summary("/nonexistent/directory")
        
        assert summary['models_count'] == 0
        assert summary['total_size_mb'] == 0
        assert len(summary['available_models']) == 0
    
    @pytest.mark.asyncio
    async def test_check_existing_models_error_handling(self, service, mock_callbacks):
        """Test error handling in check_existing_models."""
        # Use invalid directory path to trigger error
        with pytest.raises(Exception):
            await service.check_existing_models(
                "/invalid/path/that/cannot/be/created",
                mock_callbacks['progress_callback'],
                mock_callbacks['log_callback']
            )
    
    def test_download_file_with_progress_mock(self, service, temp_dir, mock_callbacks):
        """Test _download_file_with_progress method with mocking."""
        with patch('requests.get') as mock_get:
            # Mock successful response
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.headers = {'content-length': '1000'}
            mock_response.iter_content.return_value = [b'data' * 250 for _ in range(4)]
            mock_get.return_value = mock_response
            
            file_path = str(Path(temp_dir) / "test_file.pt")
            result = service._download_file_with_progress(
                "https://test.url/file.pt",
                file_path,
                mock_callbacks['progress_callback'],
                mock_callbacks['log_callback']
            )
            
            assert result is True
            assert Path(file_path).exists()
            mock_callbacks['progress_callback'].assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])