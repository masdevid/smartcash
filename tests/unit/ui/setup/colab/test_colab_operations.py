"""
Test module for colab operations.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

from smartcash.ui.setup.colab.operations.base_operation import BaseOperation
from smartcash.ui.setup.colab.operations.environment_detection import EnvironmentDetectionOperation
from smartcash.ui.setup.colab.operations.drive_mount import DriveMountOperation
from smartcash.ui.setup.colab.operations.factory import OperationHandlerFactory


class TestBaseOperation:
    """Test cases for BaseOperation."""
    
    class MockOperation(BaseOperation):
        """Mock implementation of BaseOperation for testing."""
        
        async def _execute_impl(self) -> Dict[str, Any]:
            return {'success': True, 'message': 'Mock operation completed'}
    
    @pytest.fixture
    def base_operation(self):
        """Create a mock BaseOperation instance for testing."""
        ui_components = {'test': 'component'}
        config = {'test': 'config'}
        status_callback = Mock()
        
        return self.MockOperation(ui_components, config, status_callback)
    
    def test_init(self, base_operation):
        """Test BaseOperation initialization."""
        assert base_operation.ui_components == {'test': 'component'}
        assert base_operation.config == {'test': 'config'}
        assert base_operation.status_callback is not None
        assert base_operation._is_running is False
        assert base_operation._is_cancelled is False
    
    @pytest.mark.asyncio
    async def test_execute_success(self, base_operation):
        """Test successful operation execution."""
        result = await base_operation.execute()
        
        assert result['success'] is True
        assert result['message'] == 'Mock operation completed'
        assert base_operation._is_running is False
    
    @pytest.mark.asyncio
    async def test_execute_with_status_callback(self, base_operation):
        """Test operation execution with status callback."""
        await base_operation.execute()
        
        # Verify status callback was called
        base_operation.status_callback.assert_called()
    
    def test_cancel(self, base_operation):
        """Test operation cancellation."""
        base_operation.cancel()
        assert base_operation._is_cancelled is True
    
    def test_is_running(self, base_operation):
        """Test is_running property."""
        assert base_operation.is_running is False
        
        base_operation._is_running = True
        assert base_operation.is_running is True
    
    def test_is_cancelled(self, base_operation):
        """Test is_cancelled property."""
        assert base_operation.is_cancelled is False
        
        base_operation._is_cancelled = True
        assert base_operation.is_cancelled is True


class TestEnvironmentDetectionOperation:
    """Test cases for EnvironmentDetectionOperation."""
    
    @pytest.fixture
    def sample_ui_components(self):
        """Sample UI components for testing."""
        return {
            'environment_type_dropdown': Mock(),
            'project_name_text': Mock()
        }
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'environment': {
                'type': 'colab',
                'project_name': 'SmartCash'
            }
        }
    
    @pytest.fixture
    def env_detection_operation(self, sample_ui_components, sample_config):
        """Create an EnvironmentDetectionOperation instance for testing."""
        status_callback = Mock()
        return EnvironmentDetectionOperation(sample_ui_components, sample_config, status_callback)
    
    def test_init(self, env_detection_operation):
        """Test EnvironmentDetectionOperation initialization."""
        assert env_detection_operation is not None
        assert hasattr(env_detection_operation, 'ui_components')
        assert hasattr(env_detection_operation, 'config')
        assert hasattr(env_detection_operation, 'status_callback')
    
    @pytest.mark.asyncio
    async def test_execute_impl_colab(self, env_detection_operation):
        """Test environment detection for Colab."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.google.colab'):
            with patch('os.path.exists', return_value=False):
                result = await env_detection_operation._execute_impl()
                
                assert result['success'] is True
                assert result['environment'] == 'colab'
                assert 'system_info' in result
                assert 'python_info' in result
    
    @pytest.mark.asyncio
    async def test_execute_impl_kaggle(self, env_detection_operation):
        """Test environment detection for Kaggle."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.google.colab', side_effect=ImportError):
            with patch('os.path.exists', return_value=True):
                result = await env_detection_operation._execute_impl()
                
                assert result['success'] is True
                assert result['environment'] == 'kaggle'
    
    @pytest.mark.asyncio
    async def test_execute_impl_local(self, env_detection_operation):
        """Test environment detection for local."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.google.colab', side_effect=ImportError):
            with patch('os.path.exists', return_value=False):
                result = await env_detection_operation._execute_impl()
                
                assert result['success'] is True
                assert result['environment'] == 'local'
    
    @pytest.mark.asyncio
    async def test_execute_impl_exception(self, env_detection_operation):
        """Test environment detection with exception."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.platform.system', side_effect=Exception("Test error")):
            result = await env_detection_operation._execute_impl()
            
            assert result['success'] is False
            assert 'Test error' in result['error']
    
    def test_detect_environment_colab(self, env_detection_operation):
        """Test Colab environment detection."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.google.colab'):
            result = env_detection_operation._detect_environment()
            assert result == 'colab'
    
    def test_detect_environment_kaggle(self, env_detection_operation):
        """Test Kaggle environment detection."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.google.colab', side_effect=ImportError):
            with patch('os.path.exists', return_value=True):
                result = env_detection_operation._detect_environment()
                assert result == 'kaggle'
    
    def test_detect_environment_local(self, env_detection_operation):
        """Test local environment detection."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.google.colab', side_effect=ImportError):
            with patch('os.path.exists', return_value=False):
                result = env_detection_operation._detect_environment()
                assert result == 'local'
    
    def test_get_system_info(self, env_detection_operation):
        """Test system information collection."""
        with patch('platform.system', return_value='Linux'):
            with patch('platform.release', return_value='5.4.0'):
                with patch('platform.machine', return_value='x86_64'):
                    info = env_detection_operation._get_system_info()
                    
                    assert info['os'] == 'Linux'
                    assert info['release'] == '5.4.0'
                    assert info['machine'] == 'x86_64'
    
    def test_get_python_info(self, env_detection_operation):
        """Test Python information collection."""
        with patch('sys.version', '3.8.5'):
            with patch('sys.executable', '/usr/bin/python'):
                info = env_detection_operation._get_python_info()
                
                assert 'version' in info
                assert 'executable' in info
    
    def test_get_gpu_info_available(self, env_detection_operation):
        """Test GPU information collection when GPU is available."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.torch.cuda.is_available', return_value=True):
            with patch('smartcash.ui.setup.colab.operations.environment_detection.torch.cuda.device_count', return_value=1):
                with patch('smartcash.ui.setup.colab.operations.environment_detection.torch.cuda.get_device_name', return_value='Tesla T4'):
                    info = env_detection_operation._get_gpu_info()
                    
                    assert info['available'] is True
                    assert info['count'] == 1
                    assert 'Tesla T4' in info['devices']
    
    def test_get_gpu_info_not_available(self, env_detection_operation):
        """Test GPU information collection when GPU is not available."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.torch.cuda.is_available', return_value=False):
            info = env_detection_operation._get_gpu_info()
            
            assert info['available'] is False
            assert info['count'] == 0
    
    def test_get_gpu_info_no_torch(self, env_detection_operation):
        """Test GPU information collection when torch is not available."""
        with patch('smartcash.ui.setup.colab.operations.environment_detection.torch', None):
            info = env_detection_operation._get_gpu_info()
            
            assert info['available'] is False
            assert info['count'] == 0
            assert info['error'] == 'PyTorch not available'


class TestDriveMountOperation:
    """Test cases for DriveMountOperation."""
    
    @pytest.fixture
    def sample_ui_components(self):
        """Sample UI components for testing."""
        return {
            'auto_mount_drive_checkbox': Mock(),
            'project_name_text': Mock()
        }
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'environment': {
                'type': 'colab',
                'auto_mount_drive': True,
                'project_name': 'SmartCash'
            }
        }
    
    @pytest.fixture
    def drive_mount_operation(self, sample_ui_components, sample_config):
        """Create a DriveMountOperation instance for testing."""
        status_callback = Mock()
        return DriveMountOperation(sample_ui_components, sample_config, status_callback)
    
    def test_init(self, drive_mount_operation):
        """Test DriveMountOperation initialization."""
        assert drive_mount_operation is not None
        assert hasattr(drive_mount_operation, 'ui_components')
        assert hasattr(drive_mount_operation, 'config')
        assert hasattr(drive_mount_operation, 'status_callback')
    
    @pytest.mark.asyncio
    async def test_execute_impl_success(self, drive_mount_operation):
        """Test successful drive mount execution."""
        with patch.object(drive_mount_operation, '_mount_drive', return_value=True):
            with patch.object(drive_mount_operation, '_verify_mount', return_value=True):
                with patch.object(drive_mount_operation, '_get_drive_info', return_value={'space': '15GB'}):
                    result = await drive_mount_operation._execute_impl()
                    
                    assert result['success'] is True
                    assert result['mounted'] is True
                    assert 'drive_info' in result
    
    @pytest.mark.asyncio
    async def test_execute_impl_mount_failed(self, drive_mount_operation):
        """Test drive mount execution when mount fails."""
        with patch.object(drive_mount_operation, '_mount_drive', return_value=False):
            result = await drive_mount_operation._execute_impl()
            
            assert result['success'] is False
            assert 'Failed to mount Google Drive' in result['error']
    
    @pytest.mark.asyncio
    async def test_execute_impl_not_colab(self, drive_mount_operation):
        """Test drive mount execution in non-Colab environment."""
        drive_mount_operation.config['environment']['type'] = 'local'
        
        result = await drive_mount_operation._execute_impl()
        
        assert result['success'] is False
        assert 'Google Drive mounting is only available in Colab' in result['error']
    
    @pytest.mark.asyncio
    async def test_execute_impl_exception(self, drive_mount_operation):
        """Test drive mount execution with exception."""
        with patch.object(drive_mount_operation, '_mount_drive', side_effect=Exception("Test error")):
            result = await drive_mount_operation._execute_impl()
            
            assert result['success'] is False
            assert 'Test error' in result['error']
    
    def test_mount_drive_success(self, drive_mount_operation):
        """Test successful drive mounting."""
        with patch('smartcash.ui.setup.colab.operations.drive_mount.drive.mount') as mock_mount:
            result = drive_mount_operation._mount_drive()
            
            mock_mount.assert_called_once_with('/content/drive')
            assert result is True
    
    def test_mount_drive_exception(self, drive_mount_operation):
        """Test drive mounting with exception."""
        with patch('smartcash.ui.setup.colab.operations.drive_mount.drive.mount', side_effect=Exception("Mount error")):
            result = drive_mount_operation._mount_drive()
            
            assert result is False
    
    def test_mount_drive_no_drive_module(self, drive_mount_operation):
        """Test drive mounting when drive module is not available."""
        with patch('smartcash.ui.setup.colab.operations.drive_mount.drive', None):
            result = drive_mount_operation._mount_drive()
            
            assert result is False
    
    def test_verify_mount_success(self, drive_mount_operation):
        """Test successful mount verification."""
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=['MyDrive']):
                result = drive_mount_operation._verify_mount()
                
                assert result is True
    
    def test_verify_mount_path_not_exists(self, drive_mount_operation):
        """Test mount verification when path doesn't exist."""
        with patch('os.path.exists', return_value=False):
            result = drive_mount_operation._verify_mount()
            
            assert result is False
    
    def test_verify_mount_mydrive_not_exists(self, drive_mount_operation):
        """Test mount verification when MyDrive doesn't exist."""
        with patch('os.path.exists', return_value=True):
            with patch('os.listdir', return_value=[]):
                result = drive_mount_operation._verify_mount()
                
                assert result is False
    
    def test_get_drive_info_success(self, drive_mount_operation):
        """Test successful drive info collection."""
        with patch('os.path.exists', return_value=True):
            with patch('shutil.disk_usage', return_value=(1000000000, 800000000, 200000000)):
                with patch('os.listdir', return_value=['MyDrive', 'Shareddrives']):
                    info = drive_mount_operation._get_drive_info()
                    
                    assert 'total_space' in info
                    assert 'used_space' in info
                    assert 'free_space' in info
                    assert 'directories' in info
    
    def test_get_drive_info_not_mounted(self, drive_mount_operation):
        """Test drive info collection when not mounted."""
        with patch('os.path.exists', return_value=False):
            info = drive_mount_operation._get_drive_info()
            
            assert info == {}


class TestOperationHandlerFactory:
    """Test cases for OperationHandlerFactory."""
    
    @pytest.fixture
    def sample_ui_components(self):
        """Sample UI components for testing."""
        return {'test': 'component'}
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {'test': 'config'}
    
    def test_create_handler_environment_detection(self, sample_ui_components, sample_config):
        """Test creating environment detection handler."""
        handler = OperationHandlerFactory.create_handler(
            'environment_detection',
            sample_ui_components,
            sample_config
        )
        
        assert isinstance(handler, EnvironmentDetectionOperation)
        assert handler.ui_components == sample_ui_components
        assert handler.config == sample_config
    
    def test_create_handler_drive_mount(self, sample_ui_components, sample_config):
        """Test creating drive mount handler."""
        handler = OperationHandlerFactory.create_handler(
            'drive_mount',
            sample_ui_components,
            sample_config
        )
        
        assert isinstance(handler, DriveMountOperation)
        assert handler.ui_components == sample_ui_components
        assert handler.config == sample_config
    
    def test_create_handler_unavailable_operation(self, sample_ui_components, sample_config):
        """Test creating handler for unavailable operation."""
        handler = OperationHandlerFactory.create_handler(
            'gpu_setup',  # Not implemented yet
            sample_ui_components,
            sample_config
        )
        
        assert handler is None
    
    def test_create_handler_nonexistent_operation(self, sample_ui_components, sample_config):
        """Test creating handler for nonexistent operation."""
        handler = OperationHandlerFactory.create_handler(
            'nonexistent_operation',
            sample_ui_components,
            sample_config
        )
        
        assert handler is None
    
    def test_get_available_operations(self):
        """Test getting available operations."""
        operations = OperationHandlerFactory.get_available_operations()
        
        assert isinstance(operations, list)
        assert 'environment_detection' in operations
        assert 'drive_mount' in operations
        assert 'gpu_setup' not in operations  # Not implemented yet
    
    def test_is_operation_available_true(self):
        """Test checking if operation is available (true case)."""
        assert OperationHandlerFactory.is_operation_available('environment_detection') is True
        assert OperationHandlerFactory.is_operation_available('drive_mount') is True
    
    def test_is_operation_available_false(self):
        """Test checking if operation is available (false case)."""
        assert OperationHandlerFactory.is_operation_available('gpu_setup') is False
        assert OperationHandlerFactory.is_operation_available('nonexistent_operation') is False