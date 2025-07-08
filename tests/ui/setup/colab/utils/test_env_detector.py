"""
Tests for smartcash.ui.setup.colab.utils.env_detector
"""

import pytest
import os
import platform
from unittest.mock import Mock, patch, MagicMock
from smartcash.ui.setup.colab.utils.env_detector import (
    detect_environment_info,
    _get_python_version,
    _get_os_info,
    _is_google_colab,
    _get_gpu_info,
    _get_gpu_details,
    _get_memory_info,
    _get_storage_info,
    _get_network_info,
    _get_environment_variables,
    _get_cpu_cores,
    _get_total_ram,
    _is_drive_mounted,
    get_runtime_type
)


class TestEnvDetector:
    """Test cases for environment detection utilities."""
    
    def test_get_python_version(self):
        """Test Python version detection."""
        version = _get_python_version()
        
        # Should return a version string in format "major.minor.micro"
        assert isinstance(version, str)
        assert len(version.split('.')) == 3
        
        # Compare with actual version
        import sys
        expected = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        assert version == expected
    
    def test_get_os_info(self):
        """Test OS information detection."""
        os_info = _get_os_info()
        
        assert isinstance(os_info, dict)
        required_keys = ['system', 'release', 'version', 'machine', 'processor', 'platform', 'node']
        for key in required_keys:
            assert key in os_info
        
        # Should match platform module results
        assert os_info['system'] == platform.system()
        assert os_info['release'] == platform.release()
        assert os_info['machine'] == platform.machine()
    
    @patch('builtins.__import__')
    def test_is_google_colab_true(self, mock_import):
        """Test Google Colab detection when in Colab."""
        # Mock successful import of google.colab
        mock_import.return_value = Mock()
        
        result = _is_google_colab()
        
        assert result is True
        mock_import.assert_called_with('google.colab')
    
    @patch('builtins.__import__')
    def test_is_google_colab_false(self, mock_import):
        """Test Google Colab detection when not in Colab."""
        # Mock ImportError for google.colab
        mock_import.side_effect = ImportError("No module named 'google.colab'")
        
        result = _is_google_colab()
        
        assert result is False
    
    @patch('builtins.__import__')
    def test_get_gpu_info_available(self, mock_import):
        """Test GPU info when GPU is available."""
        # Mock torch module
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Tesla T4"
        mock_import.return_value = mock_torch
        
        result = _get_gpu_info()
        
        assert result == "Tesla T4"
    
    @patch('builtins.__import__')
    def test_get_gpu_info_not_available(self, mock_import):
        """Test GPU info when GPU is not available."""
        # Mock torch module with no GPU
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_import.return_value = mock_torch
        
        result = _get_gpu_info()
        
        assert result == "No GPU available"
    
    @patch('builtins.__import__')
    def test_get_gpu_info_no_torch(self, mock_import):
        """Test GPU info when PyTorch is not available."""
        mock_import.side_effect = ImportError("No module named 'torch'")
        
        result = _get_gpu_info()
        
        assert result == "PyTorch not available"
    
    @patch('builtins.__import__')
    def test_get_gpu_details_available(self, mock_import):
        """Test detailed GPU info when GPU is available."""
        # Mock torch module with detailed GPU info
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "Tesla T4"
        mock_torch.version.cuda = "11.8"
        
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 15843721216  # ~15GB
        mock_props.major = 7
        mock_props.minor = 5
        mock_props.multi_processor_count = 40
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        mock_import.return_value = mock_torch
        
        result = _get_gpu_details()
        
        assert result['available'] is True
        assert result['device_count'] == 1
        assert result['current_device'] == 0
        assert result['cuda_version'] == "11.8"
        assert len(result['devices']) == 1
        
        device = result['devices'][0]
        assert device['name'] == "Tesla T4"
        assert device['memory_total_gb'] == pytest.approx(14.76, rel=0.1)
        assert device['major'] == 7
        assert device['minor'] == 5
    
    @patch('builtins.__import__')
    def test_get_gpu_details_not_available(self, mock_import):
        """Test detailed GPU info when GPU is not available."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = False
        mock_import.return_value = mock_torch
        
        result = _get_gpu_details()
        
        assert result['available'] is False
        assert result['reason'] == 'CUDA not available'
        assert result['device_count'] == 0
        assert result['devices'] == []
    
    @patch('builtins.__import__')
    def test_get_memory_info_success(self, mock_import):
        """Test memory info detection."""
        # Mock psutil module
        mock_psutil = Mock()
        mock_memory = Mock()
        mock_memory.total = 17179869184  # 16GB
        mock_memory.available = 8589934592  # 8GB
        mock_memory.percent = 50.0
        mock_memory.used = 8589934592  # 8GB
        mock_memory.free = 8589934592  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_import.return_value = mock_psutil
        
        result = _get_memory_info()
        
        assert result['total'] == 17179869184
        assert result['available'] == 8589934592
        assert result['percent_used'] == 50.0
        assert result['total_gb'] == 16.0
        assert result['available_gb'] == 8.0
        assert result['used_gb'] == 8.0
    
    @patch('builtins.__import__')
    def test_get_storage_info_success(self, mock_import):
        """Test storage info detection."""
        # Mock psutil module
        mock_psutil = Mock()
        mock_disk = Mock()
        mock_disk.total = 107374182400  # 100GB
        mock_disk.used = 53687091200   # 50GB
        mock_disk.free = 53687091200   # 50GB
        mock_psutil.disk_usage.return_value = mock_disk
        mock_import.return_value = mock_psutil
        
        result = _get_storage_info()
        
        assert result['total'] == 107374182400
        assert result['used'] == 53687091200
        assert result['free'] == 53687091200
        assert result['percent_used'] == 50.0
        assert result['total_gb'] == 100.0
        assert result['used_gb'] == 50.0
        assert result['free_gb'] == 50.0
    
    @patch('builtins.__import__')
    def test_get_network_info_success(self, mock_import):
        """Test network info detection."""
        # Mock psutil and socket modules
        mock_psutil = Mock()
        mock_socket = Mock()
        
        # Mock network interfaces
        mock_addr = Mock()
        mock_addr.family = mock_socket.AF_INET
        mock_addr.address = '192.168.1.100'
        mock_addr.netmask = '255.255.255.0'
        mock_addr.broadcast = '192.168.1.255'
        
        mock_psutil.net_if_addrs.return_value = {
            'eth0': [mock_addr]
        }
        mock_socket.gethostname.return_value = 'test-hostname'
        mock_socket.AF_INET = 2
        
        def mock_import_side_effect(name):
            if name == 'psutil':
                return mock_psutil
            elif name == 'socket':
                return mock_socket
            return Mock()
        
        mock_import.side_effect = mock_import_side_effect
        
        result = _get_network_info()
        
        assert result['hostname'] == 'test-hostname'
        assert len(result['interfaces']) == 1
        interface = result['interfaces'][0]
        assert interface['interface'] == 'eth0'
        assert interface['ip'] == '192.168.1.100'
        assert interface['netmask'] == '255.255.255.0'
    
    def test_get_environment_variables(self):
        """Test environment variables collection."""
        # Set some test environment variables
        test_vars = {
            'SMARTCASH_ROOT': '/test/smartcash',
            'PYTHONPATH': '/test/path',
            'HOME': '/home/test'
        }
        
        with patch.dict(os.environ, test_vars, clear=False):
            result = _get_environment_variables()
            
            assert 'SMARTCASH_ROOT' in result
            assert result['SMARTCASH_ROOT'] == '/test/smartcash'
            assert 'PYTHONPATH' in result
            assert result['PYTHONPATH'] == '/test/path'
            assert 'HOME' in result
            assert result['HOME'] == '/home/test'
    
    @patch('multiprocessing.cpu_count')
    def test_get_cpu_cores(self, mock_cpu_count):
        """Test CPU core count detection."""
        mock_cpu_count.return_value = 8
        
        result = _get_cpu_cores()
        
        assert result == 8
        mock_cpu_count.assert_called_once()
    
    @patch('builtins.__import__')
    def test_get_total_ram_success(self, mock_import):
        """Test total RAM detection."""
        mock_psutil = Mock()
        mock_memory = Mock()
        mock_memory.total = 17179869184  # 16GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_import.return_value = mock_psutil
        
        result = _get_total_ram()
        
        assert result == 17179869184
    
    @patch('builtins.__import__')
    def test_get_total_ram_no_psutil(self, mock_import):
        """Test total RAM detection when psutil not available."""
        mock_import.side_effect = ImportError("No module named 'psutil'")
        
        result = _get_total_ram()
        
        assert result == 0
    
    @patch('smartcash.ui.setup.colab.utils.env_detector._is_google_colab')
    @patch('os.path.exists')
    @patch('os.path.ismount')
    def test_is_drive_mounted_success(self, mock_ismount, mock_exists, mock_is_colab):
        """Test drive mount detection when mounted."""
        mock_is_colab.return_value = True
        mock_ismount.return_value = True
        
        is_mounted, mount_path = _is_drive_mounted()
        
        assert is_mounted is True
        assert mount_path == '/content/drive'
    
    @patch('smartcash.ui.setup.colab.utils.env_detector._is_google_colab')
    def test_is_drive_mounted_not_colab(self, mock_is_colab):
        """Test drive mount detection when not in Colab."""
        mock_is_colab.return_value = False
        
        is_mounted, mount_path = _is_drive_mounted()
        
        assert is_mounted is False
        assert mount_path == ''
    
    @patch('smartcash.ui.setup.colab.utils.env_detector._is_google_colab')
    @patch('smartcash.ui.setup.colab.utils.env_detector._get_gpu_info')
    def test_get_runtime_type_colab_with_gpu(self, mock_get_gpu, mock_is_colab):
        """Test runtime type detection for Colab with GPU."""
        mock_is_colab.return_value = True
        mock_get_gpu.return_value = "Tesla T4"
        
        result = get_runtime_type()
        
        assert result['type'] == 'colab'
        assert result['gpu'] == 'available'
        assert result['display'] == 'Colab (available GPU)'
    
    @patch('smartcash.ui.setup.colab.utils.env_detector._is_google_colab')
    @patch('smartcash.ui.setup.colab.utils.env_detector._get_gpu_info')
    def test_get_runtime_type_local_no_gpu(self, mock_get_gpu, mock_is_colab):
        """Test runtime type detection for local without GPU."""
        mock_is_colab.return_value = False
        mock_get_gpu.return_value = "No GPU available"
        
        result = get_runtime_type()
        
        assert result['type'] == 'local'
        assert result['gpu'] == 'not available'
        assert result['display'] == 'Local (not available GPU)'
    
    @patch('smartcash.ui.setup.colab.utils.env_detector.safe_get')
    def test_detect_environment_info_comprehensive(self, mock_safe_get):
        """Test comprehensive environment info detection."""
        # Mock all the component functions
        mock_safe_get.side_effect = lambda func, default, *args, **kwargs: {
            '_get_python_version': '3.8.10',
            '_get_os_info': {'system': 'Linux', 'release': '5.4.0'},
            'get_runtime_type': {'type': 'colab', 'display': 'Google Colab'},
            '_get_gpu_info': 'Tesla T4',
            '_get_gpu_details': {'available': True, 'device_count': 1},
            '_get_storage_info': {'total_gb': 100.0, 'used_gb': 50.0},
            '_get_cpu_cores': 2,
            '_get_total_ram': 13958643712,
            '_get_memory_info': {'total_gb': 13.0, 'available_gb': 8.0},
            '_get_network_info': {'hostname': 'colab-vm'},
            '_get_environment_variables': {'PYTHONPATH': '/content'},
            '_is_drive_mounted': (True, '/content/drive'),
            '_is_google_colab': True
        }.get(func.__name__, default)
        
        result = detect_environment_info()
        
        assert result['status'] == 'success'
        assert result['python_version'] == '3.8.10'
        assert 'os' in result
        assert 'runtime' in result
        assert 'gpu' in result
        assert 'gpu_details' in result
        assert 'storage_info' in result
        assert 'cpu_cores' in result
        assert 'total_ram' in result
        assert 'memory_info' in result
        assert 'network_info' in result
        assert 'environment_variables' in result
        assert result['drive_mounted'] is True
        assert result['is_colab'] is True
    
    def test_detect_environment_info_exception_handling(self):
        """Test exception handling in environment detection."""
        with patch('smartcash.ui.setup.colab.utils.env_detector._get_python_version', side_effect=Exception("Test error")):
            result = detect_environment_info()
            
            # Should still return a result with error status
            assert 'status' in result
            # The safe_get function should handle the exception
            assert isinstance(result, dict)