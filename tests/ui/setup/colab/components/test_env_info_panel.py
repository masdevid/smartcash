"""
Tests for smartcash.ui.setup.colab.components.env_info_panel
"""

import pytest
from unittest.mock import Mock, patch
from smartcash.ui.setup.colab.components.env_info_panel import (
    create_env_info_panel,
    _format_env_info_content,
    _format_enhanced_storage_info,
    _format_enhanced_gpu_info,
    _get_enhanced_drive_status,
    _get_smartcash_env_status,
    _format_additional_info,
    _format_bytes
)


class TestEnvInfoPanel:
    """Test cases for environment info panel components."""
    
    @pytest.fixture
    def mock_env_info(self):
        """Mock environment information for testing."""
        return {
            'runtime': {
                'type': 'colab',
                'display': 'Google Colab (GPU available)'
            },
            'python_version': '3.8.10',
            'os': {
                'system': 'Linux',
                'release': '5.4.0',
                'machine': 'x86_64'
            },
            'memory_info': {
                'total_gb': 13.0,
                'available_gb': 8.5,
                'percent_used': 34.6,
                'used_gb': 4.5
            },
            'cpu_cores': 2,
            'storage_info': {
                'total_gb': 100.0,
                'used_gb': 45.2,
                'free_gb': 54.8,
                'percent_used': 45.2
            },
            'gpu_details': {
                'available': True,
                'device_count': 1,
                'devices': [{
                    'name': 'Tesla T4',
                    'memory_total_gb': 15.0,
                    'id': 0
                }],
                'cuda_version': '11.8'
            },
            'network_info': {
                'hostname': 'colab-vm-123',
                'interfaces': [
                    {'interface': 'eth0', 'ip': '10.0.0.1'},
                    {'interface': 'lo', 'ip': '127.0.0.1'}
                ]
            },
            'environment_variables': {
                'SMARTCASH_ROOT': '/content/smartcash',
                'SMARTCASH_ENV': 'colab',
                'SMARTCASH_DATA_ROOT': '/content/data'
            },
            'drive_mounted': True,
            'drive_mount_path': '/content/drive',
            'is_colab': True
        }
    
    @patch('smartcash.ui.setup.colab.components.env_info_panel.detect_environment_info')
    def test_create_env_info_panel_with_auto_detect(self, mock_detect, mock_env_info):
        """Test creating env info panel with auto-detection."""
        mock_detect.return_value = mock_env_info
        
        widget = create_env_info_panel()
        
        assert widget is not None
        assert hasattr(widget, 'value')
        assert 'Google Colab' in widget.value
        assert 'Tesla T4' in widget.value
        mock_detect.assert_called_once()
    
    def test_create_env_info_panel_with_provided_info(self, mock_env_info):
        """Test creating env info panel with provided info."""
        widget = create_env_info_panel(mock_env_info)
        
        assert widget is not None
        assert hasattr(widget, 'value')
        assert 'Google Colab' in widget.value
        assert 'Tesla T4' in widget.value
    
    def test_format_env_info_content_comprehensive(self, mock_env_info):
        """Test comprehensive environment info formatting."""
        content = _format_env_info_content(mock_env_info)
        
        # Check main sections are present
        assert '🌐 Google Colab (GPU available)' in content
        assert '🖥️ System Information' in content
        assert '⚡ Resources' in content
        assert '🎮 Hardware & Drive' in content
        
        # Check system information
        assert 'Linux 5.4.0' in content
        assert '3.8.10' in content  # Python version
        assert 'colab-vm-123' in content  # Hostname
        
        # Check resources
        assert '2' in content  # CPU cores
        assert '13.0GB' in content  # Total RAM
        assert '8.5GB' in content  # Available RAM
        
        # Check GPU info
        assert 'Tesla T4' in content
        assert '15.0GB' in content  # GPU memory
        
        # Check SmartCash status
        assert 'Configured (colab)' in content
    
    def test_format_enhanced_storage_info_complete(self):
        """Test enhanced storage info formatting with complete data."""
        storage_info = {
            'total_gb': 100.0,
            'used_gb': 45.2,
            'free_gb': 54.8,
            'percent_used': 45.2
        }
        
        result = _format_enhanced_storage_info(storage_info)
        
        assert '45.2GB / 100.0GB' in result
        assert '45.2% used' in result
        assert '54.8GB free' in result
    
    def test_format_enhanced_storage_info_empty(self):
        """Test enhanced storage info formatting with empty data."""
        result = _format_enhanced_storage_info({})
        
        assert result == 'N/A'
    
    def test_format_enhanced_storage_info_exception(self):
        """Test enhanced storage info formatting with invalid data."""
        storage_info = {'invalid': 'data'}
        
        result = _format_enhanced_storage_info(storage_info)
        
        assert result == "Storage info unavailable"
    
    def test_format_enhanced_gpu_info_available(self):
        """Test enhanced GPU info formatting when available."""
        env_info = {
            'gpu_details': {
                'available': True,
                'device_count': 2,
                'devices': [
                    {'name': 'Tesla T4', 'memory_total_gb': 15.0},
                    {'name': 'Tesla V100', 'memory_total_gb': 32.0}
                ]
            }
        }
        
        result = _format_enhanced_gpu_info(env_info)
        
        assert 'Tesla T4' in result
        assert '15.0GB' in result
        assert '+1 more' in result
    
    def test_format_enhanced_gpu_info_not_available(self):
        """Test enhanced GPU info formatting when not available."""
        env_info = {
            'gpu_details': {
                'available': False,
                'reason': 'CUDA not available'
            }
        }
        
        result = _format_enhanced_gpu_info(env_info)
        
        assert '❌ CUDA not available' in result
    
    def test_format_enhanced_gpu_info_no_devices(self):
        """Test enhanced GPU info formatting with no devices."""
        env_info = {
            'gpu_details': {
                'available': True,
                'devices': []
            }
        }
        
        result = _format_enhanced_gpu_info(env_info)
        
        assert '❌ No GPU devices found' in result
    
    def test_get_enhanced_drive_status_mounted(self):
        """Test enhanced drive status when mounted."""
        env_info = {
            'drive_mounted': True,
            'drive_mount_path': '/content/drive'
        }
        
        result = _get_enhanced_drive_status(env_info)
        
        assert '✅ Mounted at /content/drive' in result
    
    def test_get_enhanced_drive_status_not_mounted(self):
        """Test enhanced drive status when not mounted."""
        env_info = {
            'drive_mounted': False
        }
        
        result = _get_enhanced_drive_status(env_info)
        
        assert '❌ Not mounted' in result
    
    def test_get_smartcash_env_status_configured(self):
        """Test SmartCash environment status when fully configured."""
        env_vars = {
            'SMARTCASH_ROOT': '/content/smartcash',
            'SMARTCASH_ENV': 'colab',
            'SMARTCASH_DATA_ROOT': '/content/data'
        }
        
        result = _get_smartcash_env_status(env_vars)
        
        assert '✅ Configured (colab)' in result
    
    def test_get_smartcash_env_status_partial(self):
        """Test SmartCash environment status when partially configured."""
        env_vars = {
            'SMARTCASH_ROOT': '/content/smartcash',
            'SMARTCASH_ENV': 'colab'
            # Missing SMARTCASH_DATA_ROOT
        }
        
        result = _get_smartcash_env_status(env_vars)
        
        assert '⚠️ Partial (2/3 vars)' in result
    
    def test_get_smartcash_env_status_not_configured(self):
        """Test SmartCash environment status when not configured."""
        env_vars = {}
        
        result = _get_smartcash_env_status(env_vars)
        
        assert '❌ Not configured' in result
    
    def test_format_additional_info_with_network_and_cuda(self):
        """Test additional info formatting with network and CUDA info."""
        env_info = {
            'network_info': {
                'interfaces': [
                    {'interface': 'eth0'},
                    {'interface': 'lo'}
                ]
            },
            'gpu_details': {
                'available': True,
                'cuda_version': '11.8'
            }
        }
        
        result = _format_additional_info(env_info)
        
        assert 'Network:</strong> 2 interface(s)' in result
        assert 'CUDA:</strong> 11.8' in result
    
    def test_format_additional_info_empty(self):
        """Test additional info formatting with no additional info."""
        env_info = {}
        
        result = _format_additional_info(env_info)
        
        assert result == ''
    
    def test_format_bytes_valid_values(self):
        """Test byte formatting with valid values."""
        assert _format_bytes(1024) == "1.0 KB"
        assert _format_bytes(1048576) == "1.0 MB"
        assert _format_bytes(1073741824) == "1.0 GB"
        assert _format_bytes(1099511627776) == "1.0 TB"
        assert _format_bytes(512) == "512 B"
    
    def test_format_bytes_invalid_values(self):
        """Test byte formatting with invalid values."""
        assert _format_bytes(-1) == 'N/A'
        assert _format_bytes('invalid') == 'N/A'
        assert _format_bytes(None) == 'N/A'
    
    def test_format_bytes_large_values(self):
        """Test byte formatting with very large values."""
        petabyte = 1125899906842624
        assert _format_bytes(petabyte) == "1.0 PB"
        
        # Test larger than petabyte
        larger = petabyte * 1024
        result = _format_bytes(larger)
        assert 'PB' in result and '1024.0' in result


class TestEnvInfoPanelEdgeCases:
    """Test edge cases for environment info panel."""
    
    def test_format_env_info_content_minimal_data(self):
        """Test formatting with minimal environment data."""
        minimal_env_info = {
            'runtime': {'display': 'Unknown Environment'},
            'python_version': 'N/A',
            'os': {},
            'memory_info': {},
            'network_info': {},
            'environment_variables': {}
        }
        
        content = _format_env_info_content(minimal_env_info)
        
        # Should not crash and should contain basic structure
        assert '🌐 Unknown Environment' in content
        assert '🖥️ System Information' in content
        assert 'N/A' in content  # Should show N/A for missing data
    
    def test_format_env_info_content_zero_values(self):
        """Test formatting with zero values."""
        env_info_with_zeros = {
            'runtime': {'display': 'Test Environment'},
            'python_version': '3.8.0',
            'os': {'system': 'Linux', 'release': '5.4.0'},
            'memory_info': {
                'total_gb': 0,
                'available_gb': 0,
                'percent_used': 0
            },
            'cpu_cores': 0,
            'network_info': {'hostname': 'test'},
            'environment_variables': {}
        }
        
        content = _format_env_info_content(env_info_with_zeros)
        
        # Should handle zero values gracefully
        assert '0.0GB' in content  # RAM values
        assert '0' in content  # CPU cores
        assert 'Test Environment' in content
    
    def test_widget_layout_properties(self):
        """Test that widget has correct layout properties."""
        env_info = {'runtime': {'display': 'Test'}}
        widget = create_env_info_panel(env_info)
        
        layout = widget.layout
        assert layout.width == '100%'
        assert layout.padding == '15px'
        assert '1px solid #e0e0e0' in layout.border
        assert layout.border_radius == '6px'
        assert layout.background == '#f9f9f9'