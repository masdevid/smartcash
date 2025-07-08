"""
Tests for smartcash.ui.setup.colab.operations.init_operation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from smartcash.ui.setup.colab.operations.init_operation import InitOperation


class TestInitOperation:
    """Test cases for InitOperation."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'environment': {
                'type': 'colab',
                'auto_mount_drive': True,
                'project_name': 'SmartCash'
            }
        }
    
    @pytest.fixture
    def init_operation(self, mock_config):
        """Create InitOperation instance for testing."""
        return InitOperation(config=mock_config)
    
    def test_init_operation_creation(self, init_operation):
        """Test InitOperation can be created."""
        assert init_operation is not None
        assert init_operation.config is not None
        assert 'environment' in init_operation.config
    
    def test_get_operations(self, init_operation):
        """Test get_operations returns correct operations."""
        operations = init_operation.get_operations()
        assert 'init' in operations
        assert callable(operations['init'])
    
    @patch('smartcash.ui.setup.colab.operations.init_operation.detect_environment_info')
    def test_execute_init_success_colab(self, mock_detect_env, init_operation):
        """Test successful initialization in Colab environment."""
        # Mock environment detection
        mock_detect_env.return_value = {
            'runtime': {'type': 'colab', 'display': 'Google Colab'},
            'os': {'system': 'Linux', 'release': '5.4.0'},
            'total_ram': 13958643712,  # ~13GB
            'cpu_cores': 2,
            'gpu': 'Tesla T4',
            'is_colab': True,
            'drive_mounted': False
        }
        
        # Mock progress callback
        progress_callback = Mock()
        
        # Execute operation
        result = init_operation.execute_init(progress_callback)
        
        # Assertions
        assert result['success'] is True
        assert result['environment'] == 'colab'
        assert 'system_info' in result
        assert 'env_info' in result
        assert 'validation' in result
        
        # Check progress callbacks were called
        assert progress_callback.call_count >= 4
        progress_callback.assert_any_call(10, "🔍 Detecting runtime environment...")
        progress_callback.assert_any_call(100, "✅ Initialization complete")
    
    @patch('smartcash.ui.setup.colab.operations.init_operation.detect_environment_info')
    def test_execute_init_success_local(self, mock_detect_env, init_operation):
        """Test successful initialization in local environment."""
        # Mock environment detection
        mock_detect_env.return_value = {
            'runtime': {'type': 'local', 'display': 'Local Environment'},
            'os': {'system': 'Darwin', 'release': '21.6.0'},
            'total_ram': 17179869184,  # 16GB
            'cpu_cores': 8,
            'gpu': 'No GPU available',
            'is_colab': False,
            'drive_mounted': False
        }
        
        result = init_operation.execute_init()
        
        # Assertions
        assert result['success'] is True
        assert result['environment'] == 'local'
        assert result['system_info']['os_display'] == 'Darwin 21.6.0'
        assert result['system_info']['ram_gb'] == 16.0
        assert result['system_info']['cpu_cores'] == 8
    
    def test_execute_init_validation_failure(self, init_operation):
        """Test initialization with validation failure."""
        # Remove required config
        init_operation.config = {}
        
        with patch('smartcash.ui.setup.colab.operations.init_operation.detect_environment_info') as mock_detect:
            mock_detect.return_value = {
                'runtime': {'type': 'local'},
                'os': {'system': 'Linux'},
                'total_ram': 1000000000
            }
            
            result = init_operation.execute_init()
            
            assert result['success'] is False
            assert 'Configuration validation failed' in result['error']
    
    @patch('smartcash.ui.setup.colab.operations.init_operation.detect_environment_info')
    def test_execute_init_exception_handling(self, mock_detect_env, init_operation):
        """Test exception handling during initialization."""
        # Mock exception during environment detection
        mock_detect_env.side_effect = Exception("Detection failed")
        
        result = init_operation.execute_init()
        
        assert result['success'] is False
        assert 'Initialization failed' in result['error']
    
    def test_get_system_info(self, init_operation):
        """Test _get_system_info method."""
        env_info = {
            'os': {'system': 'Linux', 'release': '5.4.0', 'machine': 'x86_64'},
            'total_ram': 13958643712,
            'cpu_cores': 2,
            'gpu': 'Tesla T4',
            'is_colab': True,
            'drive_mounted': True,
            'drive_mount_path': '/content/drive'
        }
        
        system_info = init_operation._get_system_info(env_info)
        
        assert system_info['os'] == 'Linux'
        assert system_info['release'] == '5.4.0'
        assert system_info['machine'] == 'x86_64'
        assert system_info['os_display'] == 'Linux 5.4.0'
        assert system_info['ram_gb'] == pytest.approx(13.0, rel=0.1)
        assert system_info['cpu_cores'] == 2
        assert system_info['gpu_available'] is True
        assert system_info['gpu_name'] == 'Tesla T4'
        assert system_info['is_colab'] is True
        assert system_info['drive_mounted'] is True
        assert system_info['drive_mount_path'] == '/content/drive'
    
    def test_validate_config_success(self, init_operation):
        """Test successful configuration validation."""
        validation = init_operation._validate_config()
        
        assert validation['valid'] is True
        assert len(validation['issues']) == 0
    
    def test_validate_config_missing_environment(self, init_operation):
        """Test validation with missing environment config."""
        init_operation.config = {}
        
        validation = init_operation._validate_config()
        
        assert validation['valid'] is False
        assert 'Missing environment configuration' in validation['issues']
    
    def test_validate_config_missing_type(self, init_operation):
        """Test validation with missing environment type."""
        init_operation.config = {'environment': {}}
        
        validation = init_operation._validate_config()
        
        assert validation['valid'] is False
        assert 'Environment type not specified' in validation['issues']
    
    @patch('builtins.__import__')
    def test_validate_config_colab_mismatch(self, mock_import, init_operation):
        """Test validation when config says Colab but not in Colab."""
        # Mock ImportError for google.colab
        mock_import.side_effect = ImportError("No module named 'google.colab'")
        
        init_operation.config = {'environment': {'type': 'colab'}}
        
        validation = init_operation._validate_config()
        
        assert validation['valid'] is False
        assert any('not running in Colab environment' in issue for issue in validation['issues'])