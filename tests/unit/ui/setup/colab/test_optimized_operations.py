"""
Integration tests for optimized colab operations with mixin integration.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from smartcash.ui.setup.colab.operations.init_operation import InitOperation
from smartcash.ui.setup.colab.operations.folders_operation import FoldersOperation
from smartcash.ui.setup.colab.operations.env_setup_operation import EnvSetupOperation
from smartcash.ui.setup.colab.operations.config_sync_operation import ConfigSyncOperation
from smartcash.ui.setup.colab.operations.symlink_operation import SymlinkOperation
from smartcash.ui.setup.colab.operations.verify_operation import VerifyOperation
from smartcash.ui.components.operation_container import OperationContainer


class TestOptimizedColabOperations:
    """Test suite for optimized colab operations with mixin integration."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'paths': {
                'colab_base': '/tmp/test_smartcash',
                'drive_base': '/tmp/test_drive/SmartCash'
            },
            'environment': {
                'type': 'colab',
                'project_name': 'SmartCash',
                'gpu_enabled': False
            }
        }
    
    @pytest.fixture
    def mock_operation_container(self):
        """Mock operation container for testing."""
        container = Mock(spec=OperationContainer)
        container.get = Mock(return_value=Mock())
        return container
    
    @pytest.fixture
    def temp_directories(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            colab_base = os.path.join(temp_dir, 'test_smartcash')
            drive_base = os.path.join(temp_dir, 'test_drive', 'SmartCash')
            
            os.makedirs(colab_base, exist_ok=True)
            os.makedirs(drive_base, exist_ok=True)
            
            yield {
                'colab_base': colab_base,
                'drive_base': drive_base,
                'temp_dir': temp_dir
            }

    def test_init_operation_mixin_integration(self, mock_config, mock_operation_container):
        """Test InitOperation with mixin integration."""
        # Create operation
        operation = InitOperation('init_operation', mock_config, mock_operation_container)
        
        # Verify mixin methods are available
        assert hasattr(operation, 'initialize_backend_services')
        assert hasattr(operation, 'get_module_config')
        assert hasattr(operation, 'sync_config_to_ui')
        
        # Test operation methods
        operations = operation.get_operations()
        assert 'init' in operations
        
        # Test actual execution with mocked dependencies
        with patch.object(operation, 'detect_environment_enhanced') as mock_detect:
            mock_detect.return_value = {
                'runtime': {'type': 'colab'},
                'is_colab': True,
                'drive_mounted': False
            }
            
            # Execute operation
            result = operation.execute_init()
            
            # Verify results
            assert result['success'] is True
            assert 'environment_info' in result

    def test_folders_operation_mixin_integration(self, mock_config, mock_operation_container, temp_directories):
        """Test FoldersOperation with mixin integration."""
        # Update config with temp directories
        test_config = mock_config.copy()
        test_config['paths']['colab_base'] = temp_directories['colab_base']
        
        # Create operation
        operation = FoldersOperation('folders_operation', test_config, mock_operation_container)
        
        # Verify mixin methods are available
        assert hasattr(operation, 'initialize_backend_services')
        assert hasattr(operation, 'get_module_config')
        
        # Test operation methods
        operations = operation.get_operations()
        assert 'create_folders' in operations
        
        # Execute operation
        result = operation.execute_create_folders()
        
        # Verify results
        assert result['success'] is True
        assert 'creation_results' in result

    def test_env_setup_operation_mixin_integration(self, mock_config, mock_operation_container):
        """Test EnvSetupOperation with mixin integration."""
        # Create operation
        operation = EnvSetupOperation('env_setup_operation', mock_config, mock_operation_container)
        
        # Verify mixin methods are available
        assert hasattr(operation, 'initialize_backend_services')
        assert hasattr(operation, 'get_module_config')
        
        # Test operation methods
        operations = operation.get_operations()
        assert 'setup_environment' in operations
        
        # Execute operation with clean environment
        with patch.dict(os.environ, {}, clear=True):
            result = operation.execute_setup_environment()
            
            # Verify results
            assert result['success'] is True
            assert 'env_variables' in result
            
            # Verify environment variables were set
            assert 'SMARTCASH_ROOT' in os.environ

    def test_config_sync_operation_mixin_integration(self, mock_config, mock_operation_container, temp_directories):
        """Test ConfigSyncOperation with mixin integration."""
        # Update config with temp directories
        test_config = mock_config.copy()
        test_config['paths']['drive_base'] = temp_directories['drive_base']
        test_config['paths']['colab_base'] = temp_directories['colab_base']
        
        # Create test config files
        source_config_dir = os.path.join(temp_directories['drive_base'], 'configs')
        os.makedirs(source_config_dir, exist_ok=True)
        
        test_config_file = os.path.join(source_config_dir, 'data.yaml')
        with open(test_config_file, 'w') as f:
            f.write('test_config: true\n')
        
        # Create operation
        operation = ConfigSyncOperation('config_sync_operation', test_config, mock_operation_container)
        
        # Verify mixin methods are available
        assert hasattr(operation, 'initialize_backend_services')
        assert hasattr(operation, 'get_module_config')
        
        # Test operation methods
        operations = operation.get_operations()
        assert 'sync_configs' in operations
        
        # Execute operation
        result = operation.execute_sync_configs()
        
        # Verify results
        assert result['success'] is True
        assert 'sync_results' in result

    def test_symlink_operation_mixin_integration(self, mock_config, mock_operation_container, temp_directories):
        """Test SymlinkOperation with mixin integration."""
        # Update config with temp directories
        test_config = mock_config.copy()
        test_config['paths']['drive_base'] = temp_directories['drive_base']
        test_config['paths']['colab_base'] = temp_directories['colab_base']
        
        # Create source directories
        for subdir in ['data', 'models', 'configs']:
            os.makedirs(os.path.join(temp_directories['drive_base'], subdir), exist_ok=True)
        
        # Create operation
        operation = SymlinkOperation('symlink_operation', test_config, mock_operation_container)
        
        # Verify mixin methods are available
        assert hasattr(operation, 'initialize_backend_services')
        assert hasattr(operation, 'get_module_config')
        
        # Test operation methods
        operations = operation.get_operations()
        assert 'create_symlinks' in operations
        
        # Execute operation
        result = operation.execute_create_symlinks()
        
        # Verify results
        assert result['success'] is True
        assert 'creation_results' in result
        assert 'verification_results' in result

    def test_verify_operation_mixin_integration(self, mock_config, mock_operation_container):
        """Test VerifyOperation with mixin integration."""
        # Create operation
        operation = VerifyOperation('verify_operation', mock_config, mock_operation_container)
        
        # Verify mixin methods are available
        assert hasattr(operation, 'get_module_config')
        assert hasattr(operation, 'initialize_backend_services')
        
        # Test operation methods
        operations = operation.get_operations()
        assert 'verify_setup' in operations
        
        # Mock environment manager for verification
        with patch('smartcash.ui.setup.colab.operations.verify_operation.get_environment_manager') as mock_env_mgr:
            mock_env_manager = Mock()
            mock_env_manager.is_colab = True
            mock_env_manager.is_drive_mounted = False
            mock_env_manager.get_system_info.return_value = {
                'total_memory_gb': 8.0,
                'cuda_available': False
            }
            mock_env_mgr.return_value = mock_env_manager
            
            # Execute operation
            result = operation.execute_verify_setup()
            
            # Verify results
            assert result['success'] is True
            assert 'environment_verification' in result
            assert 'integrity_check' in result

    def test_backend_service_integration(self, mock_config, mock_operation_container):
        """Test backend service initialization across operations."""
        operations = [
            InitOperation('init_operation', mock_config, mock_operation_container),
            FoldersOperation('folders_operation', mock_config, mock_operation_container),
            ConfigSyncOperation('config_sync_operation', mock_config, mock_operation_container)
        ]
        
        # Test backend service initialization
        for operation in operations:
            # Verify method exists
            assert hasattr(operation, 'initialize_backend_services')
            
            # Test service initialization with mock
            test_configs = {'test_service': {'config': 'value'}}
            test_services = ['test_service']
            
            # Mock the actual service initialization to avoid dependencies
            with patch.object(operation, 'setup_service_callbacks'), \
                 patch.object(operation, 'handle_service_errors'):
                result = operation.initialize_backend_services(test_configs, test_services)
                
                # Should return a result dict
                assert isinstance(result, dict)

    def test_cross_module_config_access(self, mock_config, mock_operation_container):
        """Test cross-module configuration access."""
        operation = InitOperation('init_operation', mock_config, mock_operation_container)
        
        # Test getting module config
        module_config = operation.get_module_config('model')
        
        # Should return None or a dict (depending on whether module is loaded)
        assert module_config is None or isinstance(module_config, dict)

    def test_error_handling_integration(self, mock_config, mock_operation_container):
        """Test error handling in optimized operations."""
        operation = InitOperation('init_operation', mock_config, mock_operation_container)
        
        # Test error handling with environment detection failure
        with patch.object(operation, 'detect_environment_enhanced') as mock_detect:
            mock_detect.side_effect = Exception("Environment detection failed")
            
            # Execute operation
            result = operation.execute_init()
            
            # Verify error handling
            assert result['success'] is False
            assert 'error' in result

    def test_progress_tracking_integration(self, mock_config, mock_operation_container):
        """Test progress tracking in optimized operations."""
        operation = FoldersOperation('folders_operation', mock_config, mock_operation_container)
        
        # Mock progress callback
        progress_callback = Mock()
        
        # Execute with progress callback
        result = operation.execute_create_folders(progress_callback)
        
        # Verify progress was called
        assert progress_callback.called
        assert result['success'] is True

    def test_operation_container_integration(self, mock_config, mock_operation_container):
        """Test operation container integration."""
        operation = InitOperation('init_operation', mock_config, mock_operation_container)
        
        # Verify operation container is set
        assert operation.operation_container == mock_operation_container
        
        # Test logging integration
        assert hasattr(operation, 'log')
        assert hasattr(operation, 'log_info')
        assert hasattr(operation, 'log_error')

    def test_config_validation(self, mock_config, mock_operation_container):
        """Test configuration validation across operations."""
        operation = InitOperation('init_operation', mock_config, mock_operation_container)
        
        # Test config validation
        validation_result = operation.validate_colab_environment(mock_config)
        
        # Should return validation results
        assert isinstance(validation_result, dict)
        assert 'valid' in validation_result
        assert 'issues' in validation_result

    def test_progress_steps_consistency(self, mock_config, mock_operation_container):
        """Test progress steps consistency across operations."""
        operations = [
            ('init_operation', InitOperation),
            ('folders_operation', FoldersOperation),
            ('env_setup_operation', EnvSetupOperation),
            ('config_sync_operation', ConfigSyncOperation),
            ('symlink_operation', SymlinkOperation),
            ('verify_operation', VerifyOperation)
        ]
        
        for op_name, op_class in operations:
            operation = op_class(op_name, mock_config, mock_operation_container)
            
            # Get progress steps
            progress_steps = operation.get_progress_steps(op_name.replace('_operation', ''))
            
            # Verify progress steps structure
            assert isinstance(progress_steps, list)
            assert len(progress_steps) > 0
            
            for step in progress_steps:
                assert 'progress' in step
                assert 'message' in step
                assert isinstance(step['progress'], int)
                assert 0 <= step['progress'] <= 100