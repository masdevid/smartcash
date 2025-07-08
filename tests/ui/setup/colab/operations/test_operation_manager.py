"""
Tests for smartcash.ui.setup.colab.operations.operation_manager
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from smartcash.ui.setup.colab.operations.operation_manager import ColabOperationManager


class TestColabOperationManager:
    """Test cases for ColabOperationManager."""
    
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
    def mock_operation_container(self):
        """Mock OperationContainer for testing."""
        container = Mock()
        container.log_message = Mock()
        return container
    
    @pytest.fixture
    def operation_manager(self, mock_config, mock_operation_container):
        """Create ColabOperationManager instance for testing."""
        with patch('smartcash.ui.setup.colab.operations.operation_manager.InitOperation'), \
             patch('smartcash.ui.setup.colab.operations.operation_manager.DriveMountOperation'), \
             patch('smartcash.ui.setup.colab.operations.operation_manager.SymlinkOperation'), \
             patch('smartcash.ui.setup.colab.operations.operation_manager.FoldersOperation'), \
             patch('smartcash.ui.setup.colab.operations.operation_manager.ConfigSyncOperation'), \
             patch('smartcash.ui.setup.colab.operations.operation_manager.EnvSetupOperation'), \
             patch('smartcash.ui.setup.colab.operations.operation_manager.VerifyOperation'):
            
            manager = ColabOperationManager(
                config=mock_config,
                operation_container=mock_operation_container
            )
            return manager
    
    def test_operation_manager_creation(self, operation_manager):
        """Test ColabOperationManager can be created."""
        assert operation_manager is not None
        assert operation_manager.config is not None
        assert len(operation_manager.setup_stages) == 7
        assert operation_manager.current_stage == 0
        assert len(operation_manager.operations) == 7
    
    def test_get_operations(self, operation_manager):
        """Test get_operations returns correct operations."""
        operations = operation_manager.get_operations()
        
        expected_operations = [
            'init', 'drive', 'symlink', 'folders', 
            'config', 'env', 'verify', 'full_setup', 'post_init_check'
        ]
        
        for op in expected_operations:
            assert op in operations
            assert callable(operations[op])
    
    def test_individual_operation_delegation(self, operation_manager):
        """Test that individual operations are properly delegated."""
        # Mock the individual operations
        for op_name, op_instance in operation_manager.operations.items():
            op_instance.execute_init = Mock(return_value={'success': True})
            op_instance.execute_mount_drive = Mock(return_value={'success': True})
            op_instance.execute_create_symlinks = Mock(return_value={'success': True})
            op_instance.execute_create_folders = Mock(return_value={'success': True})
            op_instance.execute_sync_configs = Mock(return_value={'success': True})
            op_instance.execute_setup_environment = Mock(return_value={'success': True})
            op_instance.execute_verify_setup = Mock(return_value={'success': True})
        
        # Test init operation
        result = operation_manager._init_operation()
        assert result['success'] is True
        operation_manager.operations['init'].execute_init.assert_called_once()
        
        # Test drive operation
        result = operation_manager._drive_mount_operation()
        assert result['success'] is True
        operation_manager.operations['drive'].execute_mount_drive.assert_called_once()
        
        # Test symlink operation
        result = operation_manager._symlink_operation()
        assert result['success'] is True
        operation_manager.operations['symlink'].execute_create_symlinks.assert_called_once()
    
    @patch('smartcash.ui.setup.colab.operations.operation_manager.STAGE_WEIGHTS')
    def test_full_setup_operation_success(self, mock_stage_weights, operation_manager):
        """Test successful full setup operation."""
        # Mock stage weights
        mock_stage_weights.get.return_value = 15  # 15% per stage
        
        # Mock all individual operations to succeed
        for op_name, op_instance in operation_manager.operations.items():
            op_instance.execute_init = Mock(return_value={'success': True, 'message': 'Init success'})
            op_instance.execute_mount_drive = Mock(return_value={'success': True, 'message': 'Drive success'})
            op_instance.execute_create_symlinks = Mock(return_value={'success': True, 'message': 'Symlink success'})
            op_instance.execute_create_folders = Mock(return_value={'success': True, 'message': 'Folders success'})
            op_instance.execute_sync_configs = Mock(return_value={'success': True, 'message': 'Config success'})
            op_instance.execute_setup_environment = Mock(return_value={'success': True, 'message': 'Env success'})
            op_instance.execute_verify_setup = Mock(return_value={'success': True, 'message': 'Verify success'})
        
        progress_callback = Mock()
        result = operation_manager._full_setup_operation(progress_callback)
        
        assert result['success'] is True
        assert 'stage_results' in result
        assert len(result['stage_results']) == 7
        assert 'Environment setup completed successfully' in result['message']
        
        # Verify progress callback was called
        assert progress_callback.call_count > 0
        progress_callback.assert_any_call(100, "🎉 Environment setup completed!")
    
    @patch('smartcash.ui.setup.colab.operations.operation_manager.STAGE_WEIGHTS')
    def test_full_setup_operation_failure(self, mock_stage_weights, operation_manager):
        """Test full setup operation with stage failure."""
        mock_stage_weights.get.return_value = 15
        
        # Mock init to succeed, drive to fail
        operation_manager.operations['init'].execute_init = Mock(
            return_value={'success': True, 'message': 'Init success'}
        )
        operation_manager.operations['drive'].execute_mount_drive = Mock(
            return_value={'success': False, 'error': 'Drive mount failed'}
        )
        
        result = operation_manager._full_setup_operation()
        
        assert result['success'] is False
        assert result['failed_stage'] == 'drive'
        assert 'Setup failed at stage \'drive\'' in result['error']
        assert 'stage_results' in result
        assert 'init' in result['stage_results']
        assert 'drive' in result['stage_results']
    
    def test_full_setup_operation_exception(self, operation_manager):
        """Test full setup operation with exception."""
        # Force exception by making setup_stages invalid
        operation_manager.setup_stages = None
        
        result = operation_manager._full_setup_operation()
        
        assert result['success'] is False
        assert 'Full setup failed' in result['error']
    
    def test_post_init_check(self, operation_manager):
        """Test post-initialization check."""
        # Mock folder verification
        operation_manager.operations['folders'].verify_folders = Mock(
            return_value={'missing_folders': ['/content/missing_folder']}
        )
        
        # Mock config verification
        operation_manager.operations['config'].check_config_integrity = Mock(
            return_value={'missing_configs': ['missing_config.yaml']}
        )
        
        # Mock config sync
        operation_manager.operations['config'].execute_sync_configs = Mock(
            return_value={'success': True, 'synced_count': 1}
        )
        
        progress_callback = Mock()
        result = operation_manager._post_init_check(progress_callback)
        
        assert result['success'] is True
        assert len(result['missing_folders']) == 1
        assert len(result['missing_configs']) == 1
        assert result['config_sync'] is not None
        
        # Verify progress callbacks
        progress_callback.assert_any_call(20, "🔍 Checking folders and configs integrity...")
        progress_callback.assert_any_call(100, "✅ Post-init check complete")
    
    def test_update_config(self, operation_manager):
        """Test configuration update."""
        new_config = {
            'environment': {
                'type': 'local',
                'project_name': 'NewProject'
            }
        }
        
        operation_manager.update_config(new_config)
        
        assert operation_manager.config == new_config
        
        # Verify all operations got the new config
        for operation in operation_manager.operations.values():
            assert operation.config == new_config
    
    def test_get_stage_status(self, operation_manager):
        """Test stage status information."""
        operation_manager.current_stage = 3
        
        status = operation_manager.get_stage_status()
        
        assert status['current_stage'] == 3
        assert status['current_stage_name'] == 'folders'
        assert status['total_stages'] == 7
        assert status['completed_stages'] == 3
        assert status['remaining_stages'] == 4
        assert status['progress_percent'] == pytest.approx(42.86, rel=0.01)
    
    def test_get_stage_status_complete(self, operation_manager):
        """Test stage status when setup is complete."""
        operation_manager.current_stage = 7
        
        status = operation_manager.get_stage_status()
        
        assert status['current_stage'] == 7
        assert status['current_stage_name'] == 'complete'
        assert status['progress_percent'] == 100.0
    
    def test_setup_stages_order(self, operation_manager):
        """Test that setup stages are in correct order."""
        expected_stages = [
            'init', 'drive', 'symlink', 'folders', 'config', 'env', 'verify'
        ]
        
        assert operation_manager.setup_stages == expected_stages
    
    def test_operations_mapping(self, operation_manager):
        """Test that all operations are properly mapped."""
        operations = operation_manager.get_operations()
        
        # Test that each stage has a corresponding operation
        for stage in operation_manager.setup_stages:
            assert stage in operations
        
        # Test additional operations
        assert 'full_setup' in operations
        assert 'post_init_check' in operations


class TestColabOperationManagerIntegration:
    """Integration tests for ColabOperationManager."""
    
    @pytest.fixture
    def real_config(self):
        """Real configuration for integration testing."""
        return {
            'environment': {
                'type': 'local',  # Use local to avoid Colab dependencies
                'auto_mount_drive': False,
                'project_name': 'SmartCash'
            }
        }
    
    def test_integration_local_environment_init(self, real_config):
        """Integration test for local environment initialization."""
        with patch('smartcash.ui.setup.colab.operations.init_operation.detect_environment_info') as mock_detect:
            mock_detect.return_value = {
                'runtime': {'type': 'local', 'display': 'Local Environment'},
                'os': {'system': 'Darwin', 'release': '21.6.0'},
                'total_ram': 17179869184,
                'cpu_cores': 8,
                'gpu': 'No GPU available',
                'is_colab': False,
                'drive_mounted': False
            }
            
            manager = ColabOperationManager(config=real_config)
            result = manager._init_operation()
            
            assert result['success'] is True
            assert result['environment'] == 'local'
    
    def test_integration_config_validation_flow(self, real_config):
        """Integration test for configuration validation flow."""
        manager = ColabOperationManager(config=real_config)
        
        # Test with valid config
        result = manager._init_operation()
        assert result['success'] is True
        
        # Test with invalid config
        manager.config = {}
        result = manager._init_operation()
        assert result['success'] is False
        assert 'Configuration validation failed' in result['error']