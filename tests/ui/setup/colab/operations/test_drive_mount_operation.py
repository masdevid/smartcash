"""
Tests for smartcash.ui.setup.colab.operations.drive_mount_operation
"""

import pytest
import os
from unittest.mock import Mock, patch, mock_open
from smartcash.ui.setup.colab.operations.drive_mount_operation import DriveMountOperation


class TestDriveMountOperation:
    """Test cases for DriveMountOperation."""
    
    @pytest.fixture
    def mock_config_colab(self):
        """Mock configuration for Colab environment."""
        return {
            'environment': {
                'type': 'colab',
                'auto_mount_drive': True
            }
        }
    
    @pytest.fixture
    def mock_config_local(self):
        """Mock configuration for local environment."""
        return {
            'environment': {
                'type': 'local'
            }
        }
    
    @pytest.fixture
    def drive_operation_colab(self, mock_config_colab):
        """Create DriveMountOperation instance for Colab testing."""
        return DriveMountOperation(config=mock_config_colab)
    
    @pytest.fixture
    def drive_operation_local(self, mock_config_local):
        """Create DriveMountOperation instance for local testing."""
        return DriveMountOperation(config=mock_config_local)
    
    def test_drive_operation_creation(self, drive_operation_colab):
        """Test DriveMountOperation can be created."""
        assert drive_operation_colab is not None
        assert drive_operation_colab.config is not None
        assert drive_operation_colab.config['environment']['type'] == 'colab'
    
    def test_get_operations(self, drive_operation_colab):
        """Test get_operations returns correct operations."""
        operations = drive_operation_colab.get_operations()
        assert 'mount_drive' in operations
        assert callable(operations['mount_drive'])
    
    def test_execute_mount_drive_not_colab(self, drive_operation_local):
        """Test drive mount fails when not in Colab environment."""
        result = drive_operation_local.execute_mount_drive()
        
        assert result['success'] is False
        assert 'only available in Colab environment' in result['error']
    
    @patch('os.path.exists')
    def test_execute_mount_drive_already_mounted(self, mock_exists, drive_operation_colab):
        """Test drive mount when already mounted."""
        # Mock that drive is already mounted
        def mock_exists_side_effect(path):
            if path == '/content/drive':
                return True
            elif path == '/content/drive/MyDrive':
                return True
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        
        progress_callback = Mock()
        result = drive_operation_colab.execute_mount_drive(progress_callback)
        
        assert result['success'] is True
        assert result['already_mounted'] is True
        assert result['mount_path'] == '/content/drive'
        assert 'already mounted' in result['message']
        
        # Check progress callbacks
        progress_callback.assert_any_call(10, "🔍 Checking Drive mount status...")
        progress_callback.assert_any_call(100, "✅ Google Drive already mounted")
    
    @patch('smartcash.ui.setup.colab.operations.drive_mount_operation.DriveMountOperation._test_write_access')
    @patch('os.path.exists')
    @patch('google.colab.drive')
    def test_execute_mount_drive_success(self, mock_drive_module, mock_exists, mock_write_access, drive_operation_colab):
        """Test successful drive mount."""
        # Mock drive not initially mounted, then mounted after mount call
        mount_calls = [False, False, True, True]  # First two calls: not mounted, then mounted
        mock_exists.side_effect = mount_calls
        
        # Mock successful write access
        mock_write_access.return_value = True
        
        # Mock drive.mount
        mock_drive = Mock()
        mock_drive_module.mount = mock_drive
        
        progress_callback = Mock()
        result = drive_operation_colab.execute_mount_drive(progress_callback)
        
        assert result['success'] is True
        assert result['mount_path'] == '/content/drive'
        assert result['write_access'] is True
        assert 'mounted successfully' in result['message']
        
        # Verify drive.mount was called
        mock_drive.assert_called_once_with('/content/drive')
        
        # Check progress callbacks
        progress_callback.assert_any_call(30, "📁 Mounting Google Drive...")
        progress_callback.assert_any_call(70, "🔍 Verifying mount...")
        progress_callback.assert_any_call(100, "✅ Google Drive mounted successfully")
    
    @patch('os.path.exists')
    @patch('google.colab.drive')
    def test_execute_mount_drive_mount_fails(self, mock_drive_module, mock_exists, drive_operation_colab):
        """Test drive mount failure."""
        # Mock drive not mounted
        mock_exists.return_value = False
        
        # Mock drive.mount raises exception
        mock_drive = Mock()
        mock_drive.side_effect = Exception("Mount failed")
        mock_drive_module.mount = mock_drive
        
        result = drive_operation_colab.execute_mount_drive()
        
        assert result['success'] is False
        assert 'Drive mount failed' in result['error']
    
    @patch('os.path.exists')
    def test_execute_mount_drive_import_error(self, mock_exists, drive_operation_colab):
        """Test drive mount when google.colab is not available."""
        # Mock drive not mounted
        mock_exists.return_value = False
        
        # Mock ImportError for google.colab
        with patch('builtins.__import__', side_effect=ImportError("No module named 'google.colab'")):
            result = drive_operation_colab.execute_mount_drive()
            
            assert result['success'] is False
            assert 'Google Colab drive module not available' in result['error']
    
    @patch('os.path.exists')
    @patch('google.colab.drive')
    def test_execute_mount_drive_verification_fails(self, mock_drive_module, mock_exists, drive_operation_colab):
        """Test drive mount when verification fails."""
        # Mock drive mount succeeds but MyDrive doesn't exist
        def mock_exists_side_effect(path):
            if path == '/content/drive':
                return False  # Initially not mounted
            elif path == '/content/drive/MyDrive':
                return False  # MyDrive doesn't exist after mount
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        
        # Mock drive.mount
        mock_drive = Mock()
        mock_drive_module.mount = mock_drive
        
        result = drive_operation_colab.execute_mount_drive()
        
        assert result['success'] is False
        assert 'Drive mount verification failed' in result['error']
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.remove')
    def test_test_write_access_success(self, mock_remove, mock_file_open, drive_operation_colab):
        """Test successful write access test."""
        result = drive_operation_colab._test_write_access('/content/drive')
        
        assert result is True
        mock_file_open.assert_called_once()
        mock_remove.assert_called_once()
    
    @patch('builtins.open', side_effect=PermissionError("Permission denied"))
    def test_test_write_access_failure(self, mock_file_open, drive_operation_colab):
        """Test write access test failure."""
        result = drive_operation_colab._test_write_access('/content/drive')
        
        assert result is False
    
    def test_execute_mount_drive_exception_handling(self, drive_operation_colab):
        """Test exception handling during mount operation."""
        # Force an exception by passing invalid config
        drive_operation_colab.config = None
        
        result = drive_operation_colab.execute_mount_drive()
        
        assert result['success'] is False
        assert 'Drive mount operation failed' in result['error']