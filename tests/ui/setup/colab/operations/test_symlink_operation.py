"""
Tests for smartcash.ui.setup.colab.operations.symlink_operation
"""

import pytest
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from smartcash.ui.setup.colab.operations.symlink_operation import SymlinkOperation


class TestSymlinkOperation:
    """Test cases for SymlinkOperation."""
    
    @pytest.fixture
    def mock_config_colab(self):
        """Mock configuration for Colab environment."""
        return {
            'environment': {
                'type': 'colab'
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
    def symlink_operation_colab(self, mock_config_colab):
        """Create SymlinkOperation instance for Colab testing."""
        return SymlinkOperation(config=mock_config_colab)
    
    @pytest.fixture
    def symlink_operation_local(self, mock_config_local):
        """Create SymlinkOperation instance for local testing."""
        return SymlinkOperation(config=mock_config_local)
    
    @pytest.fixture
    def mock_symlink_map(self):
        """Mock SYMLINK_MAP for testing."""
        return {
            '/content/drive/MyDrive/SmartCash/data': '/content/data',
            '/content/drive/MyDrive/SmartCash/models': '/content/models',
            '/content/drive/MyDrive/SmartCash/configs': '/content/configs'
        }
    
    @pytest.fixture
    def mock_source_directories(self):
        """Mock SOURCE_DIRECTORIES for testing."""
        return [
            '/content/drive/MyDrive/SmartCash/data',
            '/content/drive/MyDrive/SmartCash/models',
            '/content/drive/MyDrive/SmartCash/configs'
        ]
    
    def test_symlink_operation_creation(self, symlink_operation_colab):
        """Test SymlinkOperation can be created."""
        assert symlink_operation_colab is not None
        assert symlink_operation_colab.config is not None
        assert symlink_operation_colab.config['environment']['type'] == 'colab'
    
    def test_get_operations(self, symlink_operation_colab):
        """Test get_operations returns correct operations."""
        operations = symlink_operation_colab.get_operations()
        assert 'create_symlinks' in operations
        assert callable(operations['create_symlinks'])
    
    def test_execute_create_symlinks_not_colab(self, symlink_operation_local):
        """Test symlink creation fails when not in Colab environment."""
        result = symlink_operation_local.execute_create_symlinks()
        
        assert result['success'] is False
        assert 'only created in Colab environment' in result['error']
    
    @patch('os.path.exists')
    def test_execute_create_symlinks_drive_not_mounted(self, mock_exists, symlink_operation_colab):
        """Test symlink creation when Drive is not mounted."""
        # Mock Drive not mounted
        mock_exists.return_value = False
        
        result = symlink_operation_colab.execute_create_symlinks()
        
        assert result['success'] is False
        assert 'Google Drive must be mounted' in result['error']
    
    @patch('smartcash.ui.setup.colab.operations.symlink_operation.SYMLINK_MAP')
    @patch('smartcash.ui.setup.colab.operations.symlink_operation.SOURCE_DIRECTORIES')
    @patch('os.symlink')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('os.path.dirname')
    def test_execute_create_symlinks_success(self, mock_dirname, mock_islink, mock_exists, 
                                           mock_makedirs, mock_symlink, mock_source_dirs, 
                                           mock_symlink_map, symlink_operation_colab):
        """Test successful symlink creation."""
        # Setup mocks
        mock_symlink_map.__iter__ = Mock(return_value=iter({
            '/content/drive/MyDrive/SmartCash/data': '/content/data',
            '/content/drive/MyDrive/SmartCash/models': '/content/models'
        }.items()))
        mock_symlink_map.items.return_value = [
            ('/content/drive/MyDrive/SmartCash/data', '/content/data'),
            ('/content/drive/MyDrive/SmartCash/models', '/content/models')
        ]
        mock_symlink_map.__len__ = Mock(return_value=2)
        
        mock_source_dirs.__iter__ = Mock(return_value=iter([
            '/content/drive/MyDrive/SmartCash/data',
            '/content/drive/MyDrive/SmartCash/models'
        ]))
        
        # Mock Drive mounted and source directories exist
        def mock_exists_side_effect(path):
            if path == '/content/drive/MyDrive':
                return True
            elif 'drive/MyDrive/SmartCash' in path:
                return True  # Source directories exist
            elif path in ['/content/data', '/content/models']:
                return False  # Target doesn't exist
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_islink.return_value = False
        mock_dirname.side_effect = lambda x: '/content'
        
        progress_callback = Mock()
        result = symlink_operation_colab.execute_create_symlinks(progress_callback)
        
        assert result['success'] is True
        assert result['total_count'] == 2
        assert result['verified_count'] == 2
        assert len(result['symlinks_created']) == 2
        assert len(result['symlinks_failed']) == 0
        
        # Verify symlink creation was called
        assert mock_symlink.call_count == 2
        
        # Check progress callbacks
        progress_callback.assert_any_call(5, "🔍 Checking Drive mount...")
        progress_callback.assert_any_call(100, "✅ Created 2/2 symlinks")
    
    @patch('smartcash.ui.setup.colab.operations.symlink_operation.SOURCE_DIRECTORIES')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_check_source_directories_missing(self, mock_makedirs, mock_exists, 
                                            mock_source_dirs, symlink_operation_colab):
        """Test checking and creating missing source directories."""
        mock_source_dirs.__iter__ = Mock(return_value=iter([
            '/content/drive/MyDrive/SmartCash/data',
            '/content/drive/MyDrive/SmartCash/models'
        ]))
        
        # Mock first directory missing, second exists
        def mock_exists_side_effect(path):
            if path == '/content/drive/MyDrive/SmartCash/data':
                return False
            elif path == '/content/drive/MyDrive/SmartCash/models':
                return True
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        
        missing = symlink_operation_colab._check_source_directories()
        
        assert len(missing) == 1
        assert '/content/drive/MyDrive/SmartCash/data' in missing
    
    @patch('os.makedirs')
    def test_create_missing_directories_success(self, mock_makedirs, symlink_operation_colab):
        """Test successful creation of missing directories."""
        missing_dirs = ['/content/drive/MyDrive/SmartCash/data']
        
        symlink_operation_colab._create_missing_directories(missing_dirs)
        
        mock_makedirs.assert_called_once_with('/content/drive/MyDrive/SmartCash/data', exist_ok=True)
    
    @patch('os.makedirs')
    def test_create_missing_directories_failure(self, mock_makedirs, symlink_operation_colab):
        """Test handling of directory creation failure."""
        mock_makedirs.side_effect = PermissionError("Permission denied")
        missing_dirs = ['/content/drive/MyDrive/SmartCash/data']
        
        # Should not raise exception, just log error
        symlink_operation_colab._create_missing_directories(missing_dirs)
        
        mock_makedirs.assert_called_once()
    
    @patch('smartcash.ui.setup.colab.operations.symlink_operation.SYMLINK_MAP')
    @patch('os.path.exists')
    def test_execute_create_symlinks_exception_handling(self, mock_exists, mock_symlink_map, 
                                                       symlink_operation_colab):
        """Test exception handling during symlink creation."""
        # Mock Drive mounted
        mock_exists.return_value = True
        
        # Force exception by making SYMLINK_MAP.items() raise
        mock_symlink_map.items.side_effect = Exception("Symlink creation failed")
        
        result = symlink_operation_colab.execute_create_symlinks()
        
        assert result['success'] is False
        assert 'Symlink creation failed' in result['error']
    
    @patch('smartcash.ui.setup.colab.operations.symlink_operation.SYMLINK_MAP')
    @patch('smartcash.ui.setup.colab.operations.symlink_operation.SOURCE_DIRECTORIES')
    @patch('os.symlink')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.path.islink')
    @patch('shutil.rmtree')
    @patch('os.remove')
    @patch('os.unlink')
    @patch('os.path.dirname')
    @patch('os.path.isdir')
    def test_execute_create_symlinks_replace_existing(self, mock_isdir, mock_dirname, mock_unlink, 
                                                     mock_remove, mock_rmtree, mock_islink, 
                                                     mock_exists, mock_makedirs, mock_symlink, 
                                                     mock_source_dirs, mock_symlink_map, 
                                                     symlink_operation_colab):
        """Test symlink creation when replacing existing files/directories."""
        # Setup mocks
        mock_symlink_map.__iter__ = Mock(return_value=iter({
            '/content/drive/MyDrive/SmartCash/data': '/content/data'
        }.items()))
        mock_symlink_map.items.return_value = [
            ('/content/drive/MyDrive/SmartCash/data', '/content/data')
        ]
        mock_symlink_map.__len__ = Mock(return_value=1)
        
        mock_source_dirs.__iter__ = Mock(return_value=iter([
            '/content/drive/MyDrive/SmartCash/data'
        ]))
        
        # Mock target exists as directory
        def mock_exists_side_effect(path):
            if path == '/content/drive/MyDrive':
                return True
            elif path == '/content/drive/MyDrive/SmartCash/data':
                return True
            elif path == '/content/data':
                return True  # Target exists
            return False
        
        mock_exists.side_effect = mock_exists_side_effect
        mock_islink.return_value = False  # Not a symlink
        mock_isdir.return_value = True  # Is a directory
        mock_dirname.return_value = '/content'
        
        result = symlink_operation_colab.execute_create_symlinks()
        
        assert result['success'] is True
        # Verify existing directory was removed
        mock_rmtree.assert_called_once_with('/content/data')
        # Verify symlink was created
        mock_symlink.assert_called_once_with('/content/drive/MyDrive/SmartCash/data', '/content/data')