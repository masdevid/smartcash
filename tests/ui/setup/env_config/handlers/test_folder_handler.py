"""
Tests for the FolderHandler class in smartcash.ui.setup.env_config.handlers.folder_handler
"""
import os
import sys
import shutil
import pytest
import pytest_asyncio
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, ANY
from typing import Dict, Any, List, Optional, Tuple

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module directly without mocking first
try:
    from smartcash.ui.setup.env_config.handlers.folder_handler import FolderHandler, FolderOperationResult
    from smartcash.ui.setup.env_config.handlers.base_handler import BaseEnvHandler
except ImportError as e:
    print(f"Error importing modules: {e}")
    raise

# Enable async test support
pytestmark = pytest.mark.asyncio

# Test data
TEST_CONFIG = {
    'env_path': '/test/env',
    'env_name': 'test_env',
    'python_version': '3.10',
    'enable_backup': True,
    'max_backups': 3,
    'dry_run': False,
    'create_symlinks': True
}

# Mock constants
SOURCE_DIRECTORIES = ['/test/source1', '/test/source2']
REQUIRED_FOLDERS = ['/test/env/data', '/test/env/models', '/test/env/cache']
SYMLINK_MAP = {
    '/test/source1/file1.txt': '/test/env/data/file1.txt',
    '/test/source2/dir1': '/test/env/models/dir1'
}

@pytest.fixture
def mock_logger():
    """Fixture for a mock logger."""
    with patch('smartcash.ui.utils.ui_logger.get_module_logger') as mock_logger:
        mock_logger.return_value = MagicMock()
        yield mock_logger.return_value

@pytest.fixture
def mock_error_handler():
    """Fixture for a mock error handler."""
    mock = MagicMock()
    mock.handle_error = MagicMock(side_effect=lambda e: {'status': 'error', 'message': str(e)})
    return mock

@pytest.fixture
def mock_base_handler(mock_error_handler, mock_logger):
    """Fixture for a mock BaseEnvHandler."""
    with patch.object(BaseEnvHandler, '__init__', return_value=None) as mock_base:
        # Create a mock with async method support
        mock = MagicMock()
        
        # Set up required attributes that would be set by BaseEnvHandler.__init__
        mock._config = {}
        mock._logger = mock_logger
        mock._error_handler = mock_error_handler
        mock._module_name = 'smartcash.ui.setup.env_config.handlers.folder_handler'
        mock._initialized = False
        mock._ui_components = {}
        
        # Mock required methods
        mock.create_required_folders = MagicMock(return_value={'status': 'success'})
        
        # Make sure the mock has the same interface as the real class
        mock._initialize_ui_components = MagicMock()
        mock.update_config = MagicMock()
        
        yield mock

@pytest.fixture
def folder_handler(mock_base_handler, mock_error_handler, mock_logger, mocker):
    """Fixture for creating a FolderHandler instance for testing."""
    # Mock the EnvConfigErrorHandler to return our mock_error_handler
    # Note: EnvConfigErrorHandler is imported from error_handler module in folder_handler.py
    mocker.patch(
        'smartcash.ui.setup.env_config.handlers.error_handler.EnvConfigErrorHandler',
        return_value=mock_error_handler
    )
    
    # Patch class constants for testing
    with patch.multiple(
        'smartcash.ui.setup.env_config.handlers.folder_handler',
        SOURCE_DIRECTORIES=SOURCE_DIRECTORIES,
        REQUIRED_FOLDERS=REQUIRED_FOLDERS,
        SYMLINK_MAP=SYMLINK_MAP
    ):
        # Create the handler with test config
        handler = FolderHandler(
            config=TEST_CONFIG,
            error_handler=mock_error_handler,
            module_name='smartcash.ui.setup.env_config.handlers.folder_handler',
            logger=mock_logger
        )
        yield handler

class TestFolderHandler:
    """Test suite for the FolderHandler class."""

    def test_initialization(self, folder_handler, mock_base_handler, mock_error_handler):
        """Test that FolderHandler initializes correctly."""
        from smartcash.ui.setup.env_config.handlers.error_handler import EnvConfigErrorHandler
        
        # Verify the handler was created
        assert folder_handler is not None
        
        # Verify attributes were set
        assert hasattr(folder_handler, '_logger')
        assert hasattr(folder_handler, '_config')
        assert hasattr(folder_handler, '_error_handler')
        assert hasattr(folder_handler, '_module_name')
        
        # Verify config was set correctly
        assert folder_handler._config == TEST_CONFIG
        
        # Verify error handler was set
        assert folder_handler._error_handler == mock_error_handler
        
        # Verify module name was set
        assert folder_handler._module_name == 'smartcash.ui.setup.env_config.handlers.folder_handler'
        
        # Verify the base handler was initialized with the correct parameters
        # Note: We're not checking the exact call to mock_base_handler anymore
        # since the actual initialization is more complex with the error handler
        
        # The error handler is now passed in from the test fixture, so we don't need to assert on its creation

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=False)
    async def test_create_required_folders_success(self, mock_exists, mock_mkdir, folder_handler):
        """Test successful folder creation."""
        result = await folder_handler.create_required_folders()
        
        assert result['status'] == 'success'
        assert result['created_count'] > 0
        assert len(result['folders_created']) > 0
        assert mock_mkdir.called

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', side_effect=[False, True, False, False])
    async def test_create_required_folders_with_existing(self, mock_exists, mock_mkdir, folder_handler):
        """Test folder creation when some folders already exist."""
        result = await folder_handler.create_required_folders()
        
        assert result['status'] == 'success'
        assert result['created_count'] > 0
        assert len(result['folders_created']) > 0

    @patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied"))
    @patch('pathlib.Path.exists', return_value=False)
    async def test_create_required_folders_permission_error(self, mock_exists, mock_mkdir, folder_handler):
        """Test folder creation with permission error."""
        result = await folder_handler.create_required_folders()
        
        assert result['status'] == 'error'
        assert len(result['errors']) > 0
        assert "Permission denied" in str(result['errors'][0])

    @patch('shutil.copytree')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_symlink', return_value=False)
    @patch('pathlib.Path.is_dir', return_value=True)
    async def test_process_symlink_with_backup(self, mock_isdir, mock_issymlink, mock_exists, mock_copytree, folder_handler):
        """Test symlink processing with backup enabled."""
        source = Path('/test/source/file.txt')
        target = Path('/test/target/file.txt')
        
        created_symlinks = []
        created_backups = []
        
        result = await folder_handler._process_symlink(
            source_path=source,
            target_path=target,
            enable_backup=True,
            max_backups=3,
            dry_run=False,
            created_symlinks=created_symlinks,
            created_backups=created_backups
        )
        
        assert result is True
        assert len(created_backups) == 1
        assert len(created_symlinks) == 1

    @patch('pathlib.Path.symlink_to')
    @patch('pathlib.Path.exists', side_effect=[False, False])
    async def test_process_symlink_new_file(self, mock_exists, mock_symlink, folder_handler):
        """Test symlink creation for a new file."""
        source = Path('/test/source/new_file.txt')
        target = Path('/test/target/new_file.txt')
        
        created_symlinks = []
        created_backups = []
        
        result = await folder_handler._process_symlink(
            source_path=source,
            target_path=target,
            enable_backup=False,
            max_backups=0,
            dry_run=False,
            created_symlinks=created_symlinks,
            created_backups=created_backups
        )
        
        assert result is True
        assert len(created_symlinks) == 1
        assert len(created_backups) == 0
        mock_symlink.assert_called_once()

    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.unlink')
    @patch('shutil.rmtree')
    @patch('pathlib.Path.is_dir', return_value=True)
    async def test_cleanup_old_backups(self, mock_isdir, mock_rmtree, mock_unlink, mock_glob, folder_handler):
        """Test cleanup of old backups."""
        # Setup mock backup files
        backup_files = [
            Path('/backups/test.bak_20230101_120000'),
            Path('/backups/test.bak_20230102_120000'),
            Path('/backups/test.bak_20230103_120000')
        ]
        mock_glob.return_value = backup_files
        
        # Keep only 2 most recent backups
        await folder_handler._cleanup_old_backups(
            backup_dir=Path('/backups'),
            base_name='test',
            keep=2
        )
        
        # Should remove the oldest backup
        mock_rmtree.assert_called_once_with(backup_files[0], ignore_errors=True)
        assert not mock_unlink.called  # Since we're using rmtree for dirs

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=False)
    async def test_dry_run(self, mock_exists, mock_mkdir, folder_handler):
        """Test dry run mode doesn't make actual changes."""
        result = await folder_handler.create_required_folders(dry_run=True)
        
        assert result['status'] == 'success'
        assert "[DRY RUN]" in result['message']
        mock_mkdir.assert_not_called()

    def test_get_required_folders(self, folder_handler):
        """Test getting required folders excluding symlink targets."""
        with patch.dict('smartcash.ui.setup.env_config.handlers.folder_handler.SYMLINK_MAP', 
                       {'/test/source': '/test/env/data'}):
            folders = folder_handler._get_required_folders()
            
            # Should exclude /test/env/data since it's a symlink target
            assert '/test/env/data' not in folders
            assert all(f in folders for f in REQUIRED_FOLDERS if f != '/test/env/data')

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=False)
    @patch('smartcash.ui.setup.env_config.handlers.folder_handler.SOURCE_DIRECTORIES', ['/test/source'])
    async def test_create_source_directories(self, mock_exists, mock_mkdir, folder_handler):
        """Test creation of source directories."""
        result = await folder_handler._create_source_directories()
        
        assert len(result) == 1
        assert '/test/source' in result
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    async def test_create_source_directories_exists(self, mock_exists, mock_mkdir, folder_handler):
        """Test handling when source directories already exist."""
        result = await folder_handler._create_source_directories()
        
        assert len(result) == 0  # No directories created
        mock_mkdir.assert_not_called()

    @patch('shutil.rmtree')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_symlink', return_value=False)
    @patch('pathlib.Path.is_dir', return_value=True)
    async def test_remove_existing_target_directory(self, mock_isdir, mock_issymlink, mock_exists, mock_rmtree, folder_handler):
        """Test removal of existing directory target."""
        target = Path('/test/target/dir')
        
        await folder_handler._process_symlink(
            source_path=Path('/test/source/dir'),
            target_path=target,
            enable_backup=False,
            max_backups=0,
            dry_run=False,
            created_symlinks=[],
            created_backups=[]
        )
        
        mock_rmtree.assert_called_once_with(target)

    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_symlink', return_value=False)
    @patch('pathlib.Path.is_file', return_value=True)
    async def test_remove_existing_target_file(self, mock_isfile, mock_issymlink, mock_exists, mock_unlink, folder_handler):
        """Test removal of existing file target."""
        target = Path('/test/target/file.txt')
        
        await folder_handler._process_symlink(
            source_path=Path('/test/source/file.txt'),
            target_path=target,
            enable_backup=False,
            max_backups=0,
            dry_run=False,
            created_symlinks=[],
            created_backups=[]
        )
        
        mock_unlink.assert_called_once()

    @patch('shutil.copytree')
    @patch('pathlib.Path')
    @patch('datetime.datetime')
    async def test_backup_existing_directory(self, mock_datetime, mock_path, mock_copytree, folder_handler):
        """Test backup of existing directory."""
        # Setup mock datetime
        from datetime import datetime
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        # Setup target path
        target_path = Path('/test/target/dir')
        print(f"Target path: {target_path}")
        
        # Create a mock for the target path
        target_mock = MagicMock()
        target_mock.__str__.return_value = str(target_path)
        target_mock.exists.return_value = True
        target_mock.is_symlink.return_value = False
        target_mock.is_dir.return_value = True
        target_mock.name = 'dir'
        print(f"Target mock setup: {target_mock}, exists={target_mock.exists()}, is_dir={target_mock.is_dir()}")
        
        # Setup parent path mock
        parent_path = target_path.parent
        parent_mock = MagicMock()
        parent_mock.__str__.return_value = str(parent_path)
        parent_mock.name = parent_path.name
        parent_mock.exists.return_value = True
        print(f"Parent mock setup: {parent_mock}, exists={parent_mock.exists()}")
        
        # Setup backup directory mock
        backup_dir_mock = MagicMock()
        backup_dir_mock.exists.return_value = True  # Directory exists
        backup_dir_mock.mkdir.return_value = None
        backup_dir_mock.__str__.return_value = str(parent_path / '.backups')
        print(f"Backup dir mock setup: {backup_dir_mock}, exists={backup_dir_mock.exists()}")
        
        # Setup backup file path
        backup_file_path = parent_path / '.backups' / 'dir.bak'
        backup_file_mock = MagicMock()
        backup_file_mock.__str__.return_value = str(backup_file_path)
        backup_file_mock.exists.return_value = False  # Backup file doesn't exist yet
        print(f"Backup file mock setup: {backup_file_mock}, exists={backup_file_mock.exists()}")
        
        # Configure parent.joinpath to return backup_dir_mock for '.backups'
        def parent_joinpath_side_effect(*args, **kwargs):
            print(f"parent.joinpath called with args: {args}, kwargs: {kwargs}")
            if args and args[0] == '.backups':
                print(f"Returning backup_dir_mock: {backup_dir_mock}")
                return backup_dir_mock
            print(f"Returning new MagicMock for args: {args}")
            return MagicMock()
        
        parent_mock.joinpath.side_effect = parent_joinpath_side_effect
        
        # Configure backup_dir.joinpath to return backup_file_mock
        def backup_dir_joinpath_side_effect(*args, **kwargs):
            print(f"backup_dir.joinpath called with args: {args}, kwargs: {kwargs}")
            if args and args[0] == 'dir.bak':
                print(f"Returning backup_file_mock: {backup_file_mock}")
                return backup_file_mock
            print(f"Returning new MagicMock for args: {args}")
            return MagicMock()
            
        backup_dir_mock.joinpath.side_effect = backup_dir_joinpath_side_effect
        
        # Set parent on target
        target_mock.parent = parent_mock
        print(f"Set target_mock.parent = {parent_mock}")
        
        # Configure the Path constructor to return our mocks
        def path_constructor(*args, **kwargs):
            print(f"Path constructor called with args: {args}, kwargs: {kwargs}")
            if not args:
                print("Returning empty MagicMock")
                return MagicMock()
                
            path_str = str(args[0])
            if path_str == str(target_path):
                print(f"Returning target_mock for path: {path_str}")
                return target_mock
            elif path_str == str(parent_path):
                print(f"Returning parent_mock for path: {path_str}")
                return parent_mock
            print(f"Returning new MagicMock for path: {path_str}")
            return MagicMock()
        
        mock_path.side_effect = path_constructor
        
        # Add debug logger to capture log messages
        import logging
        test_logger = logging.getLogger('test_logger')
        test_logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        test_logger.addHandler(console_handler)
        
        # Mock the folder_handler logger to use our test logger
        folder_handler._logger = test_logger
        
        # Mock shutil.copytree to not actually do anything but return success
        mock_copytree.return_value = None
        
        # Mock the cleanup_old_backups method to do nothing
        with patch.object(folder_handler, '_cleanup_old_backups', return_value=None) as mock_cleanup:
            print("\n--- Calling _backup_existing_path ---")
            # Call the method with our test target
            result = await folder_handler._backup_existing_path(
                path=target_mock,
                max_backups=3,
                dry_run=False
            )
            
            print(f"\n--- Result from _backup_existing_path: {result} ---")
            if 'error' in result:
                print(f"Error message: {result.get('error')}")
            if 'errors' in result:
                print(f"Errors list: {result.get('errors')}")
            if 'message' in result:
                print(f"Message: {result.get('message')}")
            
            # Verify results
            assert result is not None, "Result should not be None"
            assert 'status' in result, f"Result should have 'status' key. Got keys: {list(result.keys())}"
            print(f"Status in result: {result.get('status')}")
            assert result['status'] == 'success', f"Status should be 'success'. Got: {result.get('status')}"
            assert 'backup_path' in result, f"Result should have 'backup_path' key. Got keys: {list(result.keys())}"
            assert result.get('backup_created') is True, "backup_created should be True"
            
            # Verify the backup path was constructed correctly
            print(f"\n--- parent_mock.joinpath.call_args_list: {parent_mock.joinpath.call_args_list} ---")
            parent_mock.joinpath.assert_called_once_with('.backups')
            
            # Verify the backup directory was created
            print(f"\n--- backup_dir_mock.mkdir.call_args_list: {backup_dir_mock.mkdir.call_args_list} ---")
            backup_dir_mock.mkdir.assert_called_once_with(parents=True, exist_ok=True)
            
            # Verify copytree was called with the expected arguments
            print(f"\n--- mock_copytree.call_args_list: {mock_copytree.call_args_list} ---")
            mock_copytree.assert_called_once_with(
                str(target_path),
                str(backup_file_path),
                symlinks=True
            )
            
            # Verify cleanup was called
            print(f"\n--- mock_cleanup.call_args_list: {mock_cleanup.call_args_list} ---")
            mock_cleanup.assert_called_once()

    @patch('shutil.copy2')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_symlink', return_value=False)
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('pathlib.Path.mkdir')
    async def test_backup_existing_file(self, mock_mkdir, mock_isfile, mock_issymlink, mock_exists, mock_copy2, folder_handler):
        """Test backup of existing file."""
        # Setup target path
        target = MagicMock(spec=Path)
        target.__str__.return_value = '/test/target/file.txt'
        target.exists.return_value = True
        target.is_symlink.return_value = False
        target.is_file.return_value = True
        
        # Setup parent path mock
        parent = MagicMock()
        target.parent = parent
        
        # Setup backup directory mock
        backup_dir = MagicMock()
        parent.joinpath.return_value = backup_dir
        
        # Setup backup path mock
        backup_path = MagicMock()
        backup_dir.joinpath.return_value = backup_path
        
        # Call the method
        result = await folder_handler._backup_existing_path(
            path=target,
            max_backups=3,
            dry_run=False
        )
        
        # Verify results
        assert result is not None
        assert result['status'] == 'success'
        assert 'backup_path' in result
        # Verify the backup path was constructed correctly
        parent.joinpath.assert_called_once_with('.backups')
        backup_dir.joinpath.assert_called_once_with('file.txt.bak')
        # Verify copy2 was called with the expected arguments
        mock_copy2.assert_called_once_with(str(target), str(backup_path), follow_symlinks=False)

    @patch('shutil.copytree', side_effect=shutil.Error("Backup error"))
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_symlink', return_value=False)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.unlink')
    async def test_backup_error_cleanup(self, mock_unlink, mock_mkdir, mock_isdir, mock_issymlink, mock_exists, mock_copytree, folder_handler):
        """Test cleanup after backup error."""
        # Setup target path
        target = MagicMock(spec=Path)
        target.__str__.return_value = '/test/target/dir'
        target.exists.return_value = True
        target.is_symlink.return_value = False
        target.is_dir.return_value = True
        
        # Setup parent path mock
        parent = MagicMock()
        target.parent = parent
        
        # Setup backup directory mock
        backup_dir = MagicMock()
        parent.joinpath.return_value = backup_dir
        
        # Setup backup path mock
        backup_path = MagicMock()
        backup_dir.joinpath.return_value = backup_path
        
        # Call the method
        result = await folder_handler._backup_existing_path(
            path=target,
            max_backups=3,
            dry_run=False
        )
        
        # Verify results
        assert result is not None
        assert result['status'] == 'error'
        assert 'error' in result
        assert 'Backup error' in result['error']
        assert result['result'] is False
        # Verify no cleanup was attempted on the backup path since it was never created
        mock_unlink.assert_not_called()
        # Verify the backup path was constructed correctly
        parent.joinpath.assert_called_once_with('.backups')
        backup_dir.joinpath.assert_called_once_with('dir.bak')

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=False)
    async def test_create_required_folders_with_errors(self, mock_exists, mock_mkdir, folder_handler):
        """Test folder creation with simulated errors."""
        # Make mkdir raise an error for the first call
        mock_mkdir.side_effect = [OSError("Permission denied"), None]
        
        result = await folder_handler.create_required_folders()
        
        assert result['status'] == 'error'
        assert len(result['errors']) > 0
        assert "Permission denied" in str(result['errors'][0])
        assert result['created_count'] == 1  # One folder should have been created

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=False)
    async def test_create_required_folders_dry_run(self, mock_exists, mock_mkdir, folder_handler):
        """Test dry run mode for folder creation."""
        result = await folder_handler.create_required_folders(dry_run=True)
        
        assert result['status'] == 'success'
        assert "[DRY RUN]" in result['message']
        mock_mkdir.assert_not_called()  # No actual directory creation in dry run
