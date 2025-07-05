"""
File: tests/unit/ui/setup/dependency/test_dependency_initializer.py
Unit tests for DependencyInitializer class
"""
import pytest
from unittest.mock import MagicMock, patch, call, ANY

# Import the actual class we're testing
from smartcash.ui.setup.dependency.dependency_initializer import DependencyInitializer

class TestDependencyInitializer:
    """Test cases for DependencyInitializer"""
    
    def test_initialization(self, dependency_initializer):
        """Test that DependencyInitializer initializes correctly"""
        # Verify the instance was created with correct attributes
        assert isinstance(dependency_initializer, DependencyInitializer)
        assert dependency_initializer.module_name == 'dependency'
        assert dependency_initializer.parent_module == 'setup'
        assert hasattr(dependency_initializer, '_ui_components')
        assert hasattr(dependency_initializer, '_operation_handlers')
        assert hasattr(dependency_initializer, '_current_operation')
        assert hasattr(dependency_initializer, '_current_packages')
        
    def test_initialize_success(self, dependency_initializer, mock_ui_handler, mock_operation_factory, mock_default_config):
        """Test successful initialization of dependency module"""
        # Setup test config
        test_config = {
            'packages': ['numpy', 'pandas'],
            'operation': 'install',
            'auto_install': True
        }
        
        # Execute
        result = dependency_initializer.initialize(config=test_config)
        
        # Verify the result structure
        assert result['status'] == 'success'
        assert 'ui' in result
        assert 'handlers' in result
        
        # Verify UI components were initialized
        mock_ui_handler.initialize_ui_components.assert_called_once()
        
        # Verify operation handler was created
        mock_operation_factory.create_handler.assert_called_once_with(
            operation_type='install',
            ui_handler=mock_ui_handler,
            config=test_config
        )
    
    def test_initialize_with_default_config(self, dependency_initializer, mock_ui_handler, mock_operation_factory, mock_default_config):
        """Test initialization with default config when none provided"""
        # Execute with no config
        result = dependency_initializer.initialize()
        
        # Verify the result structure
        assert result['status'] == 'success'
        assert 'ui' in result
        assert 'handlers' in result
        
        # Verify default config was used
        mock_operation_factory.create_handler.assert_called_once_with(
            operation_type='install',
            ui_handler=mock_ui_handler,
            config=mock_default_config
        )
    
    def test_get_default_config(self, dependency_initializer, mock_default_config):
        """Test getting default configuration"""
        # Get default config
        config = dependency_initializer.get_default_config()
        
        # Verify the structure and content
        assert config == mock_default_config
        assert isinstance(config, dict)
        assert 'packages' in config
        assert 'operation' in config
        assert 'auto_install' in config
        assert isinstance(config['packages'], list)
        assert config['operation'] in ['install', 'uninstall', 'update']
    
    def test_cleanup(self, dependency_initializer, mock_ui_handler):
        """Test cleanup of resources"""
        # Setup test data
        dependency_initializer._current_operation = 'install'
        dependency_initializer._current_packages = ['numpy', 'pandas']
        dependency_initializer._operation_handlers = {'install': MagicMock()}
        
        # Execute cleanup
        dependency_initializer.cleanup()
        
        # Verify cleanup was performed
        assert dependency_initializer._current_operation is None
        assert dependency_initializer._current_packages is None
        assert not dependency_initializer._operation_handlers
        mock_ui_handler.cleanup.assert_called_once()
    
    def test_initialize_error_handling(self, dependency_initializer, mock_ui_handler, mock_operation_factory):
        """Test error handling during initialization"""
        # Setup error
        test_error = Exception("Test error")
        mock_operation_factory.create_handler.side_effect = test_error
        
        # Execute and verify error handling
        with patch.object(dependency_initializer.logger, 'error') as mock_logger:
            result = dependency_initializer.initialize()
            
            # Verify error was logged
            mock_logger.assert_called_once_with("Gagal menginisialisasi modul dependency", exc_info=test_error)
        
        # Verify error response
        assert result['status'] == 'error'
        assert 'error' in result
        assert 'Test error' in str(result['error'])
        
        # Verify UI error handling
        mock_ui_handler.show_error.assert_called_once_with("Gagal menginisialisasi modul dependency")
        
    def test_operation_execution(self, dependency_initializer, mock_operation_handler):
        """Test executing an operation"""
        # Setup test data
        test_packages = ['numpy', 'pandas']
        test_operation = 'install'
        
        # Add operation handler
        dependency_initializer._operation_handlers[test_operation] = mock_operation_handler
        
        # Execute operation
        result = dependency_initializer._execute_operation(test_operation, test_packages)
        
        # Verify operation was executed
        mock_operation_handler.execute.assert_called_once_with(test_packages)
        assert result == mock_operation_handler.execute.return_value
